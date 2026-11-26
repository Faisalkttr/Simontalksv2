"""
Knowledge base for Simon: pulls a YouTube channel (RSS + transcripts) and a
blog (RSS), plus pasted tweets, chunks it on sentence boundaries, embeds it
with Gemini, and retrieves the most relevant excerpts for a query.

Changes from v1: new google-genai SDK + gemini-embedding-001, batched
embedding calls, vectorised similarity, YouTube chunks carry timestamps so
citations deep-link to the right moment, and failures are reported
(warnings / KBError) instead of silently producing an empty index.
"""

import html
import logging
import re
import time

import numpy as np
import requests
import feedparser
import streamlit as st

from settings import secret

log = logging.getLogger("simon.kb")

MAX_ITEMS_PER_SOURCE = 15
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
SIMILARITY_THRESHOLD = 0.55   # tune against your own content; embeddings differ by model
TOP_K = 4
MAX_CHUNKS_PER_SOURCE = 2     # avoid one long video crowding out everything else

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
EMBED_BATCH = 50


class KBError(Exception):
    """Indexing failed in a way the user should hear about."""


# ------------------------------------------------------------------
# Fetching (each source yields items with 'units' = [(text, start_seconds|None)])
# ------------------------------------------------------------------
def _sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    return [(s, None) for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _fetch_youtube_segments(video_id: str) -> list:
    """Works with both the legacy (<1.0) and current youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return []
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            return [(s["text"], s["start"]) for s in YouTubeTranscriptApi.get_transcript(video_id)]
        fetched = YouTubeTranscriptApi().fetch(video_id)
        return [(s.text, s.start) for s in fetched]
    except Exception as exc:  # disabled, blocked cloud IP, no captions...
        log.info("No transcript for %s: %s", video_id, type(exc).__name__)
        return []


def _extract_video_id(url: str) -> str:
    match = re.search(r"v=([\w-]{6,})", url or "")
    return match.group(1) if match else ""


def _fetch_youtube_items(channel_id: str):
    if not channel_id:
        return [], []
    warnings = []
    try:
        resp = requests.get(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}", timeout=15)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as exc:
        return [], [f"YouTube feed couldn't be loaded ({type(exc).__name__}). Check the channel ID (it starts with UC)."]

    entries = parsed.entries[:MAX_ITEMS_PER_SOURCE]
    items = []
    for entry in entries:
        video_id = entry.get("yt_videoid") or _extract_video_id(entry.get("link", ""))
        if not video_id:
            continue
        segments = _fetch_youtube_segments(video_id)
        if not segments:
            continue
        items.append(
            {
                "title": entry.get("title", "Untitled video"),
                "url": entry.get("link", f"https://www.youtube.com/watch?v={video_id}"),
                "units": [(t.replace("\n", " "), s) for t, s in segments],
                "source_type": "YouTube",
            }
        )
    if entries and len(items) < len(entries):
        warnings.append(
            f"YouTube: {len(items)} of {len(entries)} recent videos had retrievable transcripts "
            "(cloud IPs are sometimes blocked by YouTube)."
        )
    elif not entries:
        warnings.append("YouTube: the channel feed has no videos.")
    return items, warnings


def _strip_html(raw: str) -> str:
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        return html.unescape(re.sub("<[^<]+?>", "", raw or ""))


def _fetch_article_text(url: str) -> str:
    try:
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def _fetch_blog_items(rss_url: str):
    if not rss_url:
        return [], []
    try:
        resp = requests.get(rss_url, timeout=15)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as exc:
        return [], [f"Blog feed couldn't be loaded ({type(exc).__name__}). Check the RSS URL."]

    items = []
    for entry in parsed.entries[:MAX_ITEMS_PER_SOURCE]:
        text = ""
        if "content" in entry and entry.content:
            text = entry.content[0].get("value", "")
        elif "summary" in entry:
            text = entry.summary
        text = _strip_html(text)
        if len(text) < 400 and entry.get("link"):
            text = _fetch_article_text(entry.link) or text
        if not text.strip():
            continue
        items.append(
            {
                "title": entry.get("title", "Untitled post"),
                "url": entry.get("link", ""),
                "units": _sentences(text),
                "source_type": "Blog",
            }
        )
    warnings = [] if items else ["Blog: the feed loaded but no readable posts were found."]
    return items, warnings


def _parse_manual_tweets(raw_text: str) -> list:
    if not raw_text or not raw_text.strip():
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw_text) if b.strip()]
    return [
        {"title": f"Tweet {i + 1}", "url": "", "units": _sentences(block), "source_type": "Twitter/X"}
        for i, block in enumerate(blocks)
    ]


# ------------------------------------------------------------------
# Chunking: group sentences/segments, never cut mid-word
# ------------------------------------------------------------------
def _chunk_units(units: list, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list:
    """units: [(text, start)] -> [(chunk_text, start_of_first_unit)]"""
    flat = []
    for text, start in units:  # split pathological over-long units on word boundaries
        while len(text) > size:
            cut = text.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            flat.append((text[:cut], start))
            text = text[cut:].lstrip()
        if text:
            flat.append((text, start))

    chunks, cur, cur_len = [], [], 0
    for text, start in flat:
        if cur and cur_len + len(text) + 1 > size:
            chunks.append((" ".join(t for t, _ in cur), cur[0][1]))
            keep, length = [], 0
            for unit in reversed(cur):
                length += len(unit[0]) + 1
                if length > overlap:
                    break
                keep.insert(0, unit)
            cur, cur_len = keep, sum(len(t) + 1 for t, _ in keep)
        cur.append((text, start))
        cur_len += len(text) + 1
    if cur:
        chunks.append((" ".join(t for t, _ in cur), cur[0][1]))
    return chunks


def _with_timestamp(url: str, start) -> str:
    if start is None or not url or "youtube.com" not in url:
        return url
    return f"{url}{'&' if '?' in url else '?'}t={int(start)}s"


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------
def _client():
    from google import genai

    key = secret("GEMINI_API_KEY")
    if not key:
        raise KBError("GEMINI_API_KEY is missing, so Simon's sources can't be indexed.")
    return genai.Client(api_key=key)


def _embed(texts: list, task_type: str):
    """Return (matrix[n, d] of unit vectors, ok_mask[n]). Raises KBError if nothing embedded."""
    from google.genai import types

    client = _client()
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIMS)
    vectors = [None] * len(texts)
    last_error = None

    for lo in range(0, len(texts), EMBED_BATCH):
        batch = texts[lo : lo + EMBED_BATCH]
        for attempt in range(3):
            try:
                resp = client.models.embed_content(model=EMBED_MODEL, contents=batch, config=cfg)
                for i, emb in enumerate(resp.embeddings):
                    vectors[lo + i] = np.asarray(emb.values, dtype=np.float32)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(2 * (attempt + 1))
        if lo + EMBED_BATCH < len(texts):
            time.sleep(0.5)

    ok = np.array([v is not None for v in vectors])
    if not ok.any():
        raise KBError(f"Embedding failed ({type(last_error).__name__ if last_error else 'unknown'}). Check GEMINI_API_KEY and quota.")
    dim = next(v.shape[0] for v in vectors if v is not None)
    matrix = np.vstack([v if v is not None else np.zeros(dim, dtype=np.float32) for v in vectors])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, ok


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def build_knowledge_base(youtube_channel_id: str, blog_rss_url: str, manual_tweets_text: str, refresh_nonce: int):
    """Fetch, chunk, and embed all sources. Cached for an hour or until the nonce changes.
    Raises KBError (not cached) when indexing fails outright."""
    warnings = []
    raw_items = []
    for fetch, arg in ((_fetch_youtube_items, youtube_channel_id.strip()), (_fetch_blog_items, blog_rss_url.strip())):
        items, warns = fetch(arg)
        raw_items += items
        warnings += warns
    raw_items += _parse_manual_tweets(manual_tweets_text)

    records = []
    for item in raw_items:
        for text, start in _chunk_units(item["units"]):
            records.append(
                {
                    "text": text,
                    "title": item["title"],
                    "url": _with_timestamp(item["url"], start),
                    "source_type": item["source_type"],
                }
            )

    empty_stats = {"youtube": 0, "blog": 0, "twitter": 0}
    if not records:
        return {"chunks": [], "embeddings": None, "stats": empty_stats, "warnings": warnings}

    matrix, ok = _embed([r["text"] for r in records], "RETRIEVAL_DOCUMENT")
    if not ok.all():
        warnings.append(f"{int((~ok).sum())} of {len(ok)} chunks failed to embed and were skipped.")
    kept = [r for r, good in zip(records, ok) if good]
    matrix = matrix[ok]

    stats = {
        "youtube": sum(1 for c in kept if c["source_type"] == "YouTube"),
        "blog": sum(1 for c in kept if c["source_type"] == "Blog"),
        "twitter": sum(1 for c in kept if c["source_type"] == "Twitter/X"),
    }
    return {"chunks": kept, "embeddings": matrix, "stats": stats, "warnings": warnings}


def retrieval_query(user_text: str, prior_user_text: str = "") -> str:
    """Short follow-ups ('why is that?') embed badly alone; borrow the previous question."""
    if prior_user_text and len(user_text.split()) < 8:
        return f"{prior_user_text}\n{user_text}"
    return user_text


def retrieve_relevant_chunks(kb: dict, query: str, top_k: int = TOP_K) -> list:
    """[(chunk, score)] above the threshold, at most MAX_CHUNKS_PER_SOURCE per item."""
    if not kb or not kb.get("chunks") or kb.get("embeddings") is None:
        return []
    qmat, ok = _embed([query], "RETRIEVAL_QUERY")
    scores = kb["embeddings"] @ qmat[0]
    order = np.argsort(-scores)

    results, per_source = [], {}
    for idx in order:
        score = float(scores[idx])
        if score < SIMILARITY_THRESHOLD or len(results) >= top_k:
            break
        chunk = kb["chunks"][idx]
        key = (chunk["source_type"], chunk["title"])
        if per_source.get(key, 0) >= MAX_CHUNKS_PER_SOURCE:
            continue
        per_source[key] = per_source.get(key, 0) + 1
        results.append((chunk, score))
    return results


def format_grounding_block(results: list) -> str:
    if not results:
        return ""
    lines = [
        "RELEVANT EXCERPTS FROM YOUR OWN PUBLISHED WORK (YouTube, blog, and/or Twitter/X). "
        "Use them as evidence where genuinely relevant. Cite naturally in your own voice "
        '(e.g. "as I laid out in my video on X...") and include the source URL when you cite one. '
        "Do not invent content beyond what is given here, and don't force a citation if nothing here "
        "is actually relevant.",
        "<excerpts>",
    ]
    for chunk, _score in results:
        url_part = f" ({chunk['url']})" if chunk["url"] else ""
        lines.append(f'\n[{chunk["source_type"]}] "{chunk["title"]}"{url_part}')
        lines.append(chunk["text"])
    lines.append("</excerpts>")
    return "\n".join(lines)


def sources_for_display(results: list) -> list:
    """Unique per source, best score first: [{'type','title','url','score'}]."""
    seen, out = set(), []
    for chunk, score in results:
        key = (chunk["source_type"], chunk["title"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": chunk["source_type"], "title": chunk["title"], "url": chunk["url"], "score": round(score, 2)})
    return out
