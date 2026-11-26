"""Model routing and streaming.

Text turns  -> Groq (gpt-oss-120b, falling back to Llama 3.3 70B on rate limits)
Image turns -> Gemini (vision), same system prompt and same conversation history

Both stream. Each yields Chunk(kind, text):
  "reasoning"  the model's native thinking (Groq `reasoning`, Gemini thought summaries)
  "answer"     the user-facing reply
  "notice"     a short status message (e.g. "switched to backup model")
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator

from settings import HISTORY_CHAR_BUDGET, secret

log = logging.getLogger("simon.llm")

GROQ_MODEL = secret("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = secret("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = secret("GEMINI_MODEL", "gemini-2.5-flash")

IMAGE_LOOKBACK_TURNS = 3  # follow-up questions about a recent image keep using it


@dataclass
class Chunk:
    kind: str
    text: str


class LLMError(Exception):
    """Carries a message that is safe to show to the user."""


# ------------------------------------------------------------------
# Preparing the conversation
# ------------------------------------------------------------------
def _render_user_content(msg: dict) -> str:
    """User text plus any text-file attachments, wrapped as untrusted data."""
    content = msg["content"]
    for a in msg.get("attachments") or []:
        if a.get("kind") == "text" and a.get("text"):
            note = " (truncated)" if a.get("truncated") else ""
            content += f'\n\n<attached_file name="{a["name"]}"{note}>\n{a["text"]}\n</attached_file>'
        elif a.get("kind") == "image" and not a.get("data"):
            content += f'\n\n[Earlier image "{a["name"]}" is no longer available to view.]'
    return content


def prepare_turn(messages: list, budget: int = HISTORY_CHAR_BUDGET):
    """Return (history, images) for the model.

    history: [{"role", "content"}] trimmed from the oldest end to fit `budget`
             characters (the newest message is always kept).
    images:  [(bytes, mime)] attached in the last few user turns, if any.
    """
    rendered = []
    for m in messages:
        if m["role"] == "user":
            rendered.append({"role": "user", "content": _render_user_content(m)})
        else:
            rendered.append({"role": "assistant", "content": m["content"]})

    kept, used = [], 0
    for item in reversed(rendered):
        size = len(item["content"])
        if kept and used + size > budget:
            break
        kept.append(item)
        used += size
    kept.reverse()
    while len(kept) > 1 and kept[0]["role"] != "user":  # providers expect a user-first history
        kept.pop(0)

    images, seen_user_turns = [], 0
    for m in reversed(messages):
        if m["role"] != "user":
            continue
        seen_user_turns += 1
        if seen_user_turns > IMAGE_LOOKBACK_TURNS:
            break
        for a in m.get("attachments") or []:
            if a.get("kind") == "image" and a.get("data"):
                images.append((a["data"], a["mime"]))
    images.reverse()
    return kept, images


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------
def _status(exc: Exception):
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retryable(exc: Exception) -> bool:
    return _status(exc) in (429, 500, 502, 503, 504)


def friendly_error(exc: Exception) -> str:
    log.exception("LLM call failed", exc_info=exc)
    if isinstance(exc, LLMError):
        return str(exc)
    status = _status(exc)
    if status in (401, 403):
        return "Simon can't authenticate with the model provider. Check the API keys in the app's secrets."
    if status == 429:
        return "Simon's model provider is rate-limiting requests right now. Give it a moment and try again."
    if status and status >= 500:
        return "The model provider is having a moment. Try again in a few seconds."
    if status == 400:
        return "The model rejected that request (it may be too long). Try a shorter message or a smaller file."
    return "Something went wrong reaching the model. Please try again."


# ------------------------------------------------------------------
# Providers
# ------------------------------------------------------------------
def _stream_groq(system: str, history: list, want_reasoning: bool) -> Iterator[Chunk]:
    key = secret("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is missing. Add it in the app's secrets.")
    from groq import Groq

    client = Groq(api_key=key)
    messages = [{"role": "system", "content": system}] + history
    attempts = [(GROQ_MODEL, True), (GROQ_FALLBACK_MODEL, False)]
    last_exc = None

    for model, is_reasoning_model in attempts:
        for try_no in range(2):
            started = False
            try:
                extra = {}
                if is_reasoning_model:
                    extra = {
                        "reasoning_effort": "medium" if want_reasoning else "low",
                        "include_reasoning": bool(want_reasoning),
                    }
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4000 if is_reasoning_model else 2000,
                    stream=True,
                    extra_body=extra or None,
                )
                for event in stream:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    thought = getattr(delta, "reasoning", None)
                    if thought and want_reasoning:
                        started = True
                        yield Chunk("reasoning", thought)
                    if delta.content:
                        started = True
                        yield Chunk("answer", delta.content)
                return
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                last_exc = exc
                if started:
                    raise  # can't safely restart a half-delivered answer
                if _retryable(exc) and try_no == 0:
                    time.sleep(1.5)
                    continue
                break  # give up on this model
        if model == GROQ_MODEL:
            yield Chunk("notice", "Primary model is busy, so Simon switched to a backup.")
    raise last_exc


def _stream_gemini(system: str, history: list, images: list, want_reasoning: bool) -> Iterator[Chunk]:
    key = secret("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is missing. Add it in the app's secrets to analyze images.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    contents = []
    for i, item in enumerate(history):
        role = "model" if item["role"] == "assistant" else "user"
        parts = [types.Part.from_text(text=item["content"])]
        if i == len(history) - 1 and role == "user":
            for data, mime in images:
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        contents.append(types.Content(role=role, parts=parts))

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.7,
        max_output_tokens=4000,
        thinking_config=types.ThinkingConfig(include_thoughts=bool(want_reasoning)),
    )

    for try_no in range(2):
        started = False
        try:
            for chunk in client.models.generate_content_stream(model=GEMINI_MODEL, contents=contents, config=config):
                for cand in chunk.candidates or []:
                    if not cand.content or not cand.content.parts:
                        continue
                    for part in cand.content.parts:
                        if not part.text:
                            continue
                        started = True
                        yield Chunk("reasoning" if part.thought else "answer", part.text)
            return
        except Exception as exc:  # noqa: BLE001
            if not started and _retryable(exc) and try_no == 0:
                time.sleep(1.5)
                continue
            raise


def _stream_fake(want_reasoning: bool, images: list) -> Iterator[Chunk]:
    """Offline dev/demo backend: SIMON_FAKE_LLM=1 streamlit run app.py"""
    if want_reasoning:
        for w in "First, who gains from this framing? Then, what system produced it, and what happens next?".split():
            time.sleep(0.03)
            yield Chunk("reasoning", w + " ")
    text = (
        "Here's how I'm thinking about it.\n\n"
        + ("I can see the image you attached. " if images else "")
        + "**What appears to be happening:** a decision framed as urgent when the "
        "incentives suggest it isn't.\n\n- Who benefits if you act now?\n- Who benefits if you wait?\n\n"
        "What outcome are you actually optimizing for? Costs of $5 or $50 hardly matter if the premise is wrong."
    )
    for w in text.split(" "):
        time.sleep(0.02)
        yield Chunk("answer", w + " ")


def stream_reply(system: str, history: list, images: list, want_reasoning: bool) -> Iterator[Chunk]:
    if os.environ.get("SIMON_FAKE_LLM"):
        yield from _stream_fake(want_reasoning, images)
    elif images:
        yield from _stream_gemini(system, history, images, want_reasoning)
    else:
        yield from _stream_groq(system, history, want_reasoning)
