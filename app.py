"""Simon: strategic thinking partner (Streamlit).

Text turns stream from Groq, image turns stream from Gemini. Conversations
persist per browser, sources show as citation chips, reasoning streams into
a collapsible panel.
"""

import hmac
import logging
import os
import time
import uuid

import streamlit as st

st.set_page_config(page_title="Simon", page_icon="🧭", layout="centered", initial_sidebar_state="auto")

import attachments as att  # noqa: E402
import knowledge_base as kb  # noqa: E402
import llm  # noqa: E402
import store  # noqa: E402
import styles  # noqa: E402
from prompts import DEFAULT_MODE, FILE_ONLY_PROMPT, MODES, STARTERS, build_system_prompt  # noqa: E402
from settings import RATE_LIMIT_PER_HOUR, secret  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("simon.app")

MODE_HINTS = {
    "Explore": "Widen the frame: assumptions, alternatives, what's missing.",
    "Challenge": "Stress-test my thinking: steelman, pre-mortem, weakest link.",
    "Decide": "Options, trade-offs, and a clear recommendation.",
}
MAX_IMAGES_PER_MESSAGE = 4

ss = st.session_state


@st.cache_resource
def _init_db() -> bool:
    store.init()
    return True


_init_db()
styles.inject()


# ============================================================
# Access control (optional): set APP_PASSCODE in secrets to enable
# ============================================================
def require_passcode() -> None:
    code = secret("APP_PASSCODE")
    if not code or ss.get("authed"):
        return
    styles.hero()
    with st.form("gate"):
        entered = st.text_input("Passcode", type="password")
        if st.form_submit_button("Enter"):
            if hmac.compare_digest(entered.encode(), code.encode()):
                ss.authed = True
                st.rerun()
            st.error("That passcode isn't right.")
    st.stop()


require_passcode()

# ============================================================
# Identity + session state
# ============================================================
owner = st.query_params.get("u")
if not owner or len(owner) < 8:
    owner = uuid.uuid4().hex[:20]
    st.query_params["u"] = owner  # bookmark this URL to keep your chat history

ss.setdefault("conv_id", None)
ss.setdefault("messages", [])
ss.setdefault("generate_now", False)
ss.setdefault("send_times", [])
ss.setdefault("kb_refresh_nonce", 0)
ss.setdefault("kb_error", None)
ss.setdefault("manual_tweets_text", "")


# ============================================================
# Callbacks
# ============================================================
def new_chat() -> None:
    ss.conv_id, ss.messages, ss.generate_now = None, [], False


def open_chat(conv_id: str) -> None:
    if store.owns(owner, conv_id):
        ss.conv_id = conv_id
        ss.messages = store.load_messages(conv_id)
        ss.generate_now = False


def delete_chat(conv_id: str) -> None:
    store.delete_conversation(owner, conv_id)
    if ss.conv_id == conv_id:
        new_chat()


def pick_starter() -> None:
    choice = ss.get("starter_pick")
    if choice:
        ss.pending_prompt = choice
        ss.starter_pick = None


def regenerate() -> None:
    if ss.messages and ss.messages[-1]["role"] == "assistant":
        last = ss.messages.pop()
        if last.get("id"):
            store.delete_message(last["id"])
        ss.generate_now = True


def retry() -> None:
    ss.generate_now = True


def save_feedback(msg_id: int, key: str) -> None:
    value = ss.get(key)
    store.set_feedback(msg_id, value)
    for m in ss.messages:
        if m.get("id") == msg_id:
            m["feedback"] = value


def refresh_kb() -> None:
    ss.kb_refresh_nonce += 1
    ss.kb_error = None
    kb.build_knowledge_base.clear()


def export_markdown() -> str:
    title = next((m["content"] for m in ss.messages if m["role"] == "user"), "Conversation")[:60]
    lines = [f"# Simon: {title}", ""]
    for m in ss.messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'Simon'}:**\n\n{m['content']}\n")
    return "\n".join(lines)


def load_kb(youtube_id: str, blog_url: str, tweets: str):
    """Build (or fetch cached) knowledge base. Failures are shown, and not retried every message."""
    if not (youtube_id.strip() or blog_url.strip() or tweets.strip()):
        return None
    err = ss.kb_error
    if err and time.time() - err[1] < 300:
        return None
    try:
        with st.spinner("Indexing Simon's sources…"):
            return kb.build_knowledge_base(youtube_id, blog_url, tweets, ss.kb_refresh_nonce)
    except Exception as exc:  # noqa: BLE001
        log.exception("Knowledge base build failed")
        message = str(exc) if isinstance(exc, kb.KBError) else f"Indexing failed ({type(exc).__name__})."
        ss.kb_error = (message, time.time())
        return None


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        '<p class="sidebar-title">🧭 Simon</p><p class="sidebar-sub">Strategic thinking partner</p>',
        unsafe_allow_html=True,
    )
    st.button("＋ New chat", on_click=new_chat, type="primary", key="new_chat")

    convs = store.list_conversations(owner, limit=12)
    if convs:
        st.caption("Recent")
    for c in convs:
        left, right = st.columns([6, 1], vertical_alignment="center")
        active = c["id"] == ss.conv_id
        left.button(
            ("● " if active else "") + c["title"],
            key=f"open_{c['id']}",
            on_click=open_chat,
            args=(c["id"],),
            type="secondary" if active else "tertiary",
        )
        right.button("✕", key=f"del_{c['id']}", on_click=delete_chat, args=(c["id"],), type="tertiary", help="Delete chat")

    st.divider()
    mode = st.segmented_control("Mode", list(MODES), default=DEFAULT_MODE, key="mode") or DEFAULT_MODE
    st.caption(MODE_HINTS[mode])
    show_reasoning = st.toggle("Show Simon's reasoning", value=True)

    with st.expander("📚 Simon's sources"):
        st.caption("Ground his answers in your own published content.")
        youtube_channel_id = st.text_input(
            "YouTube Channel ID",
            value=secret("YOUTUBE_CHANNEL_ID"),
            help="The UC… channel ID, not the @handle.",
        )
        blog_rss_url = st.text_input("Blog RSS feed URL", value=secret("BLOG_RSS_URL"), help="Often /feed, /rss or /rss.xml")
        manual_tweets_text = st.text_area(
            "Twitter/X content (paste)",
            value=ss.manual_tweets_text,
            height=100,
            help="Blank line between each tweet or thread.",
        )
        ss.manual_tweets_text = manual_tweets_text
        st.button("🔄 Refresh knowledge base", on_click=refresh_kb)

        knowledge_base = load_kb(youtube_channel_id, blog_rss_url, manual_tweets_text)
        if knowledge_base:
            s = knowledge_base["stats"]
            st.caption(f"Indexed: {s['youtube']} YouTube · {s['blog']} blog · {s['twitter']} tweet chunks")
            for w in knowledge_base.get("warnings", []):
                st.warning(w, icon="⚠️")
            if not knowledge_base["chunks"]:
                st.info("Nothing was indexed from these sources.")
        elif ss.kb_error:
            st.error(ss.kb_error[0])
            st.caption("Will retry automatically in a few minutes, or press Refresh.")

    st.download_button(
        "⬇ Export chat (.md)",
        data=export_markdown() if ss.messages else "",
        file_name="simon-chat.md",
        mime="text/markdown",
        disabled=not ss.messages,
        type="tertiary",
    )

    if not os.environ.get("SIMON_FAKE_LLM"):
        if not secret("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is missing. Add it under App settings → Secrets.")
        if not secret("GEMINI_API_KEY"):
            st.caption("Add GEMINI_API_KEY to enable photo analysis and source grounding.")

# ============================================================
# Input handling (chat_input is pinned to the bottom regardless of position)
# ============================================================
submitted = st.chat_input(
    "Bring Simon a situation, a question, or a file…",
    accept_file="multiple",
    file_type=att.ALLOWED_TYPES,
    max_chars=8000,
)
pending = ss.pop("pending_prompt", None)

text, files = None, []
if pending:
    text = pending
elif submitted:
    text = (getattr(submitted, "text", "") or "").strip()
    files = list(getattr(submitted, "files", None) or [])

if text is not None and (text or files):
    new_attachments = []
    for f in files:
        try:
            new_attachments.append(att.process_upload(f))
        except att.AttachmentError as exc:
            st.toast(str(exc), icon="⚠️")
    if sum(1 for a in new_attachments if a["kind"] == "image") > MAX_IMAGES_PER_MESSAGE:
        new_attachments = [a for a in new_attachments if a["kind"] != "image"]
        st.toast(f"Up to {MAX_IMAGES_PER_MESSAGE} images per message; none were attached.", icon="⚠️")

    if text or new_attachments:
        if ss.conv_id is None:
            ss.conv_id = store.create_conversation(owner)
        user_msg = {"role": "user", "content": text or FILE_ONLY_PROMPT, "attachments": new_attachments}
        user_msg["id"] = store.add_message(ss.conv_id, user_msg)
        if not ss.messages:
            store.set_title(ss.conv_id, user_msg["content"].replace("\n", " ")[:48])
        ss.messages.append(user_msg)
        ss.generate_now = True


# ============================================================
# Rendering
# ============================================================
def render_message(msg: dict, is_last: bool) -> None:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(styles.safe_md(msg["content"]))
            if msg.get("attachments"):
                st.markdown(styles.file_chips(msg["attachments"]), unsafe_allow_html=True)
        return

    with st.chat_message("assistant", avatar="🧭"):
        if msg.get("reasoning"):
            with st.expander("🧠 Simon's reasoning"):
                st.markdown(styles.safe_md(msg["reasoning"]))
        st.markdown(styles.safe_md(msg["content"]))
        if msg.get("sources"):
            st.markdown(styles.source_chips(msg["sources"]), unsafe_allow_html=True)
        if msg.get("id"):
            key = f"fb_{msg['id']}"
            if msg.get("feedback") is not None and key not in ss:
                ss[key] = msg["feedback"]
            left, right = st.columns([1, 4], vertical_alignment="center")
            with left:
                st.feedback("thumbs", key=key, on_change=save_feedback, args=(msg["id"], key))
            if is_last:
                with right:
                    st.button("↻ Regenerate", key=f"regen_{msg['id']}", on_click=regenerate, type="tertiary")


if not ss.messages:
    styles.hero()
    st.pills("Try", STARTERS, key="starter_pick", on_change=pick_starter, label_visibility="collapsed")

for i, m in enumerate(ss.messages):
    render_message(m, is_last=(i == len(ss.messages) - 1))


# ============================================================
# Generation
# ============================================================
def seconds_until_allowed() -> int:
    if not RATE_LIMIT_PER_HOUR:
        return 0
    now = time.time()
    ss.send_times = [t for t in ss.send_times if now - t < 3600]
    if len(ss.send_times) >= RATE_LIMIT_PER_HOUR:
        return int(3600 - (now - ss.send_times[0]))
    return 0


def generate_reply() -> None:
    wait = seconds_until_allowed()
    if wait:
        st.warning(f"You've reached the hourly message limit. Try again in about {max(1, wait // 60)} min.")
        return
    ss.send_times.append(time.time())

    user_msgs = [m for m in ss.messages if m["role"] == "user"]
    question = user_msgs[-1]["content"]
    prior_question = user_msgs[-2]["content"] if len(user_msgs) > 1 else ""

    grounding, sources = "", []
    if knowledge_base and knowledge_base.get("chunks"):
        try:
            hits = kb.retrieve_relevant_chunks(knowledge_base, kb.retrieval_query(question, prior_question))
            grounding = kb.format_grounding_block(hits)
            sources = kb.sources_for_display(hits)
        except Exception:  # grounding is best-effort; never block a reply on it
            log.exception("Retrieval failed")

    system_prompt = build_system_prompt(mode, show_reasoning, grounding)
    history, images = llm.prepare_turn(ss.messages)

    reasoning, answer, status, reasoning_ph = "", "", None, None
    started = time.time()

    with st.chat_message("assistant", avatar="🧭"):
        top = st.container()
        answer_ph = st.empty()
        answer_ph.markdown(styles.thinking_html(), unsafe_allow_html=True)
        try:
            for chunk in llm.stream_reply(system_prompt, history, images, want_reasoning=show_reasoning):
                if chunk.kind == "reasoning":
                    reasoning += chunk.text
                    if status is None:
                        with top:
                            status = st.status("Simon is reasoning…", expanded=True)
                            reasoning_ph = status.empty()
                    reasoning_ph.markdown(styles.safe_md(reasoning) + " ▌")
                elif chunk.kind == "answer":
                    if not answer and status is not None:
                        status.update(label=f"Reasoned for {int(time.time() - started)}s", state="complete", expanded=False)
                    answer += chunk.text
                    answer_ph.markdown(styles.safe_md(answer) + " ▌")
                elif chunk.kind == "notice":
                    st.toast(chunk.text, icon="ℹ️")
            if status is not None:
                reasoning_ph.markdown(styles.safe_md(reasoning))
                if not answer:
                    status.update(state="error", expanded=True)
        except Exception as exc:  # noqa: BLE001
            answer_ph.empty()
            if status is not None:
                status.update(state="error")
            if answer:  # partial reply: keep what arrived, tell the user it was cut off
                st.warning("The reply was cut off. You can regenerate it.")
            else:
                st.error(llm.friendly_error(exc))
                return

        if not answer.strip():
            answer_ph.empty()
            st.warning("Simon ran out of room before answering. Try again or ask something narrower.")
            return
        answer_ph.markdown(styles.safe_md(answer))

    reply = {"role": "assistant", "content": answer.strip(), "reasoning": reasoning.strip() or None, "sources": sources}
    reply["id"] = store.add_message(ss.conv_id, reply)
    ss.messages.append(reply)
    st.rerun()


if ss.pop("generate_now", False) and ss.messages and ss.messages[-1]["role"] == "user":
    generate_reply()

if ss.messages and ss.messages[-1]["role"] == "user" and not ss.get("generate_now"):
    st.button("↻ Try again", on_click=retry, key="retry_btn")
