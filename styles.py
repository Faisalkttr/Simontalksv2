"""Custom CSS and small HTML components."""
import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

/* Only restyle text containers; never touch Streamlit's icon font */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, button p, textarea, input, label, [data-testid="stCaptionContainer"] {
  font-family: 'Inter', system-ui, sans-serif;
}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
h1, h2, h3, .simon-serif { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.01em; }

/* Layout: a calmer, narrower reading column */
.block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 7rem; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
#MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; }

/* Hero / empty state */
[data-testid="stMainBlockContainer"] [data-testid="stButtonGroup"] > div { justify-content: center; }
.hero { text-align: center; padding: 3.2rem 0 1.4rem; }
.hero .mark { font-size: 2.6rem; line-height: 1; }
.hero h1 { font-size: 2.6rem; margin: .5rem 0 .2rem; font-weight: 600; }
.hero p { color: #9A978F; font-size: 1.05rem; margin: 0; }

/* Chat */
[data-testid="stChatMessage"] { background: transparent; padding: .6rem 0; gap: .9rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: rgba(255,255,255,0.035); border-radius: 14px; padding: .8rem 1rem;
}
[data-testid="stChatMessage"] p { line-height: 1.68; }
[data-testid="stChatInput"] { border-radius: 16px; }

/* Thinking indicator while waiting for the first token */
.thinking { color: #9A978F; font-style: italic; }
.thinking::after { content: ''; display: inline-block; width: .5em; height: .5em; margin-left: .5em;
  border-radius: 50%; background: #E0B25A; animation: pulse 1.1s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: .25; transform: scale(.8);} 50% { opacity: 1; transform: scale(1.1);} }

/* Citation chips */
.chips { display: flex; flex-wrap: wrap; gap: .45rem; margin: .7rem 0 .2rem; }
.chip { display: inline-flex; align-items: center; gap: .4rem; max-width: 100%;
  padding: .28rem .7rem; border-radius: 999px; font-size: .8rem; text-decoration: none !important;
  color: #E8E6E1 !important; background: rgba(224,178,90,0.10); border: 1px solid rgba(224,178,90,0.28); }
.chip:hover { background: rgba(224,178,90,0.2); }
.chip .t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 24rem; }
.chip.file { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.14); }
.chips-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #8A877F; margin-top: .6rem; }

/* Sidebar conversation list */
[data-testid="stSidebar"] .stButton { width: 100%; }
[data-testid="stSidebar"] .stButton > button { width: 100%; justify-content: flex-start; text-align: left; border-radius: 10px; }
.sidebar-title { font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600; margin: 0; }
.sidebar-sub { color: #9A978F; font-size: .82rem; margin: 0 0 .8rem; }
</style>
"""

SOURCE_ICONS = {"YouTube": "▶", "Blog": "✎", "Twitter/X": "𝕏"}


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def safe_md(text: str) -> str:
    """Escape $ so prices like $5 and $50 don't render as LaTeX."""
    return text.replace("$", "\\$")


def hero() -> None:
    st.markdown(
        '<div class="hero"><div class="mark">🧭</div><h1>Simon</h1>'
        "<p>Let's slow down and think this through.</p></div>",
        unsafe_allow_html=True,
    )


def thinking_html() -> str:
    return '<span class="thinking">Simon is thinking</span>'


def source_chips(sources: list) -> str:
    if not sources:
        return ""
    chips = []
    for s in sources:
        icon = SOURCE_ICONS.get(s["type"], "•")
        title = html.escape(s["title"])
        label = f'<span>{icon}</span><span class="t">{title}</span>'
        if s.get("url"):
            chips.append(f'<a class="chip" href="{html.escape(s["url"], quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a>')
        else:
            chips.append(f'<span class="chip">{label}</span>')
    return '<div class="chips-label">Drew on</div><div class="chips">' + "".join(chips) + "</div>"


def file_chips(attachments: list) -> str:
    if not attachments:
        return ""
    chips = []
    for a in attachments:
        icon = "🖼" if a.get("kind") == "image" else "📄"
        chips.append(f'<span class="chip file"><span>{icon}</span><span class="t">{html.escape(a["name"])}</span></span>')
    return '<div class="chips">' + "".join(chips) + "</div>"
