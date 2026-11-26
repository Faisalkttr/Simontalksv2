# Simon: Strategic Thinking Partner

A Streamlit chatbot with a custom persona ("Simon") that analyzes text, files,
and photos through incentive, systems, long-term, and signal-vs-noise lenses.
Runs on free tiers.

- **Text chat** streams from **Groq** (`gpt-oss-120b`, auto-fallback to Llama 3.3 70B if rate-limited)
- **Photos** stream from **Gemini** (`gemini-2.5-flash`), using the same persona and conversation history
- **Files** (PDF, DOCX, TXT, CSV, MD, JSON) are extracted once and stay in context for follow-ups
- **Sources**: optional grounding in your YouTube / blog / tweets with clickable, timestamped citations

## What's new in v2

**Reliability**
- Moved to the supported `google-genai` SDK and `gemini-embedding-001` (the old embedding model/SDK is deprecated)
- Embeddings are batched; indexing problems now show up in the sidebar instead of silently producing an empty index
- YouTube transcripts work with both old and new `youtube-transcript-api` versions
- Attached files are processed once (size/type checks, 12k-character cap) and no longer re-sent as ghosts on every message
- History is trimmed to a character budget; friendly error messages (no raw exceptions); retry + backup model on rate limits
- Optional passcode and per-session rate limit; uploaded files and retrieved text are treated as data, not instructions

**Experience**
- Token-by-token streaming, with Simon's reasoning streaming into a collapsible panel
- Attach files directly in the chat box; starter prompts; Explore / Challenge / Decide modes
- Citation chips (YouTube links jump to the right timestamp), thumbs up/down, regenerate, retry, export to Markdown
- Saved conversations per browser, dark theme with custom typography, mobile-friendly

## 1. Free API keys
- **Groq**: https://console.groq.com/keys
- **Gemini**: https://aistudio.google.com/apikey

## 2. Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your keys
streamlit run app.py
```
No keys yet? `SIMON_FAKE_LLM=1 streamlit run app.py` runs the full UI against a built-in fake model.

Run the tests: `python -m pytest -q tests`

## 3. Deploy on Streamlit Community Cloud
Push these files to GitHub (never the real `secrets.toml`), create the app at
https://share.streamlit.io with `app.py` as the main file, and paste your keys under
**Settings → Secrets** (see `.streamlit/secrets.toml.example` for all options).

**Public app? Set `APP_PASSCODE`.** Otherwise anyone with the URL spends your free quota.

## 4. Chat history
Each browser gets an anonymous token in the URL (`?u=...`); bookmark the URL to keep your history.
History is stored in SQLite (`simon.db`). **On Streamlit Community Cloud the disk resets on restart/redeploy**,
so history is not permanent there. For durable history, host on a service with a persistent volume
and set `SIMON_DB_PATH`, or replace `store.py` with Supabase/Postgres (the function signatures are the contract).
Images are not stored: after a reload, Simon knows an image was attached but can't see it again.

## 5. Ground Simon in your content (optional)
Open **📚 Simon's sources** in the sidebar and add a YouTube Channel ID (the `UC...` ID, not the @handle),
a blog RSS URL, and/or pasted tweets. Click **Refresh knowledge base** to re-index (otherwise cached for an hour).
Relevant excerpts are retrieved per message and shown as chips under Simon's reply.
YouTube sometimes blocks transcript fetches from cloud IPs; the sidebar tells you how many videos were indexed.

## Project layout
| File | Purpose |
|---|---|
| `app.py` | UI, chat flow, generation loop |
| `prompts.py` | Persona, modes, starter prompts |
| `llm.py` | Groq/Gemini streaming, routing, fallbacks, history trimming |
| `knowledge_base.py` | Fetch, chunk, embed, retrieve |
| `attachments.py` | File/image processing |
| `store.py` | SQLite conversation storage |
| `styles.py`, `.streamlit/config.toml` | Theme and CSS |
| `tests/` | Unit tests |

## Customizing
Edit `SIMON_SYSTEM_PROMPT` and `MODES` in `prompts.py`; change models via secrets (`GROQ_MODEL`, `GEMINI_MODEL`).
Groq moves models between tiers occasionally; check https://console.groq.com/docs/models.
