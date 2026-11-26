"""SQLite persistence for conversations.

Chats are keyed by an anonymous per-browser token (kept in the URL as ?u=...),
so visitors never see each other's history. Note: on Streamlit Community
Cloud the disk is ephemeral and resets on redeploy/restart. For permanent
history, point SIMON_DB_PATH at a persistent volume or swap this module for
Supabase/Postgres (the function signatures are the only contract).
"""

import json
import sqlite3
import time
import uuid
from contextlib import closing

from settings import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    title TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_owner ON conversations(owner, updated DESC);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    reasoning TEXT,
    sources TEXT,
    attachments TEXT,
    feedback INTEGER,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id, id);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with closing(_connect()) as conn, conn:
        conn.executescript(_SCHEMA)


def list_conversations(owner: str, limit: int = 30) -> list:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, title, updated FROM conversations WHERE owner=? ORDER BY updated DESC LIMIT ?",
            (owner, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def create_conversation(owner: str, title: str = "New conversation") -> str:
    conv_id = uuid.uuid4().hex
    now = time.time()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO conversations (id, owner, title, created, updated) VALUES (?,?,?,?,?)",
            (conv_id, owner, title, now, now),
        )
    return conv_id


def set_title(conv_id: str, title: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title[:80], conv_id))


def delete_conversation(owner: str, conv_id: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM messages WHERE conv_id IN (SELECT id FROM conversations WHERE id=? AND owner=?)", (conv_id, owner))
        conn.execute("DELETE FROM conversations WHERE id=? AND owner=?", (conv_id, owner))


def owns(owner: str, conv_id: str) -> bool:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT 1 FROM conversations WHERE id=? AND owner=?", (conv_id, owner)).fetchone()
    return row is not None


def _strip_binary(attachments: list) -> list:
    """Never store raw image bytes in the DB; keep metadata only."""
    return [{k: v for k, v in a.items() if k != "data"} for a in (attachments or [])]


def add_message(conv_id: str, msg: dict) -> int:
    now = time.time()
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO messages (conv_id, role, content, reasoning, sources, attachments, created) VALUES (?,?,?,?,?,?,?)",
            (
                conv_id,
                msg["role"],
                msg["content"],
                msg.get("reasoning"),
                json.dumps(msg.get("sources") or []),
                json.dumps(_strip_binary(msg.get("attachments"))),
                now,
            ),
        )
        conn.execute("UPDATE conversations SET updated=? WHERE id=?", (now, conv_id))
        return cur.lastrowid


def load_messages(conv_id: str) -> list:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, role, content, reasoning, sources, attachments, feedback FROM messages WHERE conv_id=? ORDER BY id",
            (conv_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "reasoning": r["reasoning"],
                "sources": json.loads(r["sources"] or "[]"),
                "attachments": json.loads(r["attachments"] or "[]"),
                "feedback": r["feedback"],
            }
        )
    return out


def set_feedback(msg_id: int, value) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE messages SET feedback=? WHERE id=?", (value, msg_id))


def delete_message(msg_id: int) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM messages WHERE id=?", (msg_id,))
