"""Config access: Streamlit secrets first, then environment variables."""
import os

import streamlit as st


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
    except Exception:  # no secrets.toml at all
        value = None
    if value in (None, ""):
        value = os.environ.get(name, default)
    return str(value) if value is not None else default


def secret_int(name: str, default: int) -> int:
    try:
        return int(secret(name, str(default)))
    except ValueError:
        return default


# Limits (override in secrets/env if your model quota allows more)
MAX_FILE_BYTES = 10 * 1024 * 1024
FILE_CHAR_CAP = secret_int("FILE_CHAR_CAP", 12000)          # per attached text file
HISTORY_CHAR_BUDGET = secret_int("HISTORY_CHAR_BUDGET", 20000)
RATE_LIMIT_PER_HOUR = secret_int("RATE_LIMIT_PER_HOUR", 40)  # per browser session; 0 = off
DB_PATH = secret("SIMON_DB_PATH", "simon.db")
