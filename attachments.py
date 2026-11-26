"""Turn uploaded files into attachment dicts, once, at upload time.

An attachment is either
  {"name", "kind": "text",  "text", "truncated"}   (PDF/DOCX/TXT/CSV/MD/JSON)
  {"name", "kind": "image", "mime", "data"}        (PNG/JPEG/WEBP/GIF)
"""

import io
import logging

from settings import FILE_CHAR_CAP, MAX_FILE_BYTES

log = logging.getLogger("simon.attachments")

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif"]
TEXT_TYPES = ["pdf", "docx", "txt", "csv", "md", "json"]
ALLOWED_TYPES = IMAGE_TYPES + TEXT_TYPES

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


class AttachmentError(Exception):
    """Raised with a message that is safe to show to the user."""


def _ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise AttachmentError("That PDF is password-protected, so I can't read it.")
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if not text:
        raise AttachmentError(
            "I couldn't find any text in that PDF. It may be a scan; try attaching it as an image instead."
        )
    return text


def _read_docx(data: bytes) -> str:
    import docx

    doc = docx.Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts).strip()


def process_upload(uploaded) -> dict:
    name = uploaded.name
    ext = _ext(name)
    data = uploaded.getvalue()

    if len(data) > MAX_FILE_BYTES:
        raise AttachmentError(f"{name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    if ext not in ALLOWED_TYPES:
        raise AttachmentError(f"{name}: unsupported file type.")

    if ext in IMAGE_TYPES:
        mime = _MIME_BY_EXT[ext]
        return {"name": name, "kind": "image", "mime": mime, "data": data}

    try:
        if ext == "pdf":
            text = _read_pdf(data)
        elif ext == "docx":
            text = _read_docx(data)
        else:
            text = data.decode("utf-8", errors="ignore")
    except AttachmentError:
        raise
    except Exception as exc:
        log.warning("Could not parse %s: %s", name, exc)
        raise AttachmentError(f"I couldn't read {name}. The file may be corrupted.") from exc

    text = text.strip()
    if not text:
        raise AttachmentError(f"{name} appears to be empty.")
    truncated = len(text) > FILE_CHAR_CAP
    if truncated:
        text = text[:FILE_CHAR_CAP]
    return {"name": name, "kind": "text", "text": text, "truncated": truncated}
