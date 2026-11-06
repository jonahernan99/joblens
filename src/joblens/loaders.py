"""Load plain text from resume and job description files."""

from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ""}
PDF_SUFFIX = ".pdf"


class UnsupportedFileError(ValueError):
    """Raised when a file type cannot be read."""


def load_text(path: str | Path) -> str:
    """Return the text content of a ``.txt``, ``.md`` or ``.pdf`` file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == PDF_SUFFIX:
        return _load_pdf(path)

    supported = ", ".join(sorted(s for s in TEXT_SUFFIXES | {PDF_SUFFIX} if s))
    raise UnsupportedFileError(f"Unsupported file type {suffix!r} (supported: {supported})")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise UnsupportedFileError(
            "PDF support requires the optional dependency: pip install 'joblens[pdf]'"
        ) from exc

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
