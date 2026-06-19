"""Извлечение текста из файла резюме (pdf/docx/txt/md).

Обычная логика: файл -> чистый текст. LLM здесь НЕ участвует — он получает уже текст.
pdf/docx требуют опциональных библиотек; при их отсутствии — понятная ошибка.
"""
from __future__ import annotations

from pathlib import Path


class ResumeError(Exception):
    """Не удалось прочитать/распарсить резюме."""


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ResumeError("Для PDF нужен пакет pypdf: pip install pypdf") from e
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise ResumeError("Для DOCX нужен пакет python-docx: pip install python-docx") from e
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


_READERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def extract_text(path: str | Path) -> str:
    """Прочитать резюме и вернуть текст. Диспетчер по расширению файла."""
    p = Path(path)
    if not p.exists():
        raise ResumeError(f"Файл резюме не найден: {p}")
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise ResumeError(f"Неподдерживаемый формат резюме: {p.suffix} (нужно pdf/docx/txt/md)")
    text = reader(p).strip()
    if not text:
        raise ResumeError(f"Из резюме не удалось извлечь текст (пусто): {p}")
    return text
