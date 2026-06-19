"""Тесты парсинга резюме (файл -> текст)."""
import pytest

from src.resume import ResumeError, extract_text


def test_txt(tmp_path):
    p = tmp_path / "cv.txt"
    p.write_text("Python, PyTorch, ML инженер", encoding="utf-8")
    assert "PyTorch" in extract_text(p)


def test_md(tmp_path):
    p = tmp_path / "cv.md"
    p.write_text("# Резюме\nNLP, LLM", encoding="utf-8")
    assert "LLM" in extract_text(p)


def test_missing_file(tmp_path):
    with pytest.raises(ResumeError):
        extract_text(tmp_path / "нет.txt")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   ", encoding="utf-8")
    with pytest.raises(ResumeError):
        extract_text(p)


def test_unsupported_format(tmp_path):
    p = tmp_path / "cv.rtf"
    p.write_text("text", encoding="utf-8")
    with pytest.raises(ResumeError):
        extract_text(p)
