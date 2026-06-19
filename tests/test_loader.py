"""Тесты загрузчика: пустой файл, битые строки, дубли."""
import json

import pytest

from src.loader import LoaderError, load_vacancies


def _write(tmp_path, data):
    p = tmp_path / "v.json"
    p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return p


def test_empty_file_raises(tmp_path):
    p = _write(tmp_path, "")
    with pytest.raises(LoaderError):
        load_vacancies(p)


def test_broken_json_raises(tmp_path):
    p = _write(tmp_path, "{not json")
    with pytest.raises(LoaderError):
        load_vacancies(p)


def test_non_object_rows_skipped(tmp_path):
    data = [
        None,
        "строка вместо объекта",
        {"id": "ok", "title": "ML Intern", "company": "A"},
    ]
    p = _write(tmp_path, data)
    res = load_vacancies(p)
    assert len(res.vacancies) == 1
    assert len(res.rejected_rows) == 2


def test_missing_required_field_skipped(tmp_path):
    data = [
        {"id": "x", "title": "", "company": "A"},  # пустой title
        {"id": "ok", "title": "ML Intern", "company": "B"},
    ]
    p = _write(tmp_path, data)
    res = load_vacancies(p)
    assert len(res.vacancies) == 1
    assert res.rejected_rows[0]["id"] == "x"


def test_dedup(tmp_path):
    data = [
        {"id": "a", "title": "ML Intern", "company": "ACME"},
        {"id": "b", "title": "ml intern ", "company": " acme"},  # дубль после нормализации
    ]
    p = _write(tmp_path, data)
    res = load_vacancies(p)
    assert len(res.vacancies) == 1
    assert len(res.duplicates) == 1
    assert res.duplicates[0]["duplicate_of"] == "a"
