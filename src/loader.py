"""Чтение, валидация, дедупликация и нормализация вакансий.

Это ЧИСТО обычная логика (без LLM): детерминированно, воспроизводимо, тестируемо.
Каждое отклонение записывается в лог-список, который потом уходит в trace/run.log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


class LoaderError(Exception):
    """Фатальная ошибка загрузки (пустой/битый файл целиком)."""


# Ключевые поля: без них вакансию невозможно оценивать -> отбрасываем.
REQUIRED_FIELDS = ("title", "company")
# Второстепенные: при отсутствии не отбрасываем, помечаем "unknown" + штраф в scoring.
OPTIONAL_FIELDS = ("level", "role", "city", "work_format", "published")


@dataclass
class Vacancy:
    """Нормализованная вакансия."""

    id: str
    title: str
    company: str
    level: str
    role: str
    stack: list[str]
    work_format: str
    city: str
    published: str  # ISO yyyy-mm-dd или "unknown"
    description: str
    url: str
    # Служебное:
    published_date: date | None = None
    missing_fields: list[str] = field(default_factory=list)

    def stack_lower(self) -> set[str]:
        return {s.strip().lower() for s in self.stack if s and s.strip()}


@dataclass
class LoadResult:
    """Результат загрузки: чистые вакансии + журнал того, что отброшено/схлопнуто."""

    vacancies: list[Vacancy]
    rejected_rows: list[dict[str, Any]] = field(default_factory=list)  # битые/без полей
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    raw_count: int = 0


def _norm_format(value: str) -> str:
    """Нормализация формата работы к remote/hybrid/office/unknown."""
    v = (value or "").strip().lower()
    if v in ("remote", "удалённо", "удаленно", "удалёнка", "удаленка"):
        return "remote"
    if v in ("hybrid", "гибрид", "гибридный"):
        return "hybrid"
    if v in ("office", "офис", "офисный", "on-site", "onsite"):
        return "office"
    return "unknown"


def _parse_date(value: str) -> date | None:
    v = (value or "").strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def _dedup_key(rec: dict[str, Any]) -> str:
    """Ключ дедупа: нормализованные title+company.

    Ограничение: чисто строковое сравнение, не семантическое
    ('ML Engineer' != 'ML инженер'). См. Limitations в отчёте.
    """
    title = str(rec.get("title", "")).strip().lower()
    company = str(rec.get("company", "")).strip().lower()
    return f"{title}|{company}"


def load_vacancies(path: str | Path) -> LoadResult:
    """Загрузить и нормализовать вакансии из JSON-файла.

    Обрабатывает: пустой файл, битый JSON (фатально), не-объекты в массиве,
    пропущенные ключевые/второстепенные поля, дубли.
    """
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"Файл вакансий не найден: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise LoaderError(f"Файл вакансий пуст: {p}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LoaderError(f"Битый JSON в {p}: {e}") from e

    if not isinstance(data, list):
        raise LoaderError("Ожидался JSON-массив вакансий")
    if len(data) == 0:
        raise LoaderError("Массив вакансий пуст (0 записей)")

    result = LoadResult(vacancies=[], raw_count=len(data))
    seen: dict[str, str] = {}  # dedup_key -> id первой вакансии

    for idx, rec in enumerate(data):
        # Битая строка: не объект (None, строка, число).
        if not isinstance(rec, dict):
            result.rejected_rows.append(
                {"index": idx, "reason": "запись не является объектом", "raw": repr(rec)[:80]}
            )
            continue

        # Пропущенные ключевые поля.
        missing_required = [
            f for f in REQUIRED_FIELDS if not str(rec.get(f, "")).strip()
        ]
        if missing_required:
            result.rejected_rows.append(
                {
                    "index": idx,
                    "id": rec.get("id", f"#{idx}"),
                    "reason": f"нет ключевых полей: {', '.join(missing_required)}",
                }
            )
            continue

        # Дедуп.
        key = _dedup_key(rec)
        if key in seen:
            result.duplicates.append(
                {"id": rec.get("id", f"#{idx}"), "duplicate_of": seen[key], "key": key}
            )
            continue
        seen[key] = str(rec.get("id", f"#{idx}"))

        # Второстепенные поля -> пометки.
        missing_optional = [f for f in OPTIONAL_FIELDS if not str(rec.get(f, "")).strip()]

        stack = rec.get("stack", [])
        if not isinstance(stack, list):
            stack = []

        published_raw = str(rec.get("published", "")).strip()
        published_date = _parse_date(published_raw)

        result.vacancies.append(
            Vacancy(
                id=str(rec.get("id", f"#{idx}")),
                title=str(rec["title"]).strip(),
                company=str(rec["company"]).strip(),
                level=str(rec.get("level", "")).strip().lower() or "unknown",
                role=str(rec.get("role", "")).strip().lower() or "unknown",
                stack=[str(s).strip() for s in stack if str(s).strip()],
                work_format=_norm_format(str(rec.get("work_format", ""))),
                city=str(rec.get("city", "")).strip() or "unknown",
                published=published_raw or "unknown",
                description=str(rec.get("description", "")).strip(),
                url=str(rec.get("url", "")).strip(),
                published_date=published_date,
                missing_fields=missing_optional,
            )
        )

    if not result.vacancies:
        raise LoaderError("После валидации не осталось ни одной пригодной вакансии")

    return result
