"""Hard-фильтр, scoring и ранжирование.

Обычная логика: детерминированный числовой score из весов config.yaml.
LLM здесь НЕ участвует — рейтинг воспроизводим и объясним по компонентам.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .config import Criteria
from .loader import Vacancy

# Окно свежести: публикация старше стольких дней даёт freshness ~0.
FRESHNESS_WINDOW_DAYS = 45


@dataclass
class Scored:
    """Вакансия с посчитанным score и разбивкой по компонентам (для объяснимости)."""

    vacancy: Vacancy
    score: float
    components: dict[str, float] = field(default_factory=dict)
    matched_skills: list[str] = field(default_factory=list)


@dataclass
class RejectedHard:
    """Вакансия, отсеянная hard-фильтром (детерминированная причина, без LLM)."""

    vacancy: Vacancy
    reason: str


def _skill_overlap(vac: Vacancy, crit: Criteria) -> tuple[float, list[str]]:
    """Доля желаемых навыков кандидата, покрытых стеком вакансии (0..1)."""
    want = {s.lower() for s in crit.skills}
    have = vac.stack_lower()
    matched = sorted(want & have)
    if not want:
        return 0.0, []
    return len(matched) / len(want), matched


def _role_match(vac: Vacancy, crit: Criteria) -> float:
    """1.0 точное вхождение роли, 0.5 частичное (общее слово), иначе 0.0."""
    role = vac.role.lower()
    if role == "unknown":
        return 0.3
    for r in crit.role:
        if r in role or role in r:
            return 1.0
    role_words = set(role.split())
    for r in crit.role:
        if role_words & set(r.split()):
            return 0.5
    return 0.0


def _level_match(vac: Vacancy, crit: Criteria) -> float:
    if vac.level == "unknown":
        return 0.4
    return 1.0 if vac.level in {l.lower() for l in crit.level} else 0.2


def _freshness(vac: Vacancy, ref: date | None) -> float:
    if vac.published_date is None or ref is None:
        return 0.3  # неизвестная дата — нейтрально-низко
    days = (ref - vac.published_date).days
    if days <= 0:
        return 1.0
    if days >= FRESHNESS_WINDOW_DAYS:
        return 0.0
    return 1.0 - days / FRESHNESS_WINDOW_DAYS


def _work_format(vac: Vacancy, crit: Criteria) -> float:
    if vac.work_format == "unknown":
        return 0.4
    wanted = {_norm(c) for c in crit.work_format}
    return 1.0 if vac.work_format in wanted else 0.2


def _city_match(vac: Vacancy, crit: Criteria) -> float:
    city = vac.city.lower()
    if vac.work_format == "remote" or city == "remote":
        return 1.0
    wanted = {c.lower() for c in crit.city}
    if city == "unknown":
        return 0.4
    return 1.0 if city in wanted else 0.3


def _norm(value: str) -> str:
    v = value.strip().lower()
    mapping = {"удалённо": "remote", "удаленно": "remote", "гибрид": "hybrid"}
    return mapping.get(v, v)


def _reference_date(vacs: list[Vacancy]) -> date | None:
    """Опорная дата для freshness = самая поздняя публикация в наборе.

    Делает score детерминированным независимо от реальной 'сегодня'.
    """
    dates = [v.published_date for v in vacs if v.published_date]
    return max(dates) if dates else None


def filter_and_score(
    vacancies: list[Vacancy], crit: Criteria, config: dict[str, Any]
) -> tuple[list[Scored], list[RejectedHard]]:
    """Hard-фильтр по уровню, затем взвешенный score. Возвращает (отсортированные, отсеянные)."""
    reject_levels = {l.lower() for l in config.get("reject_levels", [])}
    weights = config.get("weights", {})
    ref = _reference_date(vacancies)

    scored: list[Scored] = []
    rejected: list[RejectedHard] = []

    for vac in vacancies:
        # Hard-фильтр: уровень не подходит junior-кандидату.
        if vac.level in reject_levels:
            rejected.append(
                RejectedHard(vacancy=vac, reason=f"уровень '{vac.level}' не подходит (нужен стажёр/junior)")
            )
            continue

        overlap, matched = _skill_overlap(vac, crit)
        comp = {
            "skill_overlap": overlap,
            "role_match": _role_match(vac, crit),
            "level_match": _level_match(vac, crit),
            "freshness": _freshness(vac, ref),
            "work_format": _work_format(vac, crit),
            "city_match": _city_match(vac, crit),
        }
        total = sum(weights.get(k, 0.0) * v for k, v in comp.items()) * 100.0
        scored.append(
            Scored(
                vacancy=vac,
                score=round(total, 1),
                components={k: round(v, 3) for k, v in comp.items()},
                matched_skills=matched,
            )
        )

    # Сортировка по убыванию score, при равенстве — по свежести (id для стабильности).
    scored.sort(key=lambda s: (-s.score, s.vacancy.id))
    return scored, rejected
