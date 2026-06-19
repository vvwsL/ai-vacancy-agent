"""Сборка артефактов: report.md, trace.json, run.log."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .scoring import RejectedHard, Scored

_PRIORITY_RU = {"high": "высокий", "medium": "средний", "low": "низкий"}


def _score_rank_map(scored: list[Scored]) -> dict[str, int]:
    return {s.vacancy.id: i + 1 for i, s in enumerate(scored)}


def render_report(
    analyses: list[dict[str, Any]],
    rejected_hard: list[RejectedHard],
    borderline: list[Scored],
    meta: dict[str, Any],
) -> str:
    """Собрать report.md. analyses = [{scored, analysis, mode}, ...] (топ-N)."""
    lines: list[str] = []
    lines.append("# Отчёт агента по AI/ML вакансиям\n")
    lines.append(f"- Дата прогона: {meta['timestamp']}")
    lines.append(f"- Источник вакансий: **{meta.get('source', 'file')}**")
    lines.append(f"- Режим LLM: **{meta['provider']}**")
    lines.append(
        f"- Загружено записей: {meta['raw_count']} | валидных: {meta['valid_count']} | "
        f"отброшено битых: {meta['rejected_rows']} | дублей: {meta['duplicates']}"
    )
    lines.append(f"- Прошли hard-фильтр: {meta['scored_count']} | отсеяно по уровню: {len(rejected_hard)}\n")

    # ---- Топ ----
    lines.append(f"## Топ-{len(analyses)} рекомендаций\n")
    for i, item in enumerate(analyses, 1):
        sc: Scored = item["scored"]
        an = item["analysis"]
        vac = sc.vacancy
        prio = _PRIORITY_RU.get(an.priority, an.priority)
        lines.append(f"### {i}. {vac.title} — {vac.company}")
        lines.append(
            f"**Score: {sc.score}/100** | приоритет отклика: **{prio}** | "
            f"режим разбора: `{item['mode']}`  "
        )
        lines.append(
            f"Уровень: {vac.level} · роль: {vac.role} · формат: {vac.work_format} · "
            f"город: {vac.city} · опубликовано: {vac.published}  "
        )
        lines.append(f"[ссылка на вакансию]({vac.url})\n")

        # Расхождение score <-> приоритет агента (видно влияние LLM на решение).
        if an.priority_override and an.override_reason:
            lines.append(
                f"> ⚠️ Агент скорректировал приоритет (#{i} по score): "
                f"**{an.override_reason}**\n"
            )

        if an.matched:
            lines.append("**Что совпало:**")
            lines.extend(f"- {m}" for m in an.matched)
            lines.append("")
        if an.extracted_requirements:
            lines.append("**Извлечённые требования:**")
            lines.extend(f"- {r}" for r in an.extracted_requirements)
            lines.append("")
        if an.concerns:
            lines.append("**Что смущает / риски:**")
            lines.extend(f"- {c}" for c in an.concerns)
            lines.append("")
        if an.questions_to_employer:
            lines.append("**Вопросы к работодателю:**")
            lines.extend(f"- {q}" for q in an.questions_to_employer)
            lines.append("")
        lines.append(f"**Следующий шаг:** {an.next_step}\n")
        # Объяснимость: разбивка score по компонентам.
        comp = ", ".join(f"{k}={v}" for k, v in sc.components.items())
        lines.append(f"<sub>Разбивка score: {comp}</sub>\n")

    # ---- Причины отсева ----
    lines.append("## Причины отсева\n")
    if rejected_hard:
        lines.append("**Hard-фильтр (детерминированно, без LLM):**")
        for r in rejected_hard:
            lines.append(f"- {r.vacancy.title} @ {r.vacancy.company} — {r.reason}")
        lines.append("")
    if borderline:
        lines.append("**Пограничные (прошли фильтр, но не вошли в топ — низкий score):**")
        for sc in borderline:
            top_comp = min(sc.components.items(), key=lambda kv: kv[1])
            lines.append(
                f"- {sc.vacancy.title} @ {sc.vacancy.company} (score {sc.score}) — "
                f"слабее всего: {top_comp[0]}={top_comp[1]}"
            )
        lines.append("")

    # ---- Limitations ----
    lines.append("## Limitations (ограничения)\n")
    lines.extend(
        f"- {x}"
        for x in [
            "Дедупликация строковая (title+company), не семантическая: 'ML Engineer' и 'ML инженер' не схлопнутся.",
            "Score эвристический; веса заданы экспертно (см. config.yaml), не калиброваны на данных.",
            "Совпадение навыков — точное совпадение строк, без синонимов (Python≈py, LLM≈language model).",
            "LLM-режим воспроизводим по структуре, но не дословно; полностью детерминирован только dry-run.",
            "Источник данных — локальный файл (по условию задания, без закрытых API).",
            "freshness считается относительно самой свежей вакансии в наборе, а не реальной 'сегодня'.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(out_dir: str | Path, report_md: str, trace: dict, log_lines: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report_md, encoding="utf-8")
    (out / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


class RunLogger:
    """Простой логгер: пишет и в список (для run.log), и в stdout."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} | {msg}"
        self.lines.append(line)
        print(line)
