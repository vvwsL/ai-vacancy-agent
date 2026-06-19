"""CLI и оркестрация pipeline.

Поток:
  резюме -> [парсер] текст -> [LLM] профиль -> подтверждение -> вакансии(файл) -> разбор -> отчёт

Примеры:
  python -m src.main                                       # всё из config.yaml
  python -m src.main --resume cv.pdf --provider gemini     # живой режим
  python -m src.main --dry-run --yes                       # без LLM, детерминированно
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import (
    ConfigError,
    Criteria,
    api_key_for,
    load_config,
    load_criteria,
    resolve_provider,
)
from .llm import CandidateProfile, analyze, extract_profile
from .loader import LoaderError, load_vacancies
from .report import RunLogger, render_report, write_artifacts
from .resume import ResumeError, extract_text
from .scoring import filter_and_score


def _load_dotenv(path: str = ".env") -> None:
    """Минимальный загрузчик .env (без зависимости python-dotenv)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and val and key not in os.environ:
            os.environ[key] = val


# Опыт -> уровни ИСКОМЫХ вакансий (детерминированно, не доверяем угадыванию LLM).
# Это уровень позиций для поиска, а НЕ собственная сеньорность кандидата.
_EXPERIENCE_TO_LEVELS = {
    "noExperience": ["intern", "junior"],
    "between1And3": ["junior", "middle"],
    "between3And6": ["middle", "senior"],
    "moreThan6": ["senior", "lead"],
}


def target_levels(p: CandidateProfile) -> list[str]:
    """Уровни вакансий для поиска. Из опыта (код) если он есть, иначе из level LLM."""
    if p.experience in _EXPERIENCE_TO_LEVELS:
        return _EXPERIENCE_TO_LEVELS[p.experience]
    return p.level or ["intern", "junior"]


def _criteria_from_profile(p: CandidateProfile) -> Criteria:
    """Профиль из резюме -> критерии для scoring (обычный код)."""
    return Criteria(
        role=p.role or ["ml engineer"],
        skills=p.skills or ["python"],
        level=target_levels(p),
        work_format=p.work_format,
        city=[p.city] if p.city else [],
    )


def _print_profile(profile: CandidateProfile, log: RunLogger) -> None:
    log.log("Агент понял профиль из резюме:")
    log.log(f"  роль:    {', '.join(profile.role) or '-'}")
    log.log(f"  опыт:    {profile.experience or '(не задан)'}")
    log.log(f"  уровень вакансий для поиска: {', '.join(target_levels(profile)) or '-'}")
    log.log(f"  навыки:  {', '.join(profile.skills) or '-'}")
    log.log(f"  формат:  {', '.join(profile.work_format) or '-'}")
    log.log(f"  город:   {profile.city or '-'}")


def _confirm_profile(assume_yes: bool, log: RunLogger) -> None:
    """Дать подтвердить профиль (если не --yes). Правка — через резюме/config + перезапуск."""
    if assume_yes:
        return
    try:
        input("\nПрофиль верный? [Enter — продолжить / Ctrl+C — выйти и поправить резюме]: ")
    except EOFError:
        return  # неинтерактивный stdin


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-агент подбора AI/ML вакансий по резюме")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", help="каталог артефактов (по умолч. runs/<дата>__<резюме>__<провайдер>)")
    p.add_argument("--input", default="data/vacancies.json", help="JSON-файл вакансий")
    p.add_argument("--resume", help="файл резюме (pdf/docx/txt/md); по умолч. из config")
    p.add_argument("--criteria", default="criteria.md", help="критерии (fallback, если нет резюме)")
    p.add_argument("--top-n", type=int, dest="top_n", help="сколько вакансий разбирать (топ по score)")
    p.add_argument("--dry-run", action="store_true", help="без LLM (rule-based, детерминированно)")
    p.add_argument("--yes", action="store_true", help="не спрашивать подтверждение (для тестов/CI)")
    p.add_argument("--provider", choices=["auto", "gemini", "groq", "openrouter", "cerebras", "dryrun"])
    return p


def _fail(output: str, log: RunLogger, msg: str) -> int:
    log.log(f"ОШИБКА: {msg}")
    write_artifacts(output, f"# Прогон прерван\n\nОшибка: {msg}\n", {"error": msg}, log.lines)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _load_dotenv()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = RunLogger()
    log.log("=== Запуск агента ===")

    # --- Конфиг ---
    try:
        config = load_config(args.config)
    except ConfigError as e:
        return _fail(args.output or "output", log, str(e))

    resume_path = args.resume or config.get("resume_path", "")

    # --- Провайдер (нужен до извлечения профиля) ---
    provider = resolve_provider(config, args.provider, args.dry_run)
    api_key = api_key_for(provider)
    if provider != "dryrun" and not api_key:
        log.log(f"Ключ для '{provider}' не найден -> dry-run")
        provider = "dryrun"
    log.log(f"Провайдер LLM: {provider}")

    # --- Каталог прогона: runs/<дата>__<резюме>__<провайдер>, если --output не задан ---
    if args.output:
        out_dir = args.output
    else:
        stem = Path(resume_path).stem if resume_path else "criteria"
        out_dir = str(Path("runs") / f"{run_stamp}__{stem}__{provider}")
    log.log(f"Каталог прогона: {out_dir}")

    # --- Критерии: из резюме (LLM) или из criteria.md (fallback) ---
    profile: CandidateProfile | None = None
    if resume_path and Path(resume_path).exists():
        try:
            text = extract_text(resume_path)
            log.log(f"Резюме прочитано: {resume_path} ({len(text)} симв.)")
            profile = extract_profile(text, config, provider, api_key)
        except ResumeError as e:
            return _fail(out_dir, log, f"резюме: {e}")
        criteria = _criteria_from_profile(profile)
        _print_profile(profile, log)
        _confirm_profile(args.yes, log)
    else:
        try:
            criteria = load_criteria(args.criteria)
            log.log(f"Резюме не задано -> критерии из {args.criteria}")
        except ConfigError as e:
            return _fail(out_dir, log, str(e))

    # --- Загрузка + валидация вакансий ---
    try:
        loaded = load_vacancies(args.input)
    except LoaderError as e:
        return _fail(out_dir, log, str(e))
    log.log(
        f"Загружено: {loaded.raw_count} | валидных: {len(loaded.vacancies)} | "
        f"битых: {len(loaded.rejected_rows)} | дублей: {len(loaded.duplicates)}"
    )

    # --- Hard-фильтр + scoring (обычная логика) ---
    scored, rejected_hard = filter_and_score(loaded.vacancies, criteria, config)
    log.log(f"После hard-фильтра: {len(scored)} (отсеяно по уровню: {len(rejected_hard)})")

    # --- Агентный разбор топ-N ---
    top_n = args.top_n or config.get("top_n", 5)
    top = scored[:top_n]
    borderline = scored[top_n : top_n + config.get("explain_rejected_n", 3)]

    analyses, traces = [], []
    for sc in top:
        log.log(f"Разбор {sc.vacancy.id} ({sc.vacancy.title}) score={sc.score} ...")
        res = analyze(sc, criteria, config, provider, api_key)
        log.log(f"  режим: {res['mode']}; приоритет: {res['analysis'].priority}")
        analyses.append({"scored": sc, "analysis": res["analysis"], "mode": res["mode"]})
        traces.append(res["trace"])

    # --- Артефакты ---
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "source": "file",
        "raw_count": loaded.raw_count,
        "valid_count": len(loaded.vacancies),
        "rejected_rows": len(loaded.rejected_rows),
        "duplicates": len(loaded.duplicates),
        "scored_count": len(scored),
    }
    report_md = render_report(analyses, rejected_hard, borderline, meta)
    full_trace = {
        "meta": meta,
        "profile": profile.model_dump() if profile else None,
        "rejected_rows": loaded.rejected_rows,
        "duplicates": loaded.duplicates,
        "hard_rejected": [{"id": r.vacancy.id, "reason": r.reason} for r in rejected_hard],
        "ranking": [{"id": s.vacancy.id, "score": s.score, "components": s.components} for s in scored],
        "analyses": traces,
    }
    log.log(f"Готово. Результат: {out_dir}/report.md (+ trace.json, run.log)")
    write_artifacts(out_dir, report_md, full_trace, log.lines)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем (Ctrl+C).")
        sys.exit(130)
