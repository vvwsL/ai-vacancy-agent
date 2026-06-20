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
import hashlib
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
import json

from .fetch_telegram import TelegramError, fetch_posts, normalize_channel
from . import llm as llm_mod
from .llm import CandidateProfile, analyze, extract_profile, extract_vacancies_from_text, llm_available
from .loader import LoaderError, load_vacancies
from .report import RunLogger, render_report, write_artifacts
from .resume import ResumeError, extract_text
from .scoring import filter_and_score
from . import ui
from .ui import Choice

FETCHED_FILE = "data/fetched_vacancies.json"  # сюда пишем вакансии из telegram/pdf (не трогаем мок)
PROFILE_CACHE = "data/.profile_cache.json"    # кэш LLM-профиля по хэшу резюме (экономия токенов)


def _get_profile(text: str, config, provider, api_key, log: RunLogger, refresh: bool) -> CandidateProfile:
    """Профиль из резюме с кэшированием по содержимому (не дёргаем LLM, если не менялось).

    Кэшируем только LLM-результат; rule-based (dryrun) бесплатен и не кэшируется.
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    cache: dict = {}
    p = Path(PROFILE_CACHE)
    if p.exists():
        try:
            cache = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if not refresh and h in cache:
        log.log("Профиль взят из кэша (резюме не менялось) — LLM не вызывается")
        return CandidateProfile(**cache[h])

    profile = extract_profile(text, config, provider, api_key)
    if provider != "dryrun":  # сохраняем только осмысленный LLM-профиль
        _save_profile_cache(text, profile)
        log.log("Профиль извлечён LLM и сохранён в кэш")
    return profile


def _save_profile_cache(text: str, profile: CandidateProfile) -> None:
    """Сохранить/обновить профиль в кэше по хэшу резюме (после правки тоже)."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    p = Path(PROFILE_CACHE)
    cache: dict = {}
    if p.exists():
        try:
            cache = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    cache[h] = profile.model_dump()
    try:
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


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

# Человекочитаемые подписи опыта (внутренние коды hh -> русский).
EXPERIENCE_LABELS = {
    "": "не указан",
    "noExperience": "нет опыта",
    "between1And3": "1–3 года",
    "between3And6": "3–6 лет",
    "moreThan6": "6+ лет",
}


def exp_label(code: str) -> str:
    return EXPERIENCE_LABELS.get(code, code or "не указан")


def target_levels(p: CandidateProfile) -> list[str]:
    """Уровни вакансий для поиска. Из опыта (код) если он есть, иначе из level LLM."""
    if p.experience in _EXPERIENCE_TO_LEVELS:
        return _EXPERIENCE_TO_LEVELS[p.experience]
    return p.level or ["intern", "junior"]


def _criteria_from_profile(p: CandidateProfile) -> Criteria:
    """Профиль из резюме -> критерии для scoring (обычный код).

    Всё в нижний регистр — scoring сравнивает с lowercase-полями вакансий
    (так же, как критерии из criteria.md). Иначе 'ML Engineer' != 'ml engineer'.
    """
    low = lambda xs: [str(x).lower() for x in xs]
    return Criteria(
        role=low(p.role) or ["ml engineer"],
        skills=low(p.skills) or ["python"],
        level=low(target_levels(p)),
        work_format=low(p.work_format),
        city=[p.city.lower()] if p.city else [],
    )


def _print_profile(profile: CandidateProfile, log: RunLogger) -> None:
    log.log("Агент понял профиль из резюме:")
    log.log(f"  роль:    {', '.join(profile.role) or '-'}")
    log.log(f"  опыт:    {exp_label(profile.experience)}")
    log.log(f"  уровень вакансий для поиска: {', '.join(target_levels(profile)) or '-'}")
    log.log(f"  навыки:  {', '.join(profile.skills) or '-'}")
    log.log(f"  формат:  {', '.join(profile.work_format) or '-'}")
    log.log(f"  город:   {profile.city or '-'}")


def _clean_profile(p: CandidateProfile) -> CandidateProfile:
    """Убрать мусор: пустые строки в списках, лишние пробелы."""
    p.role = [s.strip() for s in p.role if s and s.strip()]
    p.skills = [s.strip() for s in p.skills if s and s.strip()]
    p.level = [s.strip() for s in p.level if s and s.strip()]
    p.work_format = [s.strip() for s in p.work_format if s and s.strip()]
    p.city = (p.city or "").strip()
    p.experience = (p.experience or "").strip()
    return p


def _review_profile(profile: CandidateProfile, log: RunLogger) -> CandidateProfile:
    """Показать профиль и дать поправить его поля прямо в программе (если интерактивно)."""
    while True:
        print("\n--- Профиль кандидата (что понял агент) ---")
        print(f"  1) роль:    {', '.join(profile.role) or '-'}")
        print(f"  2) навыки:  {', '.join(profile.skills) or '-'}")
        print(f"  3) формат:  {', '.join(profile.work_format) or '-'}")
        print(f"  4) город:   {profile.city or '-'}")
        print(f"  5) опыт:    {exp_label(profile.experience)}  (уровень для поиска: {', '.join(target_levels(profile))})")
        try:
            ans = input("Enter — всё верно / номер поля — править: ").strip()
        except EOFError:
            return profile
        if not ans:
            return profile
        if ans == "1":
            v = input("Роли через запятую: ").strip()
            if v:
                profile.role = [x.strip() for x in v.split(",") if x.strip()]
        elif ans == "2":
            v = input("Навыки через запятую: ").strip()
            if v:
                profile.skills = [x.strip() for x in v.split(",") if x.strip()]
        elif ans == "3":
            v = input("Формат (remote/hybrid/office) через запятую: ").strip()
            profile.work_format = [x.strip() for x in v.split(",") if x.strip()]
        elif ans == "4":
            profile.city = input("Город: ").strip()
        elif ans == "5":
            codes = ["", "noExperience", "between1And3", "between3And6", "moreThan6"]
            val = ui.select("Опыт работы", [Choice(exp_label(c), c) for c in codes],
                            default=profile.experience)
            if val is not None:
                profile.experience = val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-агент подбора AI/ML вакансий по резюме")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", help="каталог артефактов (по умолч. runs/<дата>__<резюме>__<провайдер>)")
    p.add_argument("--input", default="data/vacancies.json", help="JSON-файл вакансий (source=file)")
    p.add_argument("--source", choices=["file", "telegram", "pdf"], default="file",
                   help="источник вакансий (одиночный; для нескольких см. --sources)")
    p.add_argument("--sources", default="",
                   help="несколько источников через запятую, напр. file,telegram")
    p.add_argument("--tg-channels", dest="tg_channels", default="",
                   help="каналы Telegram через запятую (ссылки/имена), для --source telegram")
    p.add_argument("--tg-limit", dest="tg_limit", type=int, default=20,
                   help="сколько постов на канал тянуть")
    p.add_argument("--pdf-vacancies", dest="pdf_vacancies", default="",
                   help="путь к PDF/txt с вакансиями, для --source pdf")
    p.add_argument("--resume", help="файл резюме (pdf/docx/txt/md); по умолч. из config")
    p.add_argument("--criteria", default="criteria.md", help="критерии (fallback, если нет резюме)")
    p.add_argument("--top-n", type=int, dest="top_n", help="сколько вакансий разбирать (топ по score)")
    p.add_argument("--dry-run", action="store_true", help="без LLM (rule-based, детерминированно)")
    p.add_argument("--refresh-profile", action="store_true", dest="refresh_profile",
                   help="заново извлечь профиль из резюме, игнорируя кэш")
    p.add_argument("--yes", action="store_true", help="не спрашивать подтверждение (для тестов/CI)")
    p.add_argument("--provider",
                   choices=["auto", "gemini", "groq", "openrouter", "cerebras", "mistral", "cohere", "dryrun"])
    p.add_argument("--model-openrouter", dest="model_openrouter", default="",
                   help="модель OpenRouter (напр. deepseek/deepseek-chat-v3-0324:free)")
    return p


def parse_sources(args) -> list[str]:
    """Список источников: --sources (через запятую) или одиночный --source."""
    raw = args.sources.split(",") if args.sources else [args.source]
    seen, out = set(), []
    for s in (x.strip() for x in raw):
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out or ["file"]


def _records_from_telegram(args, config, provider, api_key, log: RunLogger) -> list[dict]:
    channels = [c.strip() for c in args.tg_channels.split(",") if c.strip()]
    if not channels:
        log.log("  telegram пропущен: не заданы каналы")
        return []
    records: list[dict] = []
    # По каждому каналу отдельно — иначе при склейке текст обрезается и теряются хвосты.
    for ch in channels:
        name = normalize_channel(ch)
        try:
            pairs = fetch_posts(ch, limit=args.tg_limit)
        except TelegramError as e:
            log.log(f"  пропускаю канал {name}: {e}")
            continue
        # В каждый блок кладём ссылку поста, чтобы LLM вернул реальный url вакансии.
        blocks = [f"URL: {url}\n{text}" if url else text for text, url in pairs]
        recs = extract_vacancies_from_text(blocks, f"tg-{name}", config, provider, api_key)
        log.log(f"Telegram @{name}: постов {len(pairs)} -> вакансий {len(recs)}")
        records.extend(recs)
    return records


def _records_from_pdf(args, config, provider, api_key, log: RunLogger) -> list[dict]:
    path = args.pdf_vacancies
    if not path or not Path(path).exists():
        log.log(f"  pdf пропущен: не найден файл {path}")
        return []
    text = extract_text(path)  # переиспользуем парсер резюме
    blocks = [b for b in text.split("\n\n") if b.strip()]
    return extract_vacancies_from_text(blocks or [text], "pdf", config, provider, api_key)


def gather_input(args, sources: list[str], config, provider, api_key, log: RunLogger) -> str:
    """Собрать вакансии из всех выбранных источников в один файл. Вернуть путь.

    Только file -> используем args.input как есть (мок не дублируем).
    Иначе -> объединяем записи в FETCHED_FILE (дедуп сделает loader).
    """
    if sources == ["file"]:
        return args.input

    records: list[dict] = []
    if "file" in sources:
        try:
            data = json.loads(Path(args.input).read_text(encoding="utf-8"))
            if isinstance(data, list):
                records += data
                log.log(f"Файл {args.input}: добавлено записей {len(data)}")
        except Exception as e:
            log.log(f"  файл пропущен: {e}")
    if "telegram" in sources:
        records += _records_from_telegram(args, config, provider, api_key, log)
    if "pdf" in sources:
        records += _records_from_pdf(args, config, provider, api_key, log)

    if not records:
        raise LoaderError(f"из источников {sources} не извлечено ни одной вакансии")

    Path(FETCHED_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(FETCHED_FILE).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.log(f"Источники {sources}: всего записей {len(records)} -> {FETCHED_FILE}")
    return FETCHED_FILE


def _fail(output: str, log: RunLogger, msg: str) -> int:
    log.log(f"ОШИБКА: {msg}")
    write_artifacts(output, f"# Прогон прерван\n\nОшибка: {msg}\n", {"error": msg}, log.lines)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _load_dotenv()
    llm_mod.FALLBACKS.clear()  # сброс журнала переключений провайдера
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = RunLogger()
    log.log("=== Запуск агента ===")

    # --- Конфиг ---
    try:
        config = load_config(args.config)
    except ConfigError as e:
        return _fail(args.output or "output", log, str(e))

    if args.model_openrouter:  # переопределить модель OpenRouter на этот прогон
        config.setdefault("llm", {})["model_openrouter"] = args.model_openrouter

    resume_path = args.resume or config.get("resume_path", "")

    # --- Провайдер (нужен до извлечения профиля) ---
    provider = resolve_provider(config, args.provider, args.dry_run)
    api_key = api_key_for(provider)
    if provider != "dryrun" and not api_key:
        log.log(f"Ключ для '{provider}' не найден -> офлайн-режим (без LLM)")
        provider = "dryrun"
    # Проверяем, что LLM реально отвечает (мёртвая квота/ключ -> заранее в офлайн).
    if provider != "dryrun":
        ok, why = llm_available(config, provider, api_key)
        if not ok:
            log.log(f"LLM '{provider}' недоступен ({why}) -> офлайн-режим (без LLM)")
            provider, api_key = "dryrun", None
    log.log(f"Провайдер LLM: {provider}")
    sources = parse_sources(args)
    log.log(f"Источники вакансий: {', '.join(sources)}")
    if provider == "dryrun" and ({"telegram", "pdf"} & set(sources)):
        log.log("ВНИМАНИЕ: источник telegram/pdf без LLM даёт грубый результат "
                "(не извлекаются компания/стек). Для нормального разбора нужен LLM-ключ.")

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
            profile = _get_profile(text, config, provider, api_key, log, args.refresh_profile)
        except ResumeError as e:
            return _fail(out_dir, log, f"резюме: {e}")
        profile = _clean_profile(profile)
        _print_profile(profile, log)
        if not args.yes:
            before = profile.model_dump()
            profile = _clean_profile(_review_profile(profile, log))
            if profile.model_dump() != before:   # правили — обновим кэш
                _save_profile_cache(text, profile)
                log.log("Профиль отредактирован вручную и сохранён в кэш")
        criteria = _criteria_from_profile(profile)
    else:
        try:
            criteria = load_criteria(args.criteria)
            log.log(f"Резюме не задано -> критерии из {args.criteria}")
        except ConfigError as e:
            return _fail(out_dir, log, str(e))

    # --- Источники вакансий (file / telegram / pdf, можно несколько) ---
    try:
        input_path = gather_input(args, sources, config, provider, api_key, log)
    except LoaderError as e:
        return _fail(out_dir, log, str(e))

    # --- Загрузка + валидация вакансий ---
    try:
        loaded = load_vacancies(input_path)
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
    for rank, sc in enumerate(top, 1):
        log.log(f"Разбор {sc.vacancy.id} ({sc.vacancy.title}) score={sc.score} ...")
        res = analyze(sc, criteria, config, provider, api_key)
        log.log(f"  режим: {res['mode']}; приоритет: {res['analysis'].priority}")
        # score_rank — место по числовой метрике (до переупорядочивания агентом).
        analyses.append({"scored": sc, "analysis": res["analysis"], "mode": res["mode"], "score_rank": rank})
        traces.append(res["trace"])

    # Гибрид: метрика дала шорт-лист, агент переупорядочивает его по своему вердикту
    # (приоритет отклика), при равенстве — по score. Расхождение покажем в отчёте.
    _PRIO_ORDER = {"high": 0, "medium": 1, "low": 2}
    analyses.sort(key=lambda a: (_PRIO_ORDER.get(a["analysis"].priority, 1), -a["scored"].score))

    if llm_mod.FALLBACKS:
        uniq = sorted(set(llm_mod.FALLBACKS))
        log.log(f"Failover провайдера сработал: {', '.join(uniq)}")

    # --- Артефакты ---
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "source": ", ".join(sources),
        "raw_count": loaded.raw_count,
        "valid_count": len(loaded.vacancies),
        "rejected_rows": len(loaded.rejected_rows),
        "duplicates": len(loaded.duplicates),
        "scored_count": len(scored),
        "min_score": config.get("min_relevant_score", 45),
    }
    profile_info = None
    if profile:
        profile_info = [
            f"роль: {', '.join(profile.role) or '-'}",
            f"уровень для поиска: {', '.join(target_levels(profile)) or '-'} (опыт: {exp_label(profile.experience)})",
            f"навыки: {', '.join(profile.skills) or '-'}",
            f"формат: {', '.join(profile.work_format) or '-'} | город: {profile.city or '-'}",
        ]
    report_md = render_report(analyses, rejected_hard, borderline, meta, profile_info)
    meta["fallbacks"] = sorted(set(llm_mod.FALLBACKS))
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
