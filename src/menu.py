"""Интерактивное терминальное меню — лёгкий запуск без флагов.

Запуск:  python -m src.menu   (или двойной клик по start.bat на Windows)
Тонкая обёртка над src.main: меню собирает аргументы и вызывает main().
Настройки сохраняются между запусками в user_settings.json.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from . import main as main_mod
from .config import load_config

PROVIDERS = ["auto", "gemini", "groq", "openrouter", "cerebras", "dryrun"]
RESUME_EXTS = {".pdf", ".docx", ".txt", ".md"}
SETTINGS_FILE = "user_settings.json"


# --------------------------------------------------------------------------- #
# Экран: очистка + «хлебная крошка»
# --------------------------------------------------------------------------- #
def _clear() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass


def screen(breadcrumb: str) -> None:
    _utf8_stdout()
    _clear()
    bar = "=" * 48
    print(bar)
    print("  AI-агент подбора AI/ML вакансий")
    print(f"  {breadcrumb}")
    print(bar)


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Сохранение настроек
# --------------------------------------------------------------------------- #
def _default_state() -> dict:
    cfg: dict = {}
    try:
        cfg = load_config("config.yaml")
    except Exception:
        pass
    return {
        "resume": cfg.get("resume_path", "data/resume.txt"),
        "provider": "auto",
        "top_n": cfg.get("top_n", 5),
        "dry_run": False,
        "sources": ["file"],
        "tg_channels": [],
        "tg_limit": 20,
        "pdf_vacancies": "",
    }


def load_state() -> dict:
    state = _default_state()
    p = Path(SETTINGS_FILE)
    if p.exists():
        try:
            state.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    # Миграция со старого одиночного "source".
    if "sources" not in state or not isinstance(state.get("sources"), list):
        state["sources"] = [state.get("source", "file")]
    state.pop("source", None)
    return state


def save_state(state: dict) -> None:
    try:
        Path(SETTINGS_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Выбор резюме / провайдера
# --------------------------------------------------------------------------- #
def _list_resumes() -> list[Path]:
    data = Path("data")
    if not data.exists():
        return []
    return sorted(p for p in data.iterdir() if p.suffix.lower() in RESUME_EXTS)


def _preview_resume(path: str) -> None:
    """Показать первые содержательные строки резюме (без LLM)."""
    try:
        from .resume import extract_text
        text = extract_text(path)
    except Exception as e:
        print(f"\n(предпросмотр недоступен: {e})")
        return
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:12]
    print("\nПредпросмотр текущего резюме (ключевые строки):")
    for ln in lines:
        print("  | " + ln[:78])
    print("  ...")


def _choose_resume(current: str) -> str:
    screen("Меню › Резюме")
    print(f"Текущее резюме: {current}")
    _preview_resume(current)
    files = _list_resumes()
    print("\nРезюме в папке data/:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {f.name}")
    print("  0) ввести путь вручную")
    ans = input(f"\nВыбор [{current}, Enter — оставить]: ").strip()
    if not ans:
        return current
    if ans == "0":
        return input("Путь к резюме: ").strip() or current
    if ans.isdigit() and 1 <= int(ans) <= len(files):
        return str(files[int(ans) - 1])
    return current


def _choose_provider(current: str) -> str:
    screen("Меню › Провайдер LLM")
    print("auto = первый, для которого есть ключ в .env\n")
    for i, p in enumerate(PROVIDERS, 1):
        print(f"  {i}) {p}")
    ans = input(f"\nВыбор [{current}]: ").strip()
    if ans.isdigit() and 1 <= int(ans) <= len(PROVIDERS):
        return PROVIDERS[int(ans) - 1]
    return current


# --------------------------------------------------------------------------- #
# Источник вакансий
# --------------------------------------------------------------------------- #
def _enter_channels(current: list[str]) -> list[str]:
    channels = list(current)
    while True:
        screen("Меню › Источник › Telegram-каналы")
        print("Текущие каналы:")
        for c in channels:
            print(f"  • {c}")
        if not channels:
            print("  (пусто)")
        print("\nВведи ссылку на канал (напр. https://t.me/tagir_analyzes) и Enter.")
        print("Команды: 'clear' — очистить список, пустой Enter — назад.")
        ans = input("Канал: ").strip()
        if not ans:
            return channels
        if ans.lower() == "clear":
            channels = []
            continue
        channels.append(ans)


def _toggle(state: dict, src: str) -> None:
    s = state["sources"]
    if src in s:
        s.remove(src)
    else:
        s.append(src)
    if not s:                 # хотя бы один источник должен остаться
        s.append("file")


def _choose_source(state: dict) -> None:
    """Множественный выбор источников: можно включить несколько сразу."""
    while True:
        s = state["sources"]
        screen("Меню › Источники вакансий (можно несколько)")
        print(f"  1) [{'x' if 'file' in s else ' '}] Локальный файл (data/vacancies.json)")
        print(f"  2) [{'x' if 'telegram' in s else ' '}] Telegram-каналы ({len(state['tg_channels'])} шт.)")
        print(f"  3) [{'x' if 'pdf' in s else ' '}] PDF/текст ({state['pdf_vacancies'] or 'путь не задан'})")
        print("  4) Изменить список Telegram-каналов")
        print("  5) Указать путь к PDF/txt")
        print("  0) ← назад   (или Enter)")
        c = input("\nВыбор (цифра = вкл/выкл): ").strip()
        if c == "1":
            _toggle(state, "file")
        elif c == "2":
            _toggle(state, "telegram")
        elif c == "3":
            _toggle(state, "pdf")
        elif c == "4":
            state["tg_channels"] = _enter_channels(state["tg_channels"])
        elif c == "5":
            screen("Меню › Источники › PDF/текст")
            path = input(f"Путь к PDF/txt с вакансиями [{state['pdf_vacancies']}]: ").strip()
            if path:
                state["pdf_vacancies"] = path
        else:
            return


def _source_label(state: dict) -> str:
    parts = []
    for src in state["sources"]:
        if src == "telegram":
            parts.append(f"telegram({len(state['tg_channels'])})")
        elif src == "pdf":
            parts.append("pdf")
        else:
            parts.append("файл")
    return " + ".join(parts) or "файл"


# --------------------------------------------------------------------------- #
# История прогонов
# --------------------------------------------------------------------------- #
def _print_summary(run_dir: Path) -> None:
    report = run_dir / "report.md"
    if not report.exists():
        print("\nОтчёта нет (прогон прервался). Хвост run.log:")
        log = run_dir / "run.log"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines()[-8:]:
                print("  " + line)
        return
    lines = report.read_text(encoding="utf-8").splitlines()
    print(f"\n=== Сводка: {run_dir.name} ===")
    for ln in lines:
        if ln.startswith("## "):
            break
        if ln.startswith("- "):
            print(ln)
    print("\nТоп:")
    for i, ln in enumerate(lines):
        if ln.startswith("### "):
            title = ln[4:].strip()
            score = lines[i + 1].replace("*", "").strip() if i + 1 < len(lines) and "Score" in lines[i + 1] else ""
            print(f"  • {title}  ({score})")


def _open_in_os(path: Path) -> None:
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        print(f"Открываю {path} ...")
    except Exception as e:
        print(f"Не удалось открыть: {e}\nФайл: {path}")


def _latest_run() -> Path | None:
    runs = Path("runs")
    dirs = sorted((d for d in runs.iterdir() if d.is_dir()), reverse=True) if runs.exists() else []
    return dirs[0] if dirs else None


def _run_actions(run_dir: Path, title: str) -> None:
    """Подменю над одним прогоном: сводка / открыть отчёт. Enter — назад."""
    while True:
        screen(title)
        print(f"Прогон: {run_dir.name}")
        print(f"Папка с логами: {run_dir}")
        print("\n  1) Краткая сводка (без LLM)")
        print("  2) Открыть отчёт в приложении (Блокнот и т.п.)")
        print("  0) ← назад   (или Enter)")
        c = input("\nВыбор: ").strip()
        if c == "1":
            _print_summary(run_dir)
            input("\nEnter — назад...")
        elif c == "2":
            _open_in_os(run_dir / "report.md")
            input("\nEnter — назад...")
        else:
            return


def _history_menu() -> None:
    screen("Меню › История прогонов")
    runs = Path("runs")
    dirs = sorted((d for d in runs.iterdir() if d.is_dir()), reverse=True) if runs.exists() else []
    if not dirs:
        print("Прогонов пока нет (папка runs/ пуста).")
        input("\nEnter — назад...")
        return
    shown = dirs[:20]
    for i, d in enumerate(shown, 1):
        print(f"  {i}) {d.name}")
    print("  0) ← назад")
    ans = input("\nВыбери прогон: ").strip()
    if not ans.isdigit() or not (1 <= int(ans) <= len(shown)):
        return
    run_dir = shown[int(ans) - 1]
    _run_actions(run_dir, "Меню › История › прогон")


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #
def _build_argv(state: dict) -> list[str]:
    argv = [
        "--resume", state["resume"],
        "--provider", state["provider"],
        "--top-n", str(state["top_n"]),
        "--sources", ",".join(state["sources"]),
    ]
    if "telegram" in state["sources"]:
        argv += ["--tg-channels", ",".join(state["tg_channels"]), "--tg-limit", str(state["tg_limit"])]
    if "pdf" in state["sources"]:
        argv += ["--pdf-vacancies", state["pdf_vacancies"]]
    if state["dry_run"]:
        argv.append("--dry-run")
    return argv


def run_menu() -> int:
    state = load_state()
    while True:
        screen("Главное меню")
        print(f"  1) Резюме:            {state['resume']}")
        print(f"  2) Источник вакансий: {_source_label(state)}")
        print(f"  3) Провайдер LLM:     {state['provider']}")
        print(f"  4) Топ-N вакансий:    {state['top_n']}")
        print(f"  5) Режим разбора:     {'без LLM (dry-run)' if state['dry_run'] else 'с LLM'}")
        print("  6) > Запустить   (или просто Enter)")
        print("  7) История прогонов")
        print("  0) Выход")
        try:
            choice = input("\nВыбери пункт [Enter = запустить]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока!")
            return 0
        if not choice:
            choice = "6"  # Enter = шаг вперёд = запустить

        try:
            if choice == "1":
                state["resume"] = _choose_resume(state["resume"])
            elif choice == "2":
                _choose_source(state)
            elif choice == "3":
                state["provider"] = _choose_provider(state["provider"])
            elif choice == "4":
                v = input("Топ-N (число): ").strip()
                if v.isdigit() and int(v) > 0:
                    state["top_n"] = int(v)
            elif choice == "5":
                ans = input("Использовать LLM? [y = с LLM / n = без LLM (dry-run)]: ").strip().lower()
                if ans:
                    state["dry_run"] = not (ans.startswith("y") or ans.startswith("д"))
            elif choice == "6":
                save_state(state)
                screen("Запуск")
                main_mod.main(_build_argv(state))
                rd = _latest_run()
                if rd:
                    input("\nEnter — посмотреть результат...")
                    _run_actions(rd, "Результат прогона")
                else:
                    input("\nEnter — вернуться в меню...")
            elif choice == "7":
                _history_menu()
            elif choice == "0":
                save_state(state)
                print("Пока!")
                return 0
            else:
                continue
            save_state(state)
        except KeyboardInterrupt:
            print("\n(действие отменено, возврат в меню)")
            input("Enter — продолжить...")


if __name__ == "__main__":
    try:
        sys.exit(run_menu())
    except KeyboardInterrupt:
        print("\nПока!")
        sys.exit(0)
