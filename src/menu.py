"""Интерактивное терминальное меню — лёгкий запуск без флагов.

Запуск:  python -m src.menu   (или двойной клик по start.bat на Windows)
Это тонкая обёртка над src.main: меню собирает аргументы и вызывает main().
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from . import main as main_mod
from .config import load_config

PROVIDERS = ["auto", "gemini", "groq", "openrouter", "cerebras", "dryrun"]
RESUME_EXTS = {".pdf", ".docx", ".txt", ".md"}


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _list_resumes() -> list[Path]:
    data = Path("data")
    if not data.exists():
        return []
    return sorted(p for p in data.iterdir() if p.suffix.lower() in RESUME_EXTS)


def _choose_resume(current: str) -> str:
    files = _list_resumes()
    print("\nРезюме в папке data/:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {f.name}")
    print("  0) ввести путь вручную")
    ans = input(f"Выбор [{current}]: ").strip()
    if not ans:
        return current
    if ans == "0":
        return input("Путь к резюме: ").strip() or current
    if ans.isdigit() and 1 <= int(ans) <= len(files):
        return str(files[int(ans) - 1])
    print("Не понял, оставляю прежнее.")
    return current


def _choose_provider(current: str) -> str:
    print("\nПровайдер LLM (auto = первый, для которого есть ключ в .env):")
    for i, p in enumerate(PROVIDERS, 1):
        print(f"  {i}) {p}")
    ans = input(f"Выбор [{current}]: ").strip()
    if ans.isdigit() and 1 <= int(ans) <= len(PROVIDERS):
        return PROVIDERS[int(ans) - 1]
    return current


def _print_summary(run_dir: Path) -> None:
    """Краткая сводка прогона из готового report.md (без LLM): шапка + топ."""
    report = run_dir / "report.md"
    if not report.exists():
        print("\nОтчёта нет (возможно, прогон прервался). Хвост run.log:")
        log = run_dir / "run.log"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines()[-8:]:
                print("  " + line)
        return
    lines = report.read_text(encoding="utf-8").splitlines()
    print(f"\n=== Сводка: {run_dir.name} ===")
    # Шапка: пункты "- ..." до первого "## ".
    for ln in lines:
        if ln.startswith("## "):
            break
        if ln.startswith("- "):
            print(ln)
    # Топ: строки "### " + следующая строка со Score.
    print("\nТоп:")
    for i, ln in enumerate(lines):
        if ln.startswith("### "):
            title = ln[4:].strip()
            score = ""
            if i + 1 < len(lines) and "Score" in lines[i + 1]:
                score = lines[i + 1].replace("*", "").strip()
            print(f"  • {title}  ({score})")


def _open_in_os(path: Path) -> None:
    """Открыть файл в системном приложении по умолчанию (ОС сама выберет/предложит)."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        print(f"Открываю {path} в системном приложении...")
    except Exception as e:
        print(f"Не удалось открыть автоматически: {e}\nФайл лежит здесь: {path}")


def _history_menu() -> None:
    runs = Path("runs")
    dirs = sorted((d for d in runs.iterdir() if d.is_dir()), reverse=True) if runs.exists() else []
    if not dirs:
        print("\nПрогонов пока нет (папка runs/ пуста).")
        return
    print("\nПоследние прогоны (новые сверху):")
    shown = dirs[:20]
    for i, d in enumerate(shown, 1):
        print(f"  {i}) {d.name}")
    print("  0) назад")
    ans = input("Выбери прогон: ").strip()
    if not ans.isdigit() or not (1 <= int(ans) <= len(shown)):
        return
    run_dir = shown[int(ans) - 1]

    while True:
        print(f"\nПрогон: {run_dir.name}")
        print("  1) Краткая сводка (без LLM)")
        print("  2) Открыть отчёт в приложении (Блокнот и т.п.)")
        print("  0) назад")
        c = input("Выбор: ").strip()
        if c == "1":
            _print_summary(run_dir)
        elif c == "2":
            _open_in_os(run_dir / "report.md")
        elif c == "0":
            return


def run_menu() -> int:
    _utf8_stdout()
    cfg: dict = {}
    try:
        cfg = load_config("config.yaml")
    except Exception:
        pass

    state = {
        "resume": cfg.get("resume_path", "data/resume.txt"),
        "provider": "auto",
        "top_n": cfg.get("top_n", 5),
        "dry_run": False,
    }

    while True:
        print("\n" + "=" * 44)
        print("   AI-агент подбора AI/ML вакансий по резюме")
        print("=" * 44)
        print(f"  1) Резюме:            {state['resume']}")
        print(f"  2) Провайдер LLM:     {state['provider']}")
        print(f"  3) Топ-N вакансий:    {state['top_n']}")
        print(f"  4) Режим разбора:     {'без LLM (dry-run)' if state['dry_run'] else 'с LLM'}")
        print("  5) > Запустить")
        print("  6) История прогонов")
        print("  0) Выход")
        # Ctrl+C на главном приглашении = выход; во время действия = отмена (см. ниже).
        try:
            choice = input("Выбери пункт: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока!")
            return 0

        try:
            if choice == "1":
                state["resume"] = _choose_resume(state["resume"])
            elif choice == "2":
                state["provider"] = _choose_provider(state["provider"])
            elif choice == "3":
                v = input("Топ-N (число): ").strip()
                if v.isdigit() and int(v) > 0:
                    state["top_n"] = int(v)
            elif choice == "4":
                ans = input("Использовать LLM? [y = с LLM / n = без LLM (dry-run)]: ").strip().lower()
                if ans:
                    state["dry_run"] = not (ans.startswith("y") or ans.startswith("д"))
            elif choice == "5":
                argv = [
                    "--resume", state["resume"],
                    "--provider", state["provider"],
                    "--top-n", str(state["top_n"]),
                ]
                if state["dry_run"]:
                    argv.append("--dry-run")
                print("\n--- запуск ---")
                main_mod.main(argv)
                input("\nEnter — вернуться в меню...")
            elif choice == "6":
                _history_menu()
            elif choice == "0":
                print("Пока!")
                return 0
            else:
                print("Не понял выбор, попробуй ещё раз.")
        except KeyboardInterrupt:
            print("\n(действие отменено, возврат в меню)")


if __name__ == "__main__":
    try:
        sys.exit(run_menu())
    except KeyboardInterrupt:
        print("\nПока!")
        sys.exit(0)
