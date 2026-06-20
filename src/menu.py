"""Интерактивное меню на questionary (через тонкую обёртку ui.py).

Запуск:  python -m src.menu   (или двойной клик по start.bat на Windows)
Навигация — стрелки/Enter (questionary). «Назад»/«Выход» — явными пунктами меню.
Настройки сохраняются в .menu_state.json.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from . import main as main_mod
from . import ui
from .config import PROVIDER_KEY_ENV, load_config
from .ui import Choice

RESUME_EXTS = {".pdf", ".docx", ".txt", ".md"}
STATE_PATH = ".menu_state.json"

# Курируемые бесплатные модели OpenRouter (подпись, value). value="" = дефолт провайдера.
_OPENROUTER_MODELS = [
    ("по умолчанию (llama-3.3-70b-instruct)", ""),
    ("deepseek-chat-v3-0324:free", "deepseek/deepseek-chat-v3-0324:free"),
    ("gemini-2.0-flash-exp:free", "google/gemini-2.0-flash-exp:free"),
    ("qwen-2.5-72b-instruct:free", "qwen/qwen-2.5-72b-instruct:free"),
    ("llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    ("[ввести вручную]", "__manual__"),
]


# --------------------------------------------------------------------------- #
# Состояние меню (.menu_state.json)
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
        "openrouter_model": "",
    }


def load_state() -> dict:
    state = _default_state()
    for path in (STATE_PATH, "user_settings.json"):  # вторая — старое имя, для совместимости
        p = Path(path)
        if p.exists():
            try:
                state.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
            break
    if not isinstance(state.get("sources"), list):
        state["sources"] = ["file"]
    state.pop("source", None)
    return state


def save_state(state: dict) -> None:
    try:
        Path(STATE_PATH).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Подменю
# --------------------------------------------------------------------------- #
def _list_resumes() -> list[Path]:
    data = Path("data")
    return sorted(p for p in data.iterdir() if p.suffix.lower() in RESUME_EXTS) if data.exists() else []


def _choose_resume(state: dict) -> None:
    files = _list_resumes()
    choices = [Choice(f.name, str(f)) for f in files] + [Choice("[ввести путь вручную]", "__manual__")]
    val = ui.select(f"Резюме (текущее: {state['resume']})", choices, default=state["resume"])
    if val is None:
        return
    state["resume"] = ui.ask_text("Путь к резюме", state["resume"]) if val == "__manual__" else val


def _available_providers() -> list[Choice]:
    """auto + провайдеры с ключом в .env + dryrun (как Choice со значением)."""
    out = [Choice("auto (первый с ключом)", "auto")]
    for p, env in PROVIDER_KEY_ENV.items():
        if os.environ.get(env):
            out.append(Choice(p, p))
    out.append(Choice("dryrun (без LLM)", "dryrun"))
    return out


def _choose_provider(state: dict) -> None:
    val = ui.select("Провайдер LLM", _available_providers(), default=state["provider"])
    if val is None:
        return
    state["provider"] = val
    if val == "openrouter":
        _choose_openrouter_model(state)


def _choose_openrouter_model(state: dict) -> None:
    choices = [Choice(lab, v) for lab, v in _OPENROUTER_MODELS]
    val = ui.select("Модель OpenRouter", choices, default=state.get("openrouter_model", ""))
    if val is None:
        return
    state["openrouter_model"] = ui.ask_text("Модель OpenRouter (id)", "") if val == "__manual__" else val


def _enter_channels(state: dict) -> None:
    print("Текущие Telegram-каналы:", ", ".join(state["tg_channels"]) or "(пусто)")
    print("Вводи ссылки по одной (Enter — готово, 'clear' — очистить).")
    channels = list(state["tg_channels"])
    while True:
        ans = ui.ask_text("Канал", "")
        if not ans:
            break
        if ans.lower() == "clear":
            channels = []
            continue
        channels.append(ans)
    state["tg_channels"] = channels


def _choose_sources(state: dict) -> None:
    choices = [
        Choice("Локальный файл (data/vacancies.json)", "file"),
        Choice(f"Telegram-каналы ({len(state['tg_channels'])} шт.)", "telegram"),
        Choice(f"PDF/текст ({state['pdf_vacancies'] or 'путь не задан'})", "pdf"),
    ]
    state["sources"] = ui.multiselect("Источники вакансий", choices, default=state["sources"], min_select=1)
    if "telegram" in state["sources"]:
        _enter_channels(state)
    if "pdf" in state["sources"]:
        state["pdf_vacancies"] = ui.ask_text("Путь к PDF/txt с вакансиями", state["pdf_vacancies"])


def _source_label(state: dict) -> str:
    names = {"file": "файл", "telegram": f"telegram({len(state['tg_channels'])})", "pdf": "pdf"}
    return " + ".join(names.get(s, s) for s in state["sources"]) or "файл"


# --------------------------------------------------------------------------- #
# История прогонов / результат
# --------------------------------------------------------------------------- #
def _print_summary(run_dir: Path) -> None:
    report = run_dir / "report.md"
    if not report.exists():
        print("\nОтчёта нет. Хвост run.log:")
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
            score = lines[i + 1].replace("*", "").strip() if i + 1 < len(lines) and "Score" in lines[i + 1] else ""
            print(f"  • {ln[4:].strip()}  ({score})")


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


def _run_actions(run_dir: Path) -> None:
    while True:
        val = ui.select(
            f"Результат: {run_dir.name}",
            [Choice("Краткая сводка (без LLM)", "summary"),
             Choice("Открыть отчёт в приложении", "open"),
             Choice("← Назад", "back")],
        )
        if val in (None, "back"):
            return
        if val == "summary":
            _print_summary(run_dir)
        else:
            _open_in_os(run_dir / "report.md")
        ui.ask_text("Enter — назад", "")


def _history_menu() -> None:
    runs = Path("runs")
    dirs = sorted((d for d in runs.iterdir() if d.is_dir()), reverse=True) if runs.exists() else []
    if not dirs:
        print("Прогонов пока нет (папка runs/ пуста).")
        ui.ask_text("Enter — назад", "")
        return
    choices = [Choice(d.name, str(d)) for d in dirs[:20]] + [Choice("← Назад", "back")]
    val = ui.select("История прогонов", choices)
    if val and val != "back":
        _run_actions(Path(val))


def _run_tests() -> None:
    print("Запуск тестов (pytest)...\n")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    except Exception as e:
        print(f"Не удалось запустить pytest: {e}")
    ui.ask_text("Enter — назад", "")


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
    if state.get("openrouter_model"):
        argv += ["--model-openrouter", state["openrouter_model"]]
    if state["dry_run"]:
        argv.append("--dry-run")
    return argv


def run_menu() -> int:
    main_mod._load_dotenv()  # чтобы видеть, у каких провайдеров есть ключ
    state = load_state()
    cur = "run"
    while True:
        prov = state["provider"]
        if prov == "openrouter" and state.get("openrouter_model"):
            prov += f" · {state['openrouter_model']}"
        choices = [
            Choice(f"Резюме:            {state['resume']}", "resume"),
            Choice(f"Источники:         {_source_label(state)}", "sources"),
            Choice(f"Провайдер LLM:     {prov}", "provider"),
            Choice(f"Топ-N вакансий:    {state['top_n']}", "topn"),
            Choice(f"Режим разбора:     {'без LLM (dry-run)' if state['dry_run'] else 'с LLM'}", "llm"),
            Choice("▶ Запустить", "run"),
            Choice("История прогонов", "history"),
            Choice("Прогнать тесты (pytest)", "tests"),
            Choice("Выход", "exit"),
        ]
        cur = ui.select("Главное меню", choices, default=cur) or "exit"
        if cur == "exit":
            save_state(state)
            print("Пока!")
            return 0
        if cur == "resume":
            _choose_resume(state)
        elif cur == "sources":
            _choose_sources(state)
        elif cur == "provider":
            _choose_provider(state)
        elif cur == "topn":
            v = ui.ask_text("Топ-N (число)", str(state["top_n"]))
            if v.isdigit() and int(v) > 0:
                state["top_n"] = int(v)
        elif cur == "llm":
            state["dry_run"] = not state["dry_run"]  # тумблер на месте
        elif cur == "run":
            save_state(state)
            main_mod.main(_build_argv(state))
            ui.ask_text("Enter — продолжить", "")  # дать прочитать лог прогона
            rd = _latest_run()
            if rd:
                _run_actions(rd)
        elif cur == "history":
            _history_menu()
        elif cur == "tests":
            _run_tests()
        save_state(state)


if __name__ == "__main__":
    try:
        sys.exit(run_menu())
    except KeyboardInterrupt:
        print("\nПока!")
        sys.exit(0)
