"""Интерактивное меню со стрелочной навигацией (TUI).

Запуск:  python -m src.menu   (или двойной клик по start.bat на Windows)
Навигация: ↑/↓ — выбор, Enter — ок, Esc — назад. Текстовые поля вводятся обычно.
Настройки сохраняются в user_settings.json.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from . import main as main_mod
from . import tui
from .config import PROVIDER_KEY_ENV, load_config

RESUME_EXTS = {".pdf", ".docx", ".txt", ".md"}
SETTINGS_FILE = "user_settings.json"
TITLE = "AI-агент подбора AI/ML вакансий"


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
        "openrouter_model": "",
    }


def load_state() -> dict:
    state = _default_state()
    p = Path(SETTINGS_FILE)
    if p.exists():
        try:
            state.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
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
# Подменю
# --------------------------------------------------------------------------- #
def _list_resumes() -> list[Path]:
    data = Path("data")
    if not data.exists():
        return []
    return sorted(p for p in data.iterdir() if p.suffix.lower() in RESUME_EXTS)


def _preview_resume(path: str) -> list[str]:
    try:
        from .resume import extract_text
        text = extract_text(path)
    except Exception as e:
        return [f"(предпросмотр недоступен: {e})"]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:8]
    return ["Предпросмотр:"] + ["  | " + ln[:60] for ln in lines]


def _choose_resume(state: dict) -> None:
    files = _list_resumes()
    options = [f.name for f in files] + ["[ввести путь вручную]"]
    header = [TITLE, "Меню › Резюме", f"текущее: {state['resume']}"] + _preview_resume(state["resume"])
    idx = tui.select(header, options)
    if idx is None:
        return
    if idx == len(files):
        tui.clear()
        path = input("Путь к резюме: ").strip()
        if path:
            state["resume"] = path
    else:
        state["resume"] = str(files[idx])


def _available_providers() -> list[str]:
    """auto + провайдеры, у которых есть ключ в окружении, + dryrun."""
    with_keys = [p for p, env in PROVIDER_KEY_ENV.items() if os.environ.get(env)]
    return ["auto"] + with_keys + ["dryrun"]


# Курируемые бесплатные модели OpenRouter (label, value). value="" = дефолт провайдера.
_OPENROUTER_MODELS = [
    ("по умолчанию (llama-3.3-70b-instruct)", ""),
    ("deepseek-chat-v3-0324:free", "deepseek/deepseek-chat-v3-0324:free"),
    ("gemini-2.0-flash-exp:free", "google/gemini-2.0-flash-exp:free"),
    ("qwen-2.5-72b-instruct:free", "qwen/qwen-2.5-72b-instruct:free"),
    ("llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    ("[ввести вручную]", "__manual__"),
]


def _choose_openrouter_model(state: dict) -> None:
    labels = [f"{'● ' if v == state.get('openrouter_model', '') else '  '}{lab}" for lab, v in _OPENROUTER_MODELS]
    idx = tui.select([TITLE, "Меню › Модель OpenRouter"], labels)
    if idx is None:
        return
    value = _OPENROUTER_MODELS[idx][1]
    if value == "__manual__":
        tui.clear()
        state["openrouter_model"] = input("Модель OpenRouter (id, напр. mistralai/mistral-7b-instruct:free): ").strip()
    else:
        state["openrouter_model"] = value


def _choose_provider(state: dict) -> None:
    providers = _available_providers()
    hint = "только провайдеры с ключом в .env" if len(providers) > 2 else "ключей нет — доступен только офлайн"
    start = providers.index(state["provider"]) if state["provider"] in providers else 0
    idx = tui.select([TITLE, "Меню › Провайдер LLM", hint], providers, start=start)
    if idx is not None:
        state["provider"] = providers[idx]
        if state["provider"] == "openrouter":
            _choose_openrouter_model(state)


def _enter_channels(state: dict) -> None:
    tui.clear()
    print("Текущие Telegram-каналы:")
    for c in state["tg_channels"]:
        print(f"  • {c}")
    if not state["tg_channels"]:
        print("  (пусто)")
    print("\nВводи ссылку на канал (напр. https://t.me/tagir_analyzes) и Enter.")
    print("'clear' — очистить список, пустой Enter — готово.")
    channels = list(state["tg_channels"])
    while True:
        try:
            ans = input("Канал: ").strip()
        except EOFError:
            break
        if not ans:
            break
        if ans.lower() == "clear":
            channels = []
            print("  список очищен")
            continue
        channels.append(ans)
        print(f"  добавлен: {ans}")
    state["tg_channels"] = channels


def _choose_sources(state: dict) -> None:
    """Мультивыбор: Enter переключает [x], Esc — назад."""
    while True:
        s = state["sources"]
        options = [
            f"[{'x' if 'file' in s else ' '}] Локальный файл (data/vacancies.json)",
            f"[{'x' if 'telegram' in s else ' '}] Telegram-каналы ({len(state['tg_channels'])} шт.)",
            f"[{'x' if 'pdf' in s else ' '}] PDF/текст ({state['pdf_vacancies'] or 'путь не задан'})",
            "Изменить список Telegram-каналов",
            f"Указать путь к PDF/txt ({state['pdf_vacancies'] or '—'})",
        ]
        idx = tui.select([TITLE, "Меню › Источники (Enter — вкл/выкл, Esc — назад)"], options)
        if idx is None:
            if not state["sources"]:
                state["sources"] = ["file"]
            return
        if idx in (0, 1, 2):
            key = ["file", "telegram", "pdf"][idx]
            if key in s:
                s.remove(key)
            else:
                s.append(key)
        elif idx == 3:
            _enter_channels(state)
        elif idx == 4:
            tui.clear()
            path = input(f"Путь к PDF/txt [{state['pdf_vacancies']}]: ").strip()
            if path:
                state["pdf_vacancies"] = path


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


def _run_tests() -> None:
    tui.clear()
    print("Запуск тестов (pytest)...\n")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    except Exception as e:
        print(f"Не удалось запустить pytest: {e}")
    input("\nEnter — назад...")


def _latest_run() -> Path | None:
    runs = Path("runs")
    dirs = sorted((d for d in runs.iterdir() if d.is_dir()), reverse=True) if runs.exists() else []
    return dirs[0] if dirs else None


def _run_actions(run_dir: Path, crumb: str) -> None:
    while True:
        idx = tui.select(
            [TITLE, crumb, f"папка: {run_dir}"],
            ["Краткая сводка (без LLM)", "Открыть отчёт в приложении"],
        )
        if idx is None:
            return
        if idx == 0:
            tui.clear()
            _print_summary(run_dir)
        else:
            tui.clear()
            _open_in_os(run_dir / "report.md")
        input("\nEnter — назад...")


def _history_menu() -> None:
    runs = Path("runs")
    dirs = sorted((d for d in runs.iterdir() if d.is_dir()), reverse=True) if runs.exists() else []
    if not dirs:
        tui.clear()
        print("Прогонов пока нет (папка runs/ пуста).")
        input("\nEnter — назад...")
        return
    shown = dirs[:20]
    idx = tui.select([TITLE, "Меню › История прогонов"], [d.name for d in shown])
    if idx is not None:
        _run_actions(shown[idx], "Меню › История › прогон")


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
    tui.utf8_stdout()
    main_mod._load_dotenv()  # чтобы видеть, у каких провайдеров есть ключ
    state = load_state()
    cursor = 0  # запоминаем позицию, чтобы не сбрасывалась после действия
    while True:
        prov_label = state["provider"]
        if state["provider"] == "openrouter" and state.get("openrouter_model"):
            prov_label += f" · {state['openrouter_model']}"
        labels = [
            f"Резюме:            {state['resume']}",
            f"Источники:         {_source_label(state)}",
            f"Провайдер LLM:     {prov_label}",
            f"Топ-N вакансий:    {state['top_n']}",
            f"Режим разбора:     {'без LLM (dry-run)' if state['dry_run'] else 'с LLM'}",
            "▶ Запустить",
            "История прогонов",
            "Прогнать тесты (pytest)",
            "Выход",
        ]
        try:
            idx = tui.select([TITLE, "Главное меню"], labels, start=cursor)
        except KeyboardInterrupt:
            print("\nПока!")
            return 0
        if idx is None or idx == 8:
            save_state(state)
            print("Пока!")
            return 0
        cursor = idx  # остаёмся на выбранном пункте
        try:
            if idx == 0:
                _choose_resume(state)
            elif idx == 1:
                _choose_sources(state)
            elif idx == 2:
                _choose_provider(state)
            elif idx == 3:
                tui.clear()
                v = input("Топ-N (число): ").strip()
                if v.isdigit() and int(v) > 0:
                    state["top_n"] = int(v)
            elif idx == 4:
                state["dry_run"] = not state["dry_run"]
            elif idx == 5:
                save_state(state)
                tui.clear()
                main_mod.main(_build_argv(state))
                rd = _latest_run()
                input("\nEnter — посмотреть результат..." if rd else "\nEnter — в меню...")
                if rd:
                    _run_actions(rd, "Результат прогона")
            elif idx == 6:
                _history_menu()
            elif idx == 7:
                _run_tests()
            save_state(state)
        except KeyboardInterrupt:
            print("\n(действие отменено)")
            input("Enter — продолжить...")


if __name__ == "__main__":
    try:
        sys.exit(run_menu())
    except KeyboardInterrupt:
        print("\nПока!")
        sys.exit(0)
