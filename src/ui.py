"""Тонкая обёртка над questionary — единый стиль меню для всех проектов.

Всё интерактивное идёт через questionary (стрелки, галочки, ввод). Без своей рамки
и raw-режима. Если терминал не интерактивен (пайп/тесты/CI) или questionary не
установлен — деградируем до выбора по номеру через input().
"""
from __future__ import annotations

import os
import sys

try:
    import questionary
    from questionary import Choice, Style
    _HAS_Q = True
    STYLE = Style([
        ("qmark", "fg:#00afff bold"),
        ("question", "bold"),
        ("pointer", "fg:#00afff bold"),
        ("highlighted", "fg:#00afff bold"),
        ("selected", "fg:#5fd700"),
    ])
except Exception:  # questionary не установлен — работаем только в fallback-режиме
    _HAS_Q = False
    STYLE = None

    class Choice:  # минимальная замена для fallback
        def __init__(self, title: str, value=None, checked: bool = False):
            self.title = title
            self.value = value if value is not None else title
            self.checked = checked

POINTER = "►"


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def clear() -> None:
    """Очистить экран, чтобы отвеченные questionary-промпты не копились в прокрутке."""
    if not _interactive():
        return
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass


def _norm(choices: list) -> list:
    """str -> Choice(str, str); Choice оставляем как есть."""
    return [c if isinstance(c, Choice) else Choice(str(c), c) for c in choices]


def select(title: str, choices: list, default=None):
    """Одиночный выбор стрелками. Возвращает value выбранного или None."""
    items = _norm(choices)
    if _interactive() and _HAS_Q:
        clear()  # чистое окно: убираем «хвост» предыдущих промптов
        ans = questionary.select(
            title, choices=items, default=default, pointer=POINTER, style=STYLE,
        ).ask()
        return ans  # value выбранного Choice или None (Ctrl+C/отмена)
    # Fallback без TTY/questionary.
    print(title)
    for i, c in enumerate(items, 1):
        print(f"  {i}) {c.title}")
    try:
        raw = input("Выбор (номер, пусто=назад): ").strip()
    except EOFError:
        return None
    return items[int(raw) - 1].value if raw.isdigit() and 1 <= int(raw) <= len(items) else None


def multiselect(title: str, choices: list, default=None, min_select: int = 1) -> list:
    """Мультивыбор галочками. Гарантирует выбрано >= min_select. Возвращает список value."""
    default = list(default or [])
    items = _norm(choices)
    if _interactive() and _HAS_Q:
        clear()
        while True:
            q_choices = [Choice(c.title, c.value, checked=c.value in default) for c in items]
            ans = questionary.checkbox(title, choices=q_choices, style=STYLE).ask()
            if ans is None:                      # отмена -> оставляем как было
                return default or [items[0].value]
            if len(ans) >= min_select:
                return ans
            print(f"Нужно выбрать минимум {min_select}. Попробуй ещё раз.")
            default = ans
    # Fallback.
    print(title + f" (номера через запятую, минимум {min_select})")
    for i, c in enumerate(items, 1):
        mark = "x" if c.value in default else " "
        print(f"  {i}) [{mark}] {c.title}")
    try:
        raw = input("Выбор: ").strip()
    except EOFError:
        return default or [items[0].value]
    picked = [items[int(x) - 1].value for x in raw.replace(",", " ").split()
              if x.isdigit() and 1 <= int(x) <= len(items)]
    return picked if len(picked) >= min_select else (default or [items[0].value])


def ask_text(title: str, default: str = "") -> str:
    """Ввод строки. Пустой ввод -> default."""
    if _interactive() and _HAS_Q:
        ans = questionary.text(title, default=default, style=STYLE).ask()
        return (ans or default).strip()
    try:
        ans = input(f"{title} [{default}]: ").strip()
    except EOFError:
        return default
    return ans or default


def confirm(title: str, default: bool = False) -> bool:
    """Да/нет."""
    if _interactive() and _HAS_Q:
        return bool(questionary.confirm(title, default=default, style=STYLE).ask())
    try:
        ans = input(f"{title} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans.startswith("y") or ans.startswith("д")
