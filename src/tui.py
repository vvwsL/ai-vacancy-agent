"""Минимальный терминальный TUI: навигация стрелками, без внешних зависимостей.

Работает на Windows (msvcrt) и Unix (termios). Если ввод не интерактивный
(пайп/тесты) — деградирует до выбора по номеру через input().
"""
from __future__ import annotations

import os
import sys


def _is_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def clear() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass


def utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def get_key() -> str:
    """Прочитать одну клавишу. Возвращает 'up'/'down'/'enter'/'esc' или символ."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):           # спец-клавиши (стрелки)
            ch2 = msvcrt.getwch()
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(ch2, "other")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x1b":
            return "esc"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    # Unix
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc")
        if ch in ("\r", "\n"):
            return "enter"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _draw(header: list[str], options: list[str], idx: int) -> None:
    clear()
    bar = "=" * 50
    print(bar)
    for h in header:
        print("  " + h)
    print(bar)
    for i, opt in enumerate(options):
        pointer = "►" if i == idx else " "
        print(f" {pointer} {opt}")
    print("\n  ↑/↓ — навигация · Enter — выбрать · Esc — назад")


def select(header: list[str], options: list[str], start: int = 0) -> int | None:
    """Выбор пункта стрелками. Возвращает индекс или None (Esc/назад).

    Без tty — печатает нумерованный список и читает номер через input().
    """
    if not options:
        return None
    if not _is_tty():
        print("\n".join(header))
        for i, opt in enumerate(options, 1):
            print(f"  {i}) {opt}")
        try:
            raw = input("Выбор (номер, пусто — назад): ").strip()
        except EOFError:
            return None
        return int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(options) else None

    idx = max(0, min(start, len(options) - 1))
    while True:
        _draw(header, options, idx)
        key = get_key()
        if key == "up":
            idx = (idx - 1) % len(options)
        elif key == "down":
            idx = (idx + 1) % len(options)
        elif key == "enter":
            return idx
        elif key == "esc":
            return None
