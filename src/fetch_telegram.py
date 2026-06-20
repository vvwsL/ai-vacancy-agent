"""Чтение публичного превью Telegram-канала: https://t.me/s/<канал>.

Это публичная веб-превьюшка (последние посты), отдаётся без авторизации — легально,
не приватный API и не обход защиты. Достаём только тексты постов (обычный код);
структурирование в вакансии делает LLM (см. llm.extract_vacancies_from_text).
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; vacancy-agent/1.0)"}
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


class TelegramError(Exception):
    pass


def normalize_channel(value: str) -> str:
    """'https://t.me/foo', '@foo', 'foo' -> 'foo'."""
    v = value.strip()
    v = v.replace("https://", "").replace("http://", "")
    v = v.replace("t.me/s/", "").replace("t.me/", "")
    v = v.lstrip("@").strip("/")
    return v.split("/")[0].split("?")[0]


class _PostExtractor(HTMLParser):
    """Собирает (текст, ссылку) постов из превью t.me/s/.

    Ссылка берётся из контейнера сообщения (атрибут data-post="канал/123").
    """

    def __init__(self) -> None:
        super().__init__()
        self.posts: list[tuple[str, str]] = []
        self._cur_url = ""       # пермалинк текущего сообщения
        self._depth = 0          # глубина внутри div сообщения-текста
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        cls = d.get("class") or ""
        if tag == "div" and "tgme_widget_message" in cls and d.get("data-post"):
            self._cur_url = f"https://t.me/{d['data-post']}"
        if tag == "div" and "tgme_widget_message_text" in cls:
            self._depth = 1
            self._buf = []
        elif self._depth:
            self._depth += 1
            if tag == "br":
                self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._depth:
            self._depth -= 1
            if self._depth == 0:
                text = "".join(self._buf).strip()
                if text:
                    self.posts.append((text, self._cur_url))

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._buf.append(data)


def fetch_posts(channel: str, limit: int = 20, timeout: int = 30) -> list[tuple[str, str]]:
    """Вернуть последние посты канала как список (текст, ссылка-пермалинк)."""
    name = normalize_channel(channel)
    # Имя канала в t.me — латиница/цифры/подчёркивание. Кириллица/<>/пусто = не канал
    # (например, осталась заглушка вроде <ml_канал>) — пропускаем с понятным сообщением.
    if not name or not name.isascii() or not all(c.isalnum() or c == "_" for c in name):
        raise TelegramError(f"некорректное имя канала '{channel}' — вставь реальную ссылку, напр. https://t.me/ml_jobs")
    url = f"https://t.me/s/{name}"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        raise TelegramError(f"Telegram HTTP {e.code} для @{name}") from e
    except (urllib.error.URLError, TimeoutError, UnicodeError, ValueError) as e:
        raise TelegramError(f"Сеть/таймаут при чтении @{name}: {e}") from e

    html = _BR_RE.sub("\n", html)
    parser = _PostExtractor()
    parser.feed(html)
    if not parser.posts:
        raise TelegramError(f"У @{name} не найдено постов (приватный канал или пусто?)")
    # Превью отдаёт старые сверху; берём последние limit.
    return parser.posts[-limit:]
