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
    """Собирает текст из <div class="...tgme_widget_message_text...">."""

    def __init__(self) -> None:
        super().__init__()
        self.posts: list[str] = []
        self._depth = 0          # глубина вложенности внутри целевого div
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = dict(attrs).get("class") or ""
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
                    self.posts.append(text)

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._buf.append(data)


def fetch_posts(channel: str, limit: int = 20, timeout: int = 30) -> list[str]:
    """Вернуть тексты последних постов публичного канала."""
    name = normalize_channel(channel)
    if not name:
        raise TelegramError(f"Не разобрал канал: {channel}")
    url = f"https://t.me/s/{name}"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        raise TelegramError(f"Telegram HTTP {e.code} для @{name}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise TelegramError(f"Сеть/таймаут при чтении @{name}: {e}") from e

    html = _BR_RE.sub("\n", html)
    parser = _PostExtractor()
    parser.feed(html)
    if not parser.posts:
        raise TelegramError(f"У @{name} не найдено постов (приватный канал или пусто?)")
    # Превью отдаёт старые сверху; берём последние limit.
    return parser.posts[-limit:]
