"""Загрузка конфигурации и критериев кандидата.

Обычная логика: чтение YAML и парсинг criteria.md в структуру.
Здесь же — валидация criteria (симметрично валидации вакансий).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Ошибка конфигурации или критериев (понятная, без stacktrace для пользователя)."""


@dataclass
class Criteria:
    """Критерии кандидата, распарсенные из criteria.md."""

    role: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    level: list[str] = field(default_factory=list)
    work_format: list[str] = field(default_factory=list)
    city: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "role": self.role,
            "skills": self.skills,
            "level": self.level,
            "work_format": self.work_format,
            "city": self.city,
        }


def load_config(path: str | Path) -> dict[str, Any]:
    """Прочитать config.yaml. Бросает ConfigError при отсутствии/битом файле."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Не найден config: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Битый YAML в {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Config {p} должен быть YAML-словарём")
    return data


# Разделы criteria.md, которые мы ожидаем (## role / ## skills / ...).
_SECTION_RE = re.compile(r"^##\s+(\w+)\s*$", re.MULTILINE)


def load_criteria(path: str | Path) -> Criteria:
    """Распарсить criteria.md в Criteria.

    Формат: разделы '## key', под каждым строка значений через запятую.
    Валидация: файл существует, не пустой, есть хотя бы role и skills.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Не найден файл критериев: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"Файл критериев пуст: {p}")

    sections: dict[str, list[str]] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        values = [v.strip().lower() for v in re.split(r"[,\n]", body) if v.strip()]
        sections[key] = values

    crit = Criteria(
        role=sections.get("role", []),
        skills=sections.get("skills", []),
        level=sections.get("level", []),
        work_format=sections.get("work_format", []),
        city=sections.get("city", []),
    )
    # Валидация: без роли и навыков фильтрация/scoring бессмысленны.
    if not crit.role:
        raise ConfigError("В criteria.md отсутствует или пуст раздел '## role'")
    if not crit.skills:
        raise ConfigError("В criteria.md отсутствует или пуст раздел '## skills'")
    return crit


# Провайдер -> имя переменной окружения с ключом. Все OpenAI-совместимы.
PROVIDER_KEY_ENV: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}


def resolve_provider(config: dict[str, Any], cli_provider: str | None, dry_run: bool) -> str:
    """Определить итогового провайдера LLM.

    Приоритет: --dry-run флаг > --provider CLI > config.provider.
    Режим 'auto': первый провайдер, для которого есть ключ в окружении; иначе 'dryrun'.
    """
    if dry_run:
        return "dryrun"
    provider = cli_provider or config.get("llm", {}).get("provider", "auto")
    if provider == "auto":
        for name, env in PROVIDER_KEY_ENV.items():
            if os.environ.get(env):
                return name
        return "dryrun"
    return provider


def api_key_for(provider: str) -> str | None:
    """Вернуть ключ из окружения для выбранного провайдера."""
    env = PROVIDER_KEY_ENV.get(provider)
    return os.environ.get(env) if env else None
