"""LLM-агент: провайдер Groq + tool-loop + Pydantic + трёхуровневая деградация.

Здесь живёт АГЕНТНОСТЬ:
- нативный tool-calling loop (ReAct): модель сама решает, какие инструменты звать;
- structured output через Pydantic (валидация + ретрай при битом ответе);
- деградация agent -> single_shot -> rule_based при сбоях.

Инструменты детерминированы (обычный код); LLM их только ВЫЗЫВАЕТ.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from .config import Criteria
from .scoring import Scored

# Все провайдеры ниже OpenAI-совместимы (один формат запроса), различаются URL/моделью/ключом.
# Если у одного перебои с регистрацией/сетью — берём другой тем же кодом.
PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        # Google AI Studio даёт OpenAI-совместимый endpoint — тот же код, что и для остальных.
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "key_env": "OPENROUTER_API_KEY",
    },
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama-3.3-70b",
        "key_env": "CEREBRAS_API_KEY",
    },
}


# --------------------------------------------------------------------------- #
# Структурированный вывод агента
# --------------------------------------------------------------------------- #
class VacancyAnalysis(BaseModel):
    """Строгая схема ответа агента по одной вакансии."""

    extracted_requirements: list[str] = Field(default_factory=list)
    matched: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    questions_to_employer: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    priority_override: bool = False
    override_reason: str = ""
    next_step: str = ""


# --------------------------------------------------------------------------- #
# Профиль кандидата, извлечённый из резюме (главная новая агентность)
# --------------------------------------------------------------------------- #
# Допустимые значения опыта по классификации hh.ru.
HH_EXPERIENCE = ("", "noExperience", "between1And3", "between3And6", "moreThan6")


class CandidateProfile(BaseModel):
    """Структурированный профиль, который LLM вытаскивает из текста резюме.

    Это вход для scoring. experience задаёт уровень искомых вакансий (см. main).
    """

    role: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    level: list[str] = Field(default_factory=list)
    work_format: list[str] = Field(default_factory=list)
    city: str = ""
    experience: str = ""  # одно из HH_EXPERIENCE; задаёт уровень искомых позиций


# Словарь навыков для rule-based извлечения (dry-run / без ключа).
_SKILL_DICT = [
    "python", "pytorch", "tensorflow", "machine learning", "deep learning", "nlp",
    "llm", "rag", "transformers", "sql", "numpy", "pandas", "scikit-learn",
    "computer vision", "opencv", "docker", "git", "spark", "airflow",
]


def _rule_based_profile(text: str) -> CandidateProfile:
    """Грубое извлечение профиля без LLM: ищем навыки по словарю в тексте резюме."""
    low = text.lower()
    skills = [s for s in _SKILL_DICT if s in low]
    is_intern = any(w in low for w in ("intern", "стажёр", "стажер", "trainee"))
    is_junior = "junior" in low or "джуниор" in low or "джун" in low
    level = []
    if is_intern:
        level.append("intern")
    if is_junior:
        level.append("junior")
    role = []
    if any(w in low for w in ("ml", "machine learning", "машинн")):
        role.append("ml engineer")
    if any(w in low for w in ("ai engineer", "llm", "ai-инженер")):
        role.append("ai engineer")
    if "nlp" in low:
        role.append("nlp engineer")
    return CandidateProfile(
        role=role or ["ml engineer"],
        skills=skills or ["python"],
        level=level or ["intern", "junior"],
        work_format=["remote"] if any(w in low for w in ("remote", "удал")) else [],
    )


_PROFILE_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_profile",
        "description": "Отправить извлечённый из резюме профиль кандидата.",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "array", "items": {"type": "string"}},
                "skills": {"type": "array", "items": {"type": "string"}},
                "level": {"type": "array", "items": {"type": "string"}},
                "work_format": {"type": "array", "items": {"type": "string"}},
                "city": {"type": "string"},
                "experience": {"type": "string", "enum": list(HH_EXPERIENCE)},
            },
            "required": ["role", "skills", "level", "experience"],
        },
    },
}]


def extract_profile(resume_text: str, config: dict, provider: str, api_key: str | None) -> CandidateProfile:
    """Извлечь профиль кандидата из текста резюме.

    LLM (если есть провайдер/ключ) -> structured profile; иначе rule-based по словарю.
    """
    if provider == "dryrun" or provider not in PROVIDERS or not api_key:
        return _rule_based_profile(resume_text)

    text = resume_text[: config.get("max_resume_chars", 6000)]
    system = (
        "Ты извлекаешь структурированный профиль из резюме AI/ML кандидата. "
        "Определи роль(и), навыки, уровень (intern/junior/middle/senior), формат работы, город "
        "и опыт по классификации: noExperience (нет опыта), between1And3 (1-3 года), "
        "between3And6 (3-6 лет), moreThan6 (6+). Опыт определяй по годам работы в резюме. "
        "Если чего-то нет — оставь пустым. Вызови submit_profile один раз."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Текст резюме:\n{text}"},
    ]
    try:
        data = _llm_call(messages, _PROFILE_TOOL, config, provider, api_key)
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if calls:
            args = json.loads(calls[0]["function"]["arguments"] or "{}")
            return CandidateProfile(**args)
        # Модель ответила текстом — попробуем достать JSON.
        content = msg.get("content", "")
        s, e = content.find("{"), content.rfind("}")
        if s != -1 and e != -1:
            return CandidateProfile(**json.loads(content[s : e + 1]))
    except (LLMError, ValidationError, KeyError, json.JSONDecodeError):
        pass
    # Деградация: rule-based.
    return _rule_based_profile(resume_text)


# --------------------------------------------------------------------------- #
# Каскад обработки длинного описания (обычный код, 2 уровня)
# --------------------------------------------------------------------------- #
_REQ_MARKERS = (
    "требовани", "обязанност", "будет плюс", "стек", "что нужно",
    "requirements", "responsibilities", "skills", "we expect",
)


def prepare_description(text: str, max_chars: int) -> tuple[str, str]:
    """Ужать описание до max_chars. Возвращает (текст, метод).

    method: as_is | structure_trim | hard_cut
    """
    if len(text) <= max_chars:
        return text, "as_is"

    low = text.lower()
    cut_at = min((low.find(m) for m in _REQ_MARKERS if low.find(m) != -1), default=-1)
    if cut_at > 0:
        trimmed = text[cut_at:].strip()
        if len(trimmed) <= max_chars:
            return trimmed, "structure_trim"
        return trimmed[:max_chars].strip(), "hard_cut"

    return text[:max_chars].strip(), "hard_cut"


# --------------------------------------------------------------------------- #
# Инструменты агента (детерминированные; LLM их вызывает)
# --------------------------------------------------------------------------- #
def _build_tools(scored: Scored, crit: Criteria) -> dict[str, Callable[..., Any]]:
    vac = scored.vacancy

    def get_full_description() -> dict[str, Any]:
        return {"id": vac.id, "description": vac.description}

    def compute_skill_overlap(skills: list[str]) -> dict[str, Any]:
        want = {s.lower() for s in crit.skills}
        have = {s.lower() for s in (skills or [])}
        matched = sorted(want & have)
        return {
            "matched": matched,
            "matched_count": len(matched),
            "candidate_skills_total": len(want),
            "overlap_fraction": round(len(matched) / len(want), 3) if want else 0.0,
        }

    def get_candidate_criteria() -> dict[str, Any]:
        return crit.as_dict()

    return {
        "get_full_description": get_full_description,
        "compute_skill_overlap": compute_skill_overlap,
        "get_candidate_criteria": get_candidate_criteria,
    }


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_full_description",
            "description": "Вернуть полный (необрезанный) текст описания вакансии.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_skill_overlap",
            "description": "Точно посчитать пересечение списка навыков с критериями кандидата.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["skills"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candidate_criteria",
            "description": "Вернуть критерии кандидата (роль, навыки, уровень, формат, город).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_analysis",
            "description": "Отправить итоговый разбор вакансии. Вызывай ОДИН раз, когда данных достаточно.",
            "parameters": {
                "type": "object",
                "properties": {
                    "extracted_requirements": {"type": "array", "items": {"type": "string"}},
                    "matched": {"type": "array", "items": {"type": "string"}},
                    "concerns": {"type": "array", "items": {"type": "string"}},
                    "questions_to_employer": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "priority_override": {"type": "boolean"},
                    "override_reason": {"type": "string"},
                    "next_step": {"type": "string"},
                },
                "required": ["extracted_requirements", "matched", "priority", "next_step"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# HTTP-клиент Groq (urllib, без SDK)
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    pass


def _llm_call(messages: list[dict], tools: list[dict] | None, config: dict, provider: str, api_key: str) -> dict:
    """Один вызов OpenAI-совместимого chat/completions. Бросает LLMError при сети/429/таймауте."""
    llm_cfg = config.get("llm", {})
    spec = PROVIDERS[provider]
    # Модель можно переопределить в config (model_groq и т.п.), иначе дефолт провайдера.
    model = llm_cfg.get(f"model_{provider}", spec["model"])
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": llm_cfg.get("temperature", 0.0),
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    req = urllib.request.Request(
        spec["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = llm_cfg.get("request_timeout", 40)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        raise LLMError(f"HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"Сетевая ошибка: {e}") from e


# --------------------------------------------------------------------------- #
# Режим 1: настоящий агентный tool-loop
# --------------------------------------------------------------------------- #
def _run_agent(scored: Scored, crit: Criteria, config: dict, provider: str, api_key: str, trace: dict) -> VacancyAnalysis:
    vac = scored.vacancy
    tools = _build_tools(scored, crit)
    max_steps = config.get("llm", {}).get("max_agent_steps", 4)

    desc, method = prepare_description(vac.description, config.get("max_desc_chars", 1200))
    trace["desc_handling"] = {"method": method, "orig_len": len(vac.description), "final_len": len(desc)}

    system = (
        "Ты — карьерный агент, который помогает junior AI/ML кандидату решить, "
        "стоит ли откликаться на вакансию. Анализируй по сути, а не пересказывай. "
        "Используй инструменты, если нужны точные данные (полное описание, пересечение навыков, критерии). "
        "Когда данных достаточно — вызови submit_analysis ровно один раз. "
        "priority_override=true ставь, только если твой приоритет осознанно расходится с числовым score."
    )
    user = (
        f"Вакансия id={vac.id}\n"
        f"Должность: {vac.title}\nКомпания: {vac.company}\n"
        f"Уровень: {vac.level} | Роль: {vac.role} | Формат: {vac.work_format} | Город: {vac.city}\n"
        f"Стек: {', '.join(vac.stack) or 'не указан'}\n"
        f"Дата публикации: {vac.published}\n"
        f"Числовой score: {scored.score} из 100 (компоненты: {scored.components})\n"
        f"Описание (возможно сокращено, метод={method}):\n{desc}"
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    trace["tool_calls"] = []

    for step in range(max_steps):
        data = _llm_call(messages, _TOOL_SCHEMAS, config, provider, api_key)
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # Модель ответила текстом без вызова submit — подтолкнём её один раз.
            messages.append({"role": "assistant", "content": msg.get("content", "")})
            messages.append({"role": "user", "content": "Вызови submit_analysis с итогом."})
            continue

        messages.append(msg)  # ассистентское сообщение с tool_calls
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "submit_analysis":
                trace["tool_calls"].append({"step": step, "tool": name})
                try:
                    return VacancyAnalysis(**args)
                except ValidationError as e:
                    # Битый итоговый JSON — попросим исправить (ретрай внутри лимита шагов).
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": f"Ошибка валидации: {e}. Исправь и вызови submit_analysis снова.",
                    })
                    continue

            # Обычный инструмент.
            fn = tools.get(name)
            result = fn(**args) if fn else {"error": f"неизвестный инструмент {name}"}
            trace["tool_calls"].append({"step": step, "tool": name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    raise LLMError(f"Агент не вызвал submit_analysis за {max_steps} шагов")


# --------------------------------------------------------------------------- #
# Режим 2: rule-based (без LLM) — fallback и dry-run
# --------------------------------------------------------------------------- #
def _run_rule_based(scored: Scored, crit: Criteria) -> VacancyAnalysis:
    vac = scored.vacancy
    comp = scored.components
    matched = scored.matched_skills

    concerns: list[str] = []
    if comp.get("level_match", 1) < 0.5:
        concerns.append(f"уровень '{vac.level}' не совпадает с целевым (стажёр/junior)")
    if comp.get("role_match", 1) < 1.0:
        concerns.append(f"роль '{vac.role}' лишь частично совпадает с целевой")
    if "published" in vac.missing_fields or comp.get("freshness", 1) < 0.3:
        concerns.append("публикация старая или дата неизвестна")
    if not matched:
        concerns.append("стек не пересекается с желаемыми навыками")

    # Приоритет по score (детерминированно).
    if scored.score >= 70:
        priority = "high"
    elif scored.score >= 50:
        priority = "medium"
    else:
        priority = "low"

    return VacancyAnalysis(
        extracted_requirements=[f"Стек: {', '.join(vac.stack)}"] if vac.stack else [],
        matched=[f"навык '{m}' совпал с критериями" for m in matched]
        + ([f"формат '{vac.work_format}' подходит"] if comp.get("work_format", 0) >= 1 else []),
        concerns=concerns or ["явных рисков по формальным признакам не видно"],
        questions_to_employer=[
            "Какие задачи у стажёра/джуна в первые 3 месяца?",
            "Есть ли менторство и ревью кода?",
        ],
        priority=priority,
        priority_override=False,
        override_reason="",
        next_step="Откликнуться с сопроводительным" if priority != "low" else "Рассмотреть во вторую очередь",
    )


# --------------------------------------------------------------------------- #
# Точка входа: деградация agent -> rule_based
# --------------------------------------------------------------------------- #
def analyze(scored: Scored, crit: Criteria, config: dict, provider: str, api_key: str | None) -> dict:
    """Разобрать одну вакансию. Возвращает dict с analysis, mode, trace."""
    trace: dict[str, Any] = {"id": scored.vacancy.id}
    t0 = time.time()

    if provider == "dryrun" or provider not in PROVIDERS or not api_key:
        analysis = _run_rule_based(scored, crit)
        trace.update({"mode": "rule_based", "elapsed_s": round(time.time() - t0, 3)})
        return {"analysis": analysis, "mode": "rule_based", "trace": trace}

    # Попытка 1: настоящий агент (tool-loop).
    try:
        analysis = _run_agent(scored, crit, config, provider, api_key, trace)
        trace.update({"mode": "agent", "provider": provider, "elapsed_s": round(time.time() - t0, 3)})
        return {"analysis": analysis, "mode": "agent", "trace": trace}
    except (LLMError, ValidationError, KeyError, json.JSONDecodeError) as e:
        trace.setdefault("degraded", []).append(f"agent failed: {e}")

    # Попытка 2: rule-based (никогда не падает).
    analysis = _run_rule_based(scored, crit)
    trace.update({"mode": "rule_based", "elapsed_s": round(time.time() - t0, 3)})
    return {"analysis": analysis, "mode": "rule_based", "trace": trace}
