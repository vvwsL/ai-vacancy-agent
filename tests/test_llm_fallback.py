"""Тест надёжности LLM: при сбоях агент деградирует, а не падает."""
from datetime import date

from src import llm
from src.config import Criteria
from src.llm import (
    CandidateProfile,
    VacancyAnalysis,
    analyze,
    extract_profile,
    prepare_description,
)
from src.loader import Vacancy
from src.scoring import Scored

CRIT = Criteria(role=["ml engineer"], skills=["python", "pytorch"], level=["junior"])
CONFIG = {"llm": {"max_agent_steps": 2}, "max_desc_chars": 100}


def _scored():
    v = Vacancy(
        id="v1", title="ML Intern", company="C", level="junior", role="ml engineer",
        stack=["Python", "PyTorch"], work_format="remote", city="remote",
        published="2026-06-15", description="Python, PyTorch", url="u",
        published_date=date(2026, 6, 15),
    )
    return Scored(vacancy=v, score=72.0, components={"skill_overlap": 1.0}, matched_skills=["python", "pytorch"])


def test_dryrun_returns_rule_based():
    res = analyze(_scored(), CRIT, CONFIG, provider="dryrun", api_key=None)
    assert res["mode"] == "rule_based"
    assert isinstance(res["analysis"], VacancyAnalysis)
    assert res["analysis"].priority == "high"  # score 72 -> high


def test_no_key_forces_rule_based():
    res = analyze(_scored(), CRIT, CONFIG, provider="groq", api_key=None)
    assert res["mode"] == "rule_based"


def test_llm_failure_degrades_to_rule_based(monkeypatch):
    """Если все сетевые вызовы падают — режим деградирует до rule_based, без исключения."""
    def boom(*a, **k):
        raise llm.LLMError("смоделированный сбой сети")

    monkeypatch.setattr(llm, "_llm_call", boom)
    res = analyze(_scored(), CRIT, CONFIG, provider="groq", api_key="fake-key")
    assert res["mode"] == "rule_based"
    assert any("failed" in d for d in res["trace"].get("degraded", []))


def test_garbage_response_degrades(monkeypatch):
    """Модель возвращает мусор без tool_calls и без JSON -> деградация до rule_based."""
    def garbage(messages, tools, config, provider, api_key):
        return {"choices": [{"message": {"content": "это не json и не tool call"}}]}

    monkeypatch.setattr(llm, "_llm_call", garbage)
    res = analyze(_scored(), CRIT, CONFIG, provider="groq", api_key="fake-key")
    assert res["mode"] == "rule_based"


def test_extract_profile_rule_based_dryrun():
    """В dry-run профиль извлекается по словарю навыков, без LLM."""
    resume = "Junior ML инженер. Навыки: Python, PyTorch, NLP, LLM. Ищу remote."
    prof = extract_profile(resume, CONFIG, provider="dryrun", api_key=None)
    assert isinstance(prof, CandidateProfile)
    assert "python" in prof.skills
    assert "pytorch" in prof.skills
    assert "junior" in prof.level
    assert prof.work_format == ["remote"]


def test_prepare_description_cascade():
    short = "коротко"
    assert prepare_description(short, 100) == (short, "as_is")

    long_with_marker = "О компании: вода вода вода. " * 10 + "Требования: Python, PyTorch."
    text, method = prepare_description(long_with_marker, 60)
    assert method in ("structure_trim", "hard_cut")
    assert len(text) <= 60 or method == "structure_trim"
