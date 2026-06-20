"""Тесты scoring: hard-фильтр уровня и детерминированность весов."""
from src.config import Criteria
from src.loader import Vacancy
from src.scoring import filter_and_score

CONFIG = {
    "reject_levels": ["senior", "lead"],
    "weights": {
        "skill_overlap": 0.40,
        "role_match": 0.25,
        "level_match": 0.15,
        "freshness": 0.10,
        "work_format": 0.07,
        "city_match": 0.03,
    },
}

CRIT = Criteria(
    role=["ml engineer", "ai engineer"],
    skills=["python", "pytorch", "llm"],
    level=["intern", "junior"],
    work_format=["remote"],
    city=["remote"],
)


def _vac(**kw):
    base = dict(
        id="v", title="T", company="C", level="junior", role="ml engineer",
        stack=["Python", "PyTorch"], work_format="remote", city="remote",
        published="2026-06-15", description="d", url="u",
    )
    base.update(kw)
    from datetime import date
    v = Vacancy(**base)
    v.published_date = date(2026, 6, 15)
    return v


def test_senior_is_hard_rejected():
    scored, rejected = filter_and_score([_vac(level="senior")], CRIT, CONFIG)
    assert scored == []
    assert len(rejected) == 1
    assert "senior" in rejected[0].reason


def test_better_match_scores_higher():
    good = _vac(id="g", stack=["Python", "PyTorch", "LLM"], role="ml engineer")
    weak = _vac(id="w", stack=["Java"], role="backend developer", work_format="office", city="Казань")
    scored, _ = filter_and_score([weak, good], CRIT, CONFIG)
    assert scored[0].vacancy.id == "g"
    assert scored[0].score > scored[1].score


def test_score_is_deterministic():
    v = [_vac(id="a"), _vac(id="b", stack=["Python"])]
    r1, _ = filter_and_score(v, CRIT, CONFIG)
    r2, _ = filter_and_score(v, CRIT, CONFIG)
    assert [s.score for s in r1] == [s.score for s in r2]


def test_score_in_range():
    scored, _ = filter_and_score([_vac()], CRIT, CONFIG)
    assert 0 <= scored[0].score <= 100


def test_criteria_from_profile_lowercased():
    """Критерии из резюме приводятся к lowercase — 'ML Engineer' матчит 'ml engineer'."""
    from src.main import _criteria_from_profile
    from src.llm import CandidateProfile
    prof = CandidateProfile(role=["ML Engineer"], skills=["Python"],
                            level=["Junior"], experience="between1And3")
    crit = _criteria_from_profile(prof)
    assert crit.role == ["ml engineer"]
    assert crit.skills == ["python"]
    # роль вакансии 'ml engineer' теперь совпадает (role_match не нулевой)
    scored, _ = filter_and_score([_vac(role="ml engineer")], crit, CONFIG)
    assert scored[0].components["role_match"] == 1.0
