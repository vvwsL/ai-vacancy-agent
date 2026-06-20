"""Тест кэша профиля: LLM не вызывается повторно для того же резюме."""
from src import main as main_mod
from src.llm import CandidateProfile
from src.report import RunLogger


def test_profile_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(main_mod, "PROFILE_CACHE", str(tmp_path / "cache.json"))
    calls = {"n": 0}

    def fake_extract(text, config, provider, api_key):
        calls["n"] += 1
        return CandidateProfile(role=["ml engineer"], skills=["python"], level=["junior"])

    monkeypatch.setattr(main_mod, "extract_profile", fake_extract)
    log = RunLogger()

    # 1-й вызов — извлечение через LLM + запись в кэш.
    p1 = main_mod._get_profile("резюме текст", {}, "gemini", "key", log, refresh=False)
    # 2-й вызов того же текста — берётся из кэша, LLM не дёргается.
    p2 = main_mod._get_profile("резюме текст", {}, "gemini", "key", log, refresh=False)
    assert calls["n"] == 1
    assert p2.skills == p1.skills

    # refresh=True — принудительно заново.
    main_mod._get_profile("резюме текст", {}, "gemini", "key", log, refresh=True)
    assert calls["n"] == 2

    # Другое резюме — другой хэш — снова извлечение.
    main_mod._get_profile("другое резюме", {}, "gemini", "key", log, refresh=False)
    assert calls["n"] == 3


def test_dryrun_not_cached(tmp_path, monkeypatch):
    """В dryrun профиль не кэшируется (он бесплатный rule-based)."""
    monkeypatch.setattr(main_mod, "PROFILE_CACHE", str(tmp_path / "c.json"))
    calls = {"n": 0}

    def fake_extract(text, config, provider, api_key):
        calls["n"] += 1
        return CandidateProfile(skills=["python"])

    monkeypatch.setattr(main_mod, "extract_profile", fake_extract)
    log = RunLogger()
    main_mod._get_profile("txt", {}, "dryrun", None, log, refresh=False)
    main_mod._get_profile("txt", {}, "dryrun", None, log, refresh=False)
    assert calls["n"] == 2  # без кэша — оба раза заново
