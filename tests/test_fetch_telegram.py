"""Тесты: парсинг постов из HTML превью Telegram + извлечение вакансий без LLM."""
from src.fetch_telegram import _PostExtractor, normalize_channel
from src.llm import extract_vacancies_from_text

_HTML = """
<html><body>
<div class="tgme_widget_message_text js-message_text">ML Engineer Intern в DataForge. Python, PyTorch. Удалёнка.</div>
<div class="tgme_widget_message_text">Реклама курса по питону, не вакансия</div>
<div class="tgme_widget_message_text">Junior NLP в NeuroStack<br>Стек: transformers, LLM</div>
</body></html>
"""


def test_normalize_channel():
    assert normalize_channel("https://t.me/tagir_analyzes") == "tagir_analyzes"
    assert normalize_channel("@foo") == "foo"
    assert normalize_channel("t.me/s/bar/") == "bar"


def test_post_extractor():
    p = _PostExtractor()
    p.feed(_HTML)
    assert len(p.posts) == 3
    assert "ML Engineer Intern" in p.posts[0]
    assert "transformers" in p.posts[2]


def test_extract_vacancies_rule_based():
    """Без LLM (dryrun) — грубое извлечение: каждый длинный пост -> запись."""
    posts = [
        "ML Engineer Intern в DataForge. Python, PyTorch. Удалёнка, ментор, гибкий график.",
        "коротко",  # отбросится (слишком короткий)
    ]
    recs = extract_vacancies_from_text(posts, "tg", {}, provider="dryrun", api_key=None)
    assert len(recs) == 1
    assert recs[0]["id"] == "tg_0"
    assert recs[0]["title"]
