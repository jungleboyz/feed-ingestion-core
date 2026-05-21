"""Tests for keyword scoring with an injected InterestProfile — no API calls.

Confirms the central refactor: the Scorer takes its keyword list from the InterestProfile
instead of a hardcoded module constant.
"""

from feed_ingestion_core.interest_profile import InterestProfile
from feed_ingestion_core.scoring_service import Scorer


def _scorer(keywords):
    return Scorer(
        InterestProfile(keywords=keywords, interest_sentences=["placeholder"]),
        use_semantic=False,
    )


def test_score_keywords_title_weighted_3x():
    s = _scorer(["hypertrophy", "sleep"])
    assert s.score_keywords("Hypertrophy basics") == 3
    assert s.score_keywords("Basics", description="about sleep") == 1
    assert s.score_keywords("Hypertrophy", description="and sleep") == 4


def test_score_keywords_uses_injected_profile():
    # A profile with different keywords yields different scores — proves injection.
    assert _scorer(["sleep"]).score_keywords("sleep study") == 3
    assert _scorer(["nutrition"]).score_keywords("sleep study") == 0


def test_score_semantic_falls_back_to_keywords_when_disabled():
    s = _scorer(["sleep"])
    # use_semantic=False -> no API call, returns the keyword score.
    assert s.score_semantic("sleep study") == 3


def test_score_items_batch_keyword_mode():
    s = _scorer(["sleep"])
    items = [
        {"title": "sleep study", "summary": ""},
        {"title": "other", "summary": "sleep tips"},
    ]
    out = s.score_items_batch(items)
    assert out[0]["score"] == 3
    assert out[1]["score"] == 1
    assert out[0]["embedding"] is None
