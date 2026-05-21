"""Tests for the InterestProfile abstraction — no network/API needed."""

import pytest

from feed_ingestion_core.interest_profile import InterestProfile


def test_requires_keywords_or_sentences():
    with pytest.raises(ValueError):
        InterestProfile(keywords=[], interest_sentences=[])


def test_accepts_keywords_only():
    p = InterestProfile(keywords=["sleep"], interest_sentences=[])
    assert p.keywords == ["sleep"]


def test_classify_domains_orders_by_match_count():
    p = InterestProfile(
        keywords=["x"],
        interest_sentences=["y"],
        domains={
            "sleep": ["circadian", "sleep quality"],
            "nutrition": ["protein"],
        },
    )
    text = "A study on circadian rhythm and sleep quality in athletes."
    assert p.classify_domains(text) == ["sleep"]


def test_classify_domains_multiple_hits():
    p = InterestProfile(
        keywords=["x"],
        interest_sentences=["y"],
        domains={"sleep": ["circadian"], "nutrition": ["protein"]},
    )
    result = p.classify_domains("protein timing and circadian rhythm")
    assert set(result) == {"sleep", "nutrition"}


def test_classify_domains_no_match():
    p = InterestProfile(
        keywords=["x"], interest_sentences=["y"], domains={"sleep": ["circadian"]}
    )
    assert p.classify_domains("") == []
    assert p.classify_domains("nothing relevant here") == []


def test_to_from_dict_roundtrip():
    p = InterestProfile(
        keywords=["a", "b"], interest_sentences=["s"], domains={"d": ["t"]}
    )
    p2 = InterestProfile.from_dict(p.to_dict())
    assert p2.keywords == p.keywords
    assert p2.interest_sentences == p.interest_sentences
    assert p2.domains == p.domains
