"""Tests for interview-show guest detection (Channel B) — no network/API needed."""

from feed_ingestion_core.guest_detector import GuestDetector, ReferenceVoice

VOICES = [
    ReferenceVoice(name="Peter Attia", tier="T3", domains=["longevity"], aliases=["Dr. Attia"]),
    ReferenceVoice(name="Rhonda Patrick", tier="T3", domains=["nutrition"]),
    ReferenceVoice(name="Bryan Johnson", tier="T6", domains=["edge"]),
]


def test_detect_guest_in_title():
    d = GuestDetector(VOICES)
    m = d.best_match("JRE #2000 - Peter Attia on longevity")
    assert m is not None
    assert m.voice.name == "Peter Attia"
    assert m.field == "title"
    assert m.tier == "T3"


def test_detect_guest_in_description():
    d = GuestDetector(VOICES)
    matches = d.detect("Episode 50", description="This week we talk to Rhonda Patrick.")
    assert len(matches) == 1
    assert matches[0].voice.name == "Rhonda Patrick"
    assert matches[0].field == "description"


def test_no_guest():
    d = GuestDetector(VOICES)
    assert d.best_match("Random episode about nothing") is None
    assert d.has_credible_guest("nothing here") is False


def test_alias_match():
    d = GuestDetector(VOICES)
    m = d.best_match("A wide-ranging chat with Dr. Attia")
    assert m is not None
    assert m.voice.name == "Peter Attia"


def test_best_match_prefers_better_tier():
    d = GuestDetector(VOICES)
    # Both named; Peter Attia (T3) must outrank Bryan Johnson (T6).
    m = d.best_match("Bryan Johnson and Peter Attia debate aging")
    assert m.voice.name == "Peter Attia"


def test_exclude_host():
    d = GuestDetector(VOICES)
    matches = d.detect(
        "Bryan Johnson interviews Rhonda Patrick", exclude=["Bryan Johnson"]
    )
    names = [m.voice.name for m in matches]
    assert "Bryan Johnson" not in names
    assert "Rhonda Patrick" in names


def test_word_boundary_no_false_positive():
    d = GuestDetector([ReferenceVoice(name="Sam Harris", tier="T5")])
    # "Sam Harrison" must not match "Sam Harris".
    assert d.best_match("Interview with Sam Harrison") is None
