"""Guest detection for interview-show harvesting (ARCH-008 Channel B).

Big interview shows (Joe Rogan, Tim Ferriss, …) are harvesting vehicles, not tiered
sources. An episode is only worth ingesting when a recognized credible expert is the
guest — and the episode's tier is then inherited from that guest, not the show.

This module does the *metadata-level* match (guest name in episode title/description).
The consumer's pipeline confirms it with the LLM summarization step before stamping a
tier — see ARCH-008 §"Guest misidentification".
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

_TIER_RE = re.compile(r"T([1-6])", re.IGNORECASE)


def _tier_rank(tier: str) -> int:
    """Lower is better. T1 -> 1 … T6 -> 6; unknown -> 99."""
    m = _TIER_RE.search(tier or "")
    return int(m.group(1)) if m else 99


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace for matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


@dataclass
class ReferenceVoice:
    """A credible expert in the guest-matching dictionary.

    ``aliases`` carry alternate forms ("Dr. Attia", "Peter Attia MD"). Keep aliases
    distinctive — bare first names invite false positives.
    """

    name: str
    tier: str
    domains: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    caveats: str = ""

    def match_terms(self) -> List[str]:
        """All name forms to search for."""
        return [self.name, *self.aliases]


@dataclass
class GuestMatch:
    """A detected guest appearance."""

    voice: ReferenceVoice
    matched_term: str
    field: str  # 'title' or 'description'

    @property
    def tier(self) -> str:
        return self.voice.tier


class GuestDetector:
    """Detects credible guests in interview-show episode metadata."""

    def __init__(self, voices: List[ReferenceVoice]):
        self.voices = list(voices)
        # Pre-compile a word-boundary pattern for every name/alias.
        self._patterns: List[tuple] = []
        for voice in self.voices:
            for term in voice.match_terms():
                term = (term or "").strip()
                if not term:
                    continue
                pattern = re.compile(r"\b" + re.escape(_norm(term)) + r"\b")
                self._patterns.append((voice, term, pattern))

    def detect(
        self,
        title: str,
        description: str = "",
        exclude: Optional[List[str]] = None,
    ) -> List[GuestMatch]:
        """Return guest matches in *title*/*description*, best tier first.

        Args:
            title: Episode title (most reliable — guests are usually named here).
            description: Episode description / show notes.
            exclude: Names to ignore — typically the show's own host, so a host who is
                also a reference voice doesn't match every one of their own episodes.

        Returns:
            One GuestMatch per distinct voice, sorted by tier (best first), with
            title matches ranked above description-only matches.
        """
        exclude_norm = {_norm(x) for x in (exclude or [])}
        title_n = _norm(title)
        desc_n = _norm(description)

        matches: List[GuestMatch] = []
        seen: set = set()
        for voice, term, pattern in self._patterns:
            if voice.name in seen or _norm(voice.name) in exclude_norm:
                continue
            if pattern.search(title_n):
                matches.append(GuestMatch(voice, term, "title"))
                seen.add(voice.name)
            elif pattern.search(desc_n):
                matches.append(GuestMatch(voice, term, "description"))
                seen.add(voice.name)

        matches.sort(key=lambda m: (_tier_rank(m.tier), 0 if m.field == "title" else 1))
        return matches

    def best_match(
        self,
        title: str,
        description: str = "",
        exclude: Optional[List[str]] = None,
    ) -> Optional[GuestMatch]:
        """The single best (highest-tier) guest match, or None."""
        matches = self.detect(title, description, exclude=exclude)
        return matches[0] if matches else None

    def has_credible_guest(
        self,
        title: str,
        description: str = "",
        exclude: Optional[List[str]] = None,
    ) -> bool:
        """Whether the episode features at least one recognized credible guest."""
        return bool(self.detect(title, description, exclude=exclude))
