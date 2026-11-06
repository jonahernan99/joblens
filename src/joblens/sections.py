"""Split a job description into required and preferred skills.

Job postings usually separate hard requirements from "nice to have" skills,
either with a heading (``Preferred Qualifications``, ``Bonus Points``) or inline
(``Experience with Kubernetes is an asset`` - a phrasing common in Canada).
This module uses lightweight heuristics to recover that structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from joblens.skills import SkillExtractor, default_extractor

_BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")

_PREFERRED_RE = re.compile(
    r"nice[- ]to[- ]have"
    r"|good[- ]to[- ]have"
    r"|\bpreferred\b"
    r"|\bbonus\b"
    r"|\ba plus\b"
    r"|\ban asset\b"
    r"|^assets?\b"
    r"|\bdesirable\b"
    r"|\boptional\b"
)

_HEADING_HINTS = (
    "requirement",
    "qualification",
    "must have",
    "must-have",
    "what you bring",
    "what you'll need",
    "what we're looking for",
    "who you are",
    "about you",
    "skills",
    "responsibilities",
    "what you'll do",
    "about the role",
)

_MAX_HEADING_LENGTH = 60


@dataclass(frozen=True)
class JobRequirements:
    """Skills a job posting asks for, split by importance."""

    required: frozenset[str]
    preferred: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        return self.required | self.preferred


def _clean(line: str) -> str:
    """Strip bullets and markdown decoration, then lowercase."""
    line = _BULLET_RE.sub("", line)
    return line.strip().strip("#*_:").strip().lower()


def _is_preferred(line: str) -> bool:
    return bool(_PREFERRED_RE.search(_clean(line)))


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _BULLET_RE.match(line):
        return False
    if stripped.startswith("#"):
        return True
    cleaned = _clean(line)
    if len(cleaned) > _MAX_HEADING_LENGTH:
        return False
    return (
        stripped.endswith(":")
        or any(hint in cleaned for hint in _HEADING_HINTS)
        or bool(_PREFERRED_RE.search(cleaned))
    )


def split_requirements(job_text: str, extractor: SkillExtractor | None = None) -> JobRequirements:
    """Extract required and preferred skills from a job description.

    A skill mentioned as both required and preferred is treated as required.
    """
    extractor = extractor or default_extractor()
    required: set[str] = set()
    preferred: set[str] = set()
    in_preferred_section = False

    for line in job_text.splitlines():
        if _is_heading(line):
            in_preferred_section = _is_preferred(line)
        target = preferred if in_preferred_section or _is_preferred(line) else required
        target |= extractor.extract(line)

    return JobRequirements(
        required=frozenset(required),
        preferred=frozenset(preferred - required),
    )
