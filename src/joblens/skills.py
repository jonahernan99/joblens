"""Skill taxonomy loading and skill extraction from free text.

The taxonomy is a JSON document mapping ``category -> {canonical_name: [aliases]}``.
Each alias is matched case-insensitively with custom word boundaries so that
symbols commonly found in tech skills (``C++``, ``C#``, ``CI/CD``, ``Node.js``)
are handled correctly, while substrings (``git`` inside ``github``) are not.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

# A match must not be glued to a word character or "." on the left (so "js"
# never matches inside "node.js"), nor to a word character or a trailing
# "+"/"#" on the right (so "c" never matches "c++").
_LEFT_BOUNDARY = r"(?<![a-z0-9_.])"
_RIGHT_BOUNDARY = r"(?![a-z0-9_+#])"

_TYPOGRAPHIC_REPLACEMENTS = str.maketrans(
    {
        "’": "'",  # right single quotation mark
        "‘": "'",  # left single quotation mark
        "–": "-",  # en dash
        "—": "-",  # em dash
        " ": " ",  # non-breaking space
    }
)


@dataclass(frozen=True)
class Skill:
    """A canonical skill with the aliases used to detect it."""

    name: str
    category: str
    aliases: tuple[str, ...]


def normalize(text: str) -> str:
    """Lowercase text, unify typographic characters and collapse whitespace."""
    text = text.translate(_TYPOGRAPHIC_REPLACEMENTS).lower()
    return re.sub(r"\s+", " ", text)


def _compile_aliases(aliases: Iterable[str]) -> re.Pattern[str]:
    # Longest aliases first so the regex engine prefers the most specific form.
    ordered = sorted({normalize(a) for a in aliases}, key=len, reverse=True)
    alternation = "|".join(re.escape(alias) for alias in ordered)
    return re.compile(f"{_LEFT_BOUNDARY}(?:{alternation}){_RIGHT_BOUNDARY}")


class SkillExtractor:
    """Detects canonical skills mentioned in a piece of text."""

    def __init__(self, skills: Iterable[Skill]) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if not skill.aliases:
                raise ValueError(f"Skill {skill.name!r} must define at least one alias")
            if skill.name in self._skills:
                raise ValueError(f"Duplicate skill name in taxonomy: {skill.name!r}")
            self._skills[skill.name] = skill
        self._patterns = [(s.name, _compile_aliases(s.aliases)) for s in self._skills.values()]

    @classmethod
    def from_mapping(cls, taxonomy: dict[str, dict[str, list[str]]]) -> SkillExtractor:
        """Build an extractor from a ``category -> {name: aliases}`` mapping."""
        return cls(
            Skill(name=name, category=category, aliases=tuple(aliases))
            for category, entries in taxonomy.items()
            for name, aliases in entries.items()
        )

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> SkillExtractor:
        """Load a taxonomy from ``path``, or the bundled default taxonomy."""
        if path is None:
            raw = resources.files("joblens").joinpath("data/skills.json").read_text("utf-8")
        else:
            raw = Path(path).read_text(encoding="utf-8")
        return cls.from_mapping(json.loads(raw))

    def extract(self, text: str) -> set[str]:
        """Return the canonical names of all skills mentioned in ``text``."""
        normalized = normalize(text)
        return {name for name, pattern in self._patterns if pattern.search(normalized)}

    def category(self, name: str) -> str:
        """Return the category a canonical skill belongs to."""
        return self._skills[name].category

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)


@lru_cache(maxsize=1)
def default_extractor() -> SkillExtractor:
    """Return a cached extractor built from the bundled taxonomy."""
    return SkillExtractor.from_json()
