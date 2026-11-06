"""Score how well a resume matches a job description."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from joblens.sections import split_requirements
from joblens.similarity import tfidf_similarity
from joblens.skills import SkillExtractor, default_extractor

# A missing "nice to have" skill hurts half as much as a missing requirement.
REQUIRED_WEIGHT = 1.0
PREFERRED_WEIGHT = 0.5


@dataclass(frozen=True)
class MatchWeights:
    """Relative weight of each signal in the final score."""

    skills: float = 0.75
    text: float = 0.25

    def __post_init__(self) -> None:
        if self.skills < 0 or self.text < 0 or self.skills + self.text == 0:
            raise ValueError("Weights must be non-negative and not both zero")


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching a resume against a job description."""

    score: float
    """Overall match, from 0 to 100."""
    skill_coverage: float | None
    """Weighted share of the job's skills found in the resume, or ``None`` if
    no known skills were detected in the job description."""
    text_similarity: float
    """TF-IDF cosine similarity between both documents, from 0 to 1."""
    matched_required: tuple[str, ...]
    missing_required: tuple[str, ...]
    matched_preferred: tuple[str, ...]
    missing_preferred: tuple[str, ...]
    additional_skills: tuple[str, ...]
    """Skills on the resume that the job description does not mention."""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, tuple):
                data[key] = list(value)
        return data


def _sorted(skills: set[str] | frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(skills, key=str.casefold))


def _weighted_coverage(
    matched_required: int, total_required: int, matched_preferred: int, total_preferred: int
) -> float | None:
    total = total_required * REQUIRED_WEIGHT + total_preferred * PREFERRED_WEIGHT
    if total == 0:
        return None
    matched = matched_required * REQUIRED_WEIGHT + matched_preferred * PREFERRED_WEIGHT
    return matched / total


def match(
    resume_text: str,
    job_text: str,
    *,
    extractor: SkillExtractor | None = None,
    weights: MatchWeights | None = None,
) -> MatchResult:
    """Compare a resume with a job description and explain the result."""
    extractor = extractor or default_extractor()
    weights = weights or MatchWeights()

    resume_skills = extractor.extract(resume_text)
    requirements = split_requirements(job_text, extractor)

    matched_required = requirements.required & resume_skills
    matched_preferred = requirements.preferred & resume_skills

    coverage = _weighted_coverage(
        len(matched_required),
        len(requirements.required),
        len(matched_preferred),
        len(requirements.preferred),
    )
    similarity = tfidf_similarity(resume_text, job_text)

    if coverage is None:
        score = similarity * 100
    else:
        blended = weights.skills * coverage + weights.text * similarity
        score = 100 * blended / (weights.skills + weights.text)

    return MatchResult(
        score=round(score, 1),
        skill_coverage=None if coverage is None else round(coverage, 4),
        text_similarity=round(similarity, 4),
        matched_required=_sorted(matched_required),
        missing_required=_sorted(requirements.required - resume_skills),
        matched_preferred=_sorted(matched_preferred),
        missing_preferred=_sorted(requirements.preferred - resume_skills),
        additional_skills=_sorted(resume_skills - requirements.all),
    )
