"""Rank many job postings against one resume and summarize the skill gaps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from joblens.loaders import DOCUMENT_SUFFIXES, UnsupportedFileError, load_text
from joblens.matcher import MatchResult, MatchWeights, match
from joblens.similarity import SimilarityBackend
from joblens.skills import SkillExtractor


@dataclass(frozen=True)
class RankedJob:
    """One job posting and its position in a ranking."""

    rank: int
    job: str
    result: MatchResult


@dataclass(frozen=True)
class SkillGap:
    """A required skill the resume lacks, and how many postings ask for it."""

    skill: str
    missing_in: int


@dataclass(frozen=True)
class Ranking:
    """Job postings ordered from best to worst match."""

    entries: tuple[RankedJob, ...]
    skill_gaps: tuple[SkillGap, ...]
    """Missing required skills, most common first, over all ``total_jobs``."""
    total_jobs: int

    def limited(self, top: int | None = None, min_score: float | None = None) -> Ranking:
        """Keep only the best entries; skill gaps still describe every posting."""
        entries = self.entries
        if min_score is not None:
            entries = tuple(entry for entry in entries if entry.result.score >= min_score)
        if top is not None:
            entries = entries[:top]
        return replace(self, entries=entries)


def rank_jobs(
    resume_text: str,
    jobs: Mapping[str, str],
    *,
    extractor: SkillExtractor | None = None,
    weights: MatchWeights | None = None,
    infer_skills: bool = True,
    similarity: SimilarityBackend | None = None,
) -> Ranking:
    """Score ``resume_text`` against every ``{name: job_text}`` entry in ``jobs``.

    Jobs are ordered by score, with ties broken by name so output is stable.
    """
    results = {
        name: match(
            resume_text,
            job_text,
            extractor=extractor,
            weights=weights,
            infer_skills=infer_skills,
            similarity=similarity,
        )
        for name, job_text in jobs.items()
    }
    ordered = sorted(results.items(), key=lambda item: (-item[1].score, item[0].casefold()))

    gaps = Counter(skill for result in results.values() for skill in result.missing_required)
    skill_gaps = sorted(gaps.items(), key=lambda item: (-item[1], item[0].casefold()))

    return Ranking(
        entries=tuple(
            RankedJob(rank=position, job=name, result=result)
            for position, (name, result) in enumerate(ordered, start=1)
        ),
        skill_gaps=tuple(SkillGap(skill, count) for skill, count in skill_gaps),
        total_jobs=len(results),
    )


def discover_job_files(
    paths: Iterable[str | Path], *, exclude: str | Path | None = None
) -> list[Path]:
    """Expand files and directories into a de-duplicated list of job files.

    Directories contribute their supported documents, without recursing.
    ``exclude`` keeps a resume stored alongside the postings out of the results.
    """
    excluded = None if exclude is None else Path(exclude).resolve()
    found: list[Path] = []
    seen: set[Path] = set()
    for path in map(Path, paths):
        if path.is_dir():
            candidates = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in DOCUMENT_SUFFIXES
            )
        elif path.exists():
            candidates = [path]
        else:
            raise FileNotFoundError(f"File not found: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved != excluded and resolved not in seen:
                seen.add(resolved)
                found.append(candidate)
    return found


def load_jobs(paths: Iterable[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """Read job files, returning ``(jobs, skipped)`` keyed by file path.

    One bad file should not stop a whole batch, so unreadable or empty files
    (such as scanned PDFs with no text layer) are reported in ``skipped``.
    """
    jobs: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for path in paths:
        name = path.as_posix()
        try:
            text = load_text(path)
        except (OSError, UnsupportedFileError) as exc:
            skipped[name] = str(exc)
            continue
        if text.strip():
            jobs[name] = text
        else:
            skipped[name] = "no text could be extracted"
    return jobs, skipped
