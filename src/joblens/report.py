"""Render match results and job rankings for people and scripts."""

from __future__ import annotations

import csv
import io
import json

from joblens.matcher import MatchResult
from joblens.ranker import Ranking

_BAR_WIDTH = 30
_GAP_BAR_WIDTH = 10
_MAX_GAPS_SHOWN = 10


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _verdict(score: float) -> str:
    if score >= 75:
        return "Strong match - apply with confidence."
    if score >= 55:
        return "Good match - tailor your resume to close the gaps below."
    if score >= 35:
        return "Partial match - consider it a stretch role."
    return "Weak match - significant gaps for this role."


def _join(skills: tuple[str, ...]) -> str:
    return ", ".join(skills) if skills else "-"


def _join_inferred(inferred: dict[str, tuple[str, ...]]) -> str:
    if not inferred:
        return "-"
    return ", ".join(f"{skill} (via {', '.join(sources)})" for skill, sources in inferred.items())


def render_text(result: MatchResult) -> str:
    coverage = "n/a" if result.skill_coverage is None else f"{result.skill_coverage:.0%}"
    lines = [
        f"Match score      {result.score:5.1f} / 100  {_bar(result.score / 100)}",
        f"Skill coverage   {coverage}",
        f"Text similarity  {result.text_similarity:.0%} ({result.similarity_method})",
        "",
        _verdict(result.score),
        "",
        f"+ Required skills you have   {_join(result.matched_required)}",
        f"- Required skills missing    {_join(result.missing_required)}",
        f"+ Preferred skills you have  {_join(result.matched_preferred)}",
        f"- Preferred skills missing   {_join(result.missing_preferred)}",
        f"* Other skills on resume     {_join(result.additional_skills)}",
    ]
    if result.inferred_matches:
        lines.append(f"~ Credited via related      {_join_inferred(result.inferred_matches)}")
    if result.missing_required:
        lines += [
            "",
            "Tip: if you have experience with the missing required skills, name them",
            "explicitly in your resume - many screening systems match on exact keywords.",
        ]
    return "\n".join(lines)


def render_markdown(result: MatchResult) -> str:
    coverage = "n/a" if result.skill_coverage is None else f"{result.skill_coverage:.0%}"
    rows = [
        ("Required - matched", result.matched_required),
        ("Required - missing", result.missing_required),
        ("Preferred - matched", result.matched_preferred),
        ("Preferred - missing", result.missing_preferred),
        ("Other resume skills", result.additional_skills),
    ]
    lines = [
        "# JobLens Match Report",
        "",
        f"**Score:** {result.score:.1f} / 100  ",
        f"**Skill coverage:** {coverage}  ",
        f"**Text similarity:** {result.text_similarity:.0%} ({result.similarity_method})",
        "",
        f"> {_verdict(result.score)}",
        "",
        "| Group | Skills |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {_join(skills)} |" for label, skills in rows]
    if result.inferred_matches:
        lines.append(f"| Credited via related skills | {_join_inferred(result.inferred_matches)} |")
    return "\n".join(lines)


def render_json(result: MatchResult) -> str:
    return json.dumps(result.to_dict(), indent=2)


RENDERERS = {
    "text": render_text,
    "markdown": render_markdown,
    "json": render_json,
}


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _ranking_title(ranking: Ranking) -> str:
    shown, total = len(ranking.entries), ranking.total_jobs
    noun = "posting" if total == 1 else "postings"
    if shown == total:
        return f"Ranked {total} job {noun}"
    return f"Showing {shown} of {total} ranked job {noun}"


def render_ranking_text(ranking: Ranking) -> str:
    lines = [_ranking_title(ranking), ""]
    if ranking.entries:
        method = ranking.entries[0].result.similarity_method
        lines += [
            f"{'#':>3}  {'Score':>5}  {'Skills':>6}  {'Missing':>7}  Job",
            *(
                f"{entry.rank:>3}  {entry.result.score:>5.1f}  "
                f"{_percent(entry.result.skill_coverage):>6}  "
                f"{len(entry.result.missing_required):>7}  {entry.job}"
                for entry in ranking.entries
            ),
            "",
            f"Text similarity backend: {method}",
        ]
    else:
        lines.append("No job postings passed the filters.")

    if ranking.skill_gaps:
        gaps = ranking.skill_gaps[:_MAX_GAPS_SHOWN]
        width = max(len(gap.skill) for gap in gaps)
        lines += ["", f"Most common missing required skills (all {ranking.total_jobs} postings)"]
        lines += [
            f"  {gap.skill:<{width}}  {gap.missing_in:>3}/{ranking.total_jobs}  "
            f"{_bar(gap.missing_in / ranking.total_jobs, _GAP_BAR_WIDTH)}"
            for gap in gaps
        ]
    return "\n".join(lines)


def render_ranking_markdown(ranking: Ranking) -> str:
    lines = ["# JobLens Job Ranking", "", f"{_ranking_title(ranking)}.", ""]
    if ranking.entries:
        lines += [
            "| # | Score | Skill coverage | Missing required | Job |",
            "| ---: | ---: | ---: | --- | --- |",
        ]
        lines += [
            f"| {entry.rank} | {entry.result.score:.1f} "
            f"| {_percent(entry.result.skill_coverage)} "
            f"| {_join(entry.result.missing_required)} "
            f"| {entry.job.replace('|', chr(92) + '|')} |"
            for entry in ranking.entries
        ]
    if ranking.skill_gaps:
        lines += [
            "",
            "## Most common missing required skills",
            "",
            "| Skill | Postings |",
            "| --- | ---: |",
        ]
        lines += [
            f"| {gap.skill} | {gap.missing_in} / {ranking.total_jobs} |"
            for gap in ranking.skill_gaps[:_MAX_GAPS_SHOWN]
        ]
    return "\n".join(lines)


def render_ranking_json(ranking: Ranking) -> str:
    payload = {
        "total_jobs": ranking.total_jobs,
        "results": [
            {"rank": entry.rank, "job": entry.job, **entry.result.to_dict()}
            for entry in ranking.entries
        ],
        "skill_gaps": [
            {"skill": gap.skill, "missing_in": gap.missing_in} for gap in ranking.skill_gaps
        ],
    }
    return json.dumps(payload, indent=2)


def render_ranking_csv(ranking: Ranking) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "rank",
            "job",
            "score",
            "skill_coverage",
            "text_similarity",
            "similarity_method",
            "missing_required",
            "missing_preferred",
        ]
    )
    for entry in ranking.entries:
        result = entry.result
        writer.writerow(
            [
                entry.rank,
                entry.job,
                result.score,
                "" if result.skill_coverage is None else result.skill_coverage,
                result.text_similarity,
                result.similarity_method,
                "; ".join(result.missing_required),
                "; ".join(result.missing_preferred),
            ]
        )
    return buffer.getvalue().rstrip("\n")


RANK_RENDERERS = {
    "text": render_ranking_text,
    "markdown": render_ranking_markdown,
    "json": render_ranking_json,
    "csv": render_ranking_csv,
}
