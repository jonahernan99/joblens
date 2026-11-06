"""Render match results as plain text, Markdown or JSON."""

from __future__ import annotations

import json

from joblens.matcher import MatchResult

_BAR_WIDTH = 30


def _bar(fraction: float) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * _BAR_WIDTH)
    return "[" + "#" * filled + "-" * (_BAR_WIDTH - filled) + "]"


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


def render_text(result: MatchResult) -> str:
    coverage = "n/a" if result.skill_coverage is None else f"{result.skill_coverage:.0%}"
    lines = [
        f"Match score      {result.score:5.1f} / 100  {_bar(result.score / 100)}",
        f"Skill coverage   {coverage}",
        f"Text similarity  {result.text_similarity:.0%}",
        "",
        _verdict(result.score),
        "",
        f"+ Required skills you have   {_join(result.matched_required)}",
        f"- Required skills missing    {_join(result.missing_required)}",
        f"+ Preferred skills you have  {_join(result.matched_preferred)}",
        f"- Preferred skills missing   {_join(result.missing_preferred)}",
        f"* Other skills on resume     {_join(result.additional_skills)}",
    ]
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
        f"**Text similarity:** {result.text_similarity:.0%}",
        "",
        f"> {_verdict(result.score)}",
        "",
        "| Group | Skills |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {_join(skills)} |" for label, skills in rows]
    return "\n".join(lines)


def render_json(result: MatchResult) -> str:
    return json.dumps(result.to_dict(), indent=2)


RENDERERS = {
    "text": render_text,
    "markdown": render_markdown,
    "json": render_json,
}
