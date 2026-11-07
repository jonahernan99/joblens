"""Command-line interface for JobLens."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from joblens import __version__
from joblens.loaders import UnsupportedFileError, load_text
from joblens.matcher import match
from joblens.report import RENDERERS
from joblens.skills import default_extractor

EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="joblens",
        description="Explainable resume-to-job matching with skill gap analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    match_cmd = subcommands.add_parser("match", help="Score a resume against a job description")
    match_cmd.add_argument("resume", type=Path, help="Resume file (.txt, .md or .pdf)")
    match_cmd.add_argument("job", type=Path, help="Job description file (.txt, .md or .pdf)")
    match_cmd.add_argument(
        "-f", "--format", choices=sorted(RENDERERS), default="text", help="Output format"
    )
    match_cmd.add_argument(
        "--min-score",
        type=float,
        metavar="N",
        help="Exit with status 1 when the score is below N (useful in scripts)",
    )
    match_cmd.add_argument(
        "--no-inference",
        action="store_true",
        help="Only credit skills named explicitly (FAISS will not count as Vector Databases)",
    )

    skills_cmd = subcommands.add_parser("skills", help="List the skills detected in a document")
    skills_cmd.add_argument("file", type=Path, help="Any .txt, .md or .pdf document")
    skills_cmd.add_argument("-f", "--format", choices=["text", "json"], default="text")

    return parser


def _run_match(args: argparse.Namespace) -> int:
    result = match(load_text(args.resume), load_text(args.job), infer_skills=not args.no_inference)
    print(RENDERERS[args.format](result))
    if args.min_score is not None and result.score < args.min_score:
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK


def _run_skills(args: argparse.Namespace) -> int:
    extractor = default_extractor()
    found = sorted(extractor.extract(load_text(args.file)), key=str.casefold)
    if args.format == "json":
        print(json.dumps(found, indent=2))
        return EXIT_OK

    by_category: dict[str, list[str]] = {}
    for name in found:
        by_category.setdefault(extractor.category(name), []).append(name)
    for category in sorted(by_category):
        label = category.replace("_", " ").title()
        print(f"{label}: {', '.join(by_category[category])}")
    if not found:
        print("No known skills detected.")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"match": _run_match, "skills": _run_skills}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, UnsupportedFileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
