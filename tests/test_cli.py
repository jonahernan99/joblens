import json
from pathlib import Path

import pytest

from joblens.cli import EXIT_BELOW_THRESHOLD, EXIT_ERROR, EXIT_OK, main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
RESUME = EXAMPLES / "resume_ml_engineer.md"
JOB = EXAMPLES / "job_ml_engineer_toronto.md"


def test_match_text_output(capsys):
    assert main(["match", str(RESUME), str(JOB)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Match score" in out
    assert "Required skills missing" in out


def test_match_json_output(capsys):
    assert main(["match", str(RESUME), str(JOB), "--format", "json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert 0 <= data["score"] <= 100


def test_match_markdown_output(capsys):
    assert main(["match", str(RESUME), str(JOB), "-f", "markdown"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("# JobLens Match Report")


def test_min_score_threshold_sets_exit_code(capsys):
    assert main(["match", str(RESUME), str(JOB), "--min-score", "101"]) == EXIT_BELOW_THRESHOLD


def test_skills_command_groups_by_category(capsys):
    assert main(["skills", str(RESUME)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Languages: " in out
    assert "Python" in out


def test_skills_command_json(capsys):
    assert main(["skills", str(JOB), "-f", "json"]) == EXIT_OK
    assert "Python" in json.loads(capsys.readouterr().out)


def test_missing_file_returns_error(capsys, tmp_path):
    assert main(["skills", str(tmp_path / "nope.txt")]) == EXIT_ERROR
    assert "File not found" in capsys.readouterr().err


def test_unsupported_extension_returns_error(capsys, tmp_path):
    doc = tmp_path / "resume.docx"
    doc.write_bytes(b"binary")
    assert main(["skills", str(doc)]) == EXIT_ERROR
    assert "Unsupported file type" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "joblens" in capsys.readouterr().out
