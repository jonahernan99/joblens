import csv
import io
import json
import sys
from pathlib import Path

import pytest

from joblens.cli import EXIT_BELOW_THRESHOLD, EXIT_ERROR, EXIT_OK, main
from joblens.similarity import _load_encoder

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


def test_match_explains_inferred_skills(capsys):
    assert main(["match", str(RESUME), str(JOB)]) == EXIT_OK
    assert "Vector Databases (via FAISS)" in capsys.readouterr().out


def test_no_inference_flag(capsys):
    assert main(["match", str(RESUME), str(JOB), "--no-inference", "-f", "json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["inferred_matches"] == {}
    assert "Vector Databases" in data["missing_required"]


def test_text_report_names_similarity_method(capsys):
    assert main(["match", str(RESUME), str(JOB)]) == EXIT_OK
    assert "(tfidf)" in capsys.readouterr().out


def test_embeddings_without_dependency_returns_error(capsys, monkeypatch):
    _load_encoder.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert main(["match", str(RESUME), str(JOB), "--similarity", "embeddings"]) == EXIT_ERROR
    assert "joblens[embeddings]" in capsys.readouterr().err


def test_corrupted_pdf_returns_error(capsys, tmp_path):
    pytest.importorskip("pypdf")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    assert main(["skills", str(broken)]) == EXIT_ERROR
    assert "Could not read PDF" in capsys.readouterr().err


JOBS_DIR = EXAMPLES / "jobs"


def test_rank_text_output_shows_table_and_gaps(capsys):
    assert main(["rank", str(RESUME), str(JOB), str(JOBS_DIR)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Ranked 4 job postings" in out
    assert "Most common missing required skills" in out


def test_rank_json_is_sorted_best_first(capsys):
    assert main(["rank", str(RESUME), str(JOBS_DIR), "-f", "json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    scores = [item["score"] for item in data["results"]]
    assert scores == sorted(scores, reverse=True)
    assert data["total_jobs"] == 3
    assert data["skill_gaps"]


def test_rank_csv_has_header_and_one_row_per_job(capsys):
    assert main(["rank", str(RESUME), str(JOBS_DIR), "-f", "csv"]) == EXIT_OK
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 3
    assert rows[0]["rank"] == "1"


def test_rank_markdown_output(capsys):
    assert main(["rank", str(RESUME), str(JOBS_DIR), "-f", "markdown"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("# JobLens Job Ranking")


def test_rank_top_limits_results_but_not_total(capsys):
    assert main(["rank", str(RESUME), str(JOBS_DIR), "--top", "1", "-f", "json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert len(data["results"]) == 1
    assert data["total_jobs"] == 3


def test_rank_min_score_can_hide_everything(capsys):
    assert main(["rank", str(RESUME), str(JOBS_DIR), "--min-score", "101"]) == EXIT_OK
    assert "No job postings passed the filters." in capsys.readouterr().out


def test_rank_excludes_resume_stored_with_jobs(capsys, tmp_path):
    resume = tmp_path / "resume.md"
    job = tmp_path / "job.md"
    resume.write_text("Python developer", encoding="utf-8")
    job.write_text("Requirements:\n- Python", encoding="utf-8")
    assert main(["rank", str(resume), str(tmp_path), "-f", "json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["total_jobs"] == 1


def test_rank_warns_about_skipped_files(capsys, tmp_path):
    (tmp_path / "blank.md").write_text(" ", encoding="utf-8")
    (tmp_path / "job.md").write_text("Requirements:\n- Python", encoding="utf-8")
    assert main(["rank", str(RESUME), str(tmp_path)]) == EXIT_OK
    assert "warning: skipped" in capsys.readouterr().err


def test_rank_without_readable_jobs_returns_error(capsys, tmp_path):
    assert main(["rank", str(RESUME), str(tmp_path)]) == EXIT_ERROR
    assert "No readable job descriptions" in capsys.readouterr().err


def test_rank_with_blank_resume_returns_error(capsys, tmp_path):
    resume = tmp_path / "resume.md"
    resume.write_text(" ", encoding="utf-8")
    assert main(["rank", str(resume), str(JOBS_DIR)]) == EXIT_ERROR
    assert "No text could be extracted" in capsys.readouterr().err


def test_rank_rejects_non_positive_top():
    with pytest.raises(SystemExit) as exc:
        main(["rank", str(RESUME), str(JOBS_DIR), "--top", "0"])
    assert exc.value.code == 2
