import pytest

from joblens.ranker import SkillGap, discover_job_files, load_jobs, rank_jobs

RESUME = "Python SQL Docker"
JOBS = {
    "full_match": "Requirements:\n- Python, SQL, Docker",
    "partial_match": "Requirements:\n- Python, AWS",
    "no_match": "Requirements:\n- Kubernetes, AWS",
}


def test_jobs_are_ordered_best_first():
    ranking = rank_jobs(RESUME, JOBS)
    assert [entry.job for entry in ranking.entries] == [
        "full_match",
        "partial_match",
        "no_match",
    ]
    assert [entry.rank for entry in ranking.entries] == [1, 2, 3]
    assert ranking.total_jobs == 3


def test_ties_are_broken_by_name():
    text = "Requirements:\n- Python"
    ranking = rank_jobs(RESUME, {"b": text, "A": text, "c": text})
    assert [entry.job for entry in ranking.entries] == ["A", "b", "c"]


def test_skill_gaps_are_sorted_by_frequency():
    assert rank_jobs(RESUME, JOBS).skill_gaps == (SkillGap("AWS", 2), SkillGap("Kubernetes", 1))


def test_limited_top_and_min_score_keep_gaps_for_all_jobs():
    ranking = rank_jobs(RESUME, JOBS)

    assert [entry.job for entry in ranking.limited(top=1).entries] == ["full_match"]

    threshold = ranking.entries[1].result.score
    filtered = ranking.limited(min_score=threshold)
    assert len(filtered.entries) == 2
    assert filtered.skill_gaps == ranking.skill_gaps
    assert filtered.total_jobs == 3


def test_empty_jobs_give_empty_ranking():
    ranking = rank_jobs(RESUME, {})
    assert ranking.entries == ()
    assert ranking.skill_gaps == ()
    assert ranking.total_jobs == 0


def test_discover_expands_directories_to_supported_documents(tmp_path):
    for name in ("b.txt", "a.md", "c.PDF", "ignored.docx"):
        (tmp_path / name).write_text("text", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.md").write_text("text", encoding="utf-8")

    assert [path.name for path in discover_job_files([tmp_path])] == ["a.md", "b.txt", "c.PDF"]


def test_discover_removes_duplicates_and_excluded_resume(tmp_path):
    job = tmp_path / "job.md"
    resume = tmp_path / "resume.md"
    job.write_text("text", encoding="utf-8")
    resume.write_text("text", encoding="utf-8")

    assert discover_job_files([job, tmp_path], exclude=resume) == [job]


def test_discover_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        discover_job_files([tmp_path / "missing.md"])


def test_load_jobs_skips_blank_and_unsupported_files(tmp_path):
    good = tmp_path / "good.md"
    blank = tmp_path / "blank.txt"
    unsupported = tmp_path / "job.docx"
    good.write_text("Requirements: Python", encoding="utf-8")
    blank.write_text("  \n", encoding="utf-8")
    unsupported.write_bytes(b"binary")

    jobs, skipped = load_jobs([good, blank, unsupported])

    assert jobs == {good.as_posix(): "Requirements: Python"}
    assert skipped[blank.as_posix()] == "no text could be extracted"
    assert "Unsupported file type" in skipped[unsupported.as_posix()]
