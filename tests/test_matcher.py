import pytest

from joblens.matcher import MatchWeights, match
from joblens.similarity import tfidf_similarity

JOB = """\
Requirements:
- Python, PyTorch, Docker, SQL
Nice to have:
- Kubernetes, LangChain
"""


def test_perfect_skill_match_scores_high():
    resume = "Python PyTorch Docker SQL Kubernetes LangChain"
    result = match(resume, JOB)
    assert result.skill_coverage == 1.0
    assert result.missing_required == ()
    assert result.missing_preferred == ()
    assert result.score >= 75


def test_reports_gaps_and_extra_skills():
    resume = "Python and SQL developer. Also React and Kubernetes."
    result = match(resume, JOB)
    assert result.matched_required == ("Python", "SQL")
    assert result.missing_required == ("Docker", "PyTorch")
    assert result.matched_preferred == ("Kubernetes",)
    assert result.missing_preferred == ("LangChain",)
    assert result.additional_skills == ("React",)


def test_preferred_skills_weigh_half_as_much():
    # 4 required (weight 1) + 2 preferred (weight 0.5) = 5 total weight.
    only_required = match("Python PyTorch Docker SQL", JOB)
    only_preferred = match("Kubernetes LangChain", JOB)
    assert only_required.skill_coverage == pytest.approx(4 / 5)
    assert only_preferred.skill_coverage == pytest.approx(1 / 5)


def test_falls_back_to_text_similarity_without_known_skills():
    result = match("I love baking sourdough bread", "Bakery seeks sourdough bread baker")
    assert result.skill_coverage is None
    assert result.score == pytest.approx(result.text_similarity * 100, abs=0.1)
    assert result.score > 0


def test_custom_weights_skills_only():
    result = match("Python PyTorch Docker SQL", JOB, weights=MatchWeights(skills=1, text=0))
    assert result.score == pytest.approx(80.0)


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        MatchWeights(skills=0, text=0)
    with pytest.raises(ValueError):
        MatchWeights(skills=-1, text=1)


def test_to_dict_is_json_friendly():
    data = match("Python", JOB).to_dict()
    assert isinstance(data["matched_required"], list)
    assert set(data) >= {"score", "skill_coverage", "text_similarity"}


@pytest.mark.parametrize(("a", "b"), [("", "python"), ("the and of", "a an the")])
def test_similarity_degenerate_inputs(a, b):
    assert tfidf_similarity(a, b) == 0.0


def test_similarity_identical_texts():
    assert tfidf_similarity("machine learning engineer", "machine learning engineer") == (
        pytest.approx(1.0)
    )


INFERENCE_JOB = "Requirements:\n- Experience with vector databases and deep learning"


def test_specific_resume_skills_satisfy_broader_requirements():
    result = match("Built semantic search with FAISS and PyTorch", INFERENCE_JOB)
    assert result.matched_required == ("Deep Learning", "Vector Databases")
    assert result.missing_required == ()
    assert result.inferred_matches == {
        "Deep Learning": ("PyTorch",),
        "Vector Databases": ("FAISS",),
    }


def test_inference_can_be_disabled():
    result = match(
        "Built semantic search with FAISS and PyTorch", INFERENCE_JOB, infer_skills=False
    )
    assert result.matched_required == ()
    assert result.inferred_matches == {}


def test_inferred_skills_are_not_listed_as_additional():
    result = match("PyTorch", "Requirements:\n- Python")
    assert result.additional_skills == ("PyTorch",)
    assert result.inferred_matches == {}


def test_to_dict_serializes_inferred_matches():
    data = match("FAISS", INFERENCE_JOB).to_dict()
    assert data["inferred_matches"] == {"Vector Databases": ["FAISS"]}
