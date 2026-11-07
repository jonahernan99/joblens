import json

import pytest

from joblens.skills import Skill, SkillExtractor, default_extractor, normalize


@pytest.fixture(scope="module")
def extractor() -> SkillExtractor:
    return default_extractor()


def test_bundled_taxonomy_loads(extractor):
    assert len(extractor) >= 90
    assert "Python" in extractor
    assert extractor.category("PyTorch") == "ml_frameworks"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Built services in Python and Go (golang).", {"Python", "Go"}),
        ("Strong C++ and C# background", {"C++", "C#"}),
        ("Set up CI/CD with GitHub Actions", {"CI/CD", "GitHub Actions"}),
        ("APIs in Node.js and FastAPI", {"Node.js", "FastAPI"}),
        ("Retrieval-Augmented Generation over a vector DB", {"RAG", "Vector Databases"}),
        ("Tracked runs in Weights & Biases", {"Weights & Biases"}),
    ],
)
def test_extracts_skills_with_symbols(extractor, text, expected):
    assert extractor.extract(text) == expected


def test_matching_is_case_insensitive(extractor):
    assert extractor.extract("PYTORCH, pytorch, PyTorch") == {"PyTorch"}


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("Maintained our GitHub organization", "Git"),
        ("Fluent in JavaScript", "Java"),
        ("Wrote valid HTML", "Machine Learning"),
        ("Shipped a Node.js service", "JavaScript"),
    ],
)
def test_does_not_match_substrings(extractor, text, forbidden):
    assert forbidden not in extractor.extract(text)


def test_alias_spanning_line_break_is_matched(extractor):
    assert "Machine Learning" in extractor.extract("applied machine\nlearning research")


def test_normalize_handles_typography():
    assert normalize("What You’ll Need") == "what you'll need"


def test_custom_taxonomy():
    custom = SkillExtractor.from_mapping({"tools": {"Excel": ["excel", "spreadsheets"]}})
    assert custom.extract("Advanced spreadsheets") == {"Excel"}


def test_rejects_skill_without_aliases():
    with pytest.raises(ValueError, match="at least one alias"):
        SkillExtractor([Skill(name="Empty", category="misc", aliases=())])


def test_rejects_duplicate_names():
    skill = Skill(name="Python", category="languages", aliases=("python",))
    with pytest.raises(ValueError, match="Duplicate"):
        SkillExtractor([skill, skill])


TOY_TAXONOMY = {
    "misc": {
        "FAISS": ["faiss"],
        "Vector Databases": ["vector database"],
        "PyTorch": ["pytorch"],
        "Deep Learning": ["deep learning"],
        "Machine Learning": ["machine learning"],
    }
}
TOY_IMPLICATIONS = {
    "FAISS": ["Vector Databases"],
    "PyTorch": ["Deep Learning"],
    "Deep Learning": ["Machine Learning"],
}


def test_infer_follows_implications_transitively():
    toy = SkillExtractor.from_mapping(TOY_TAXONOMY, TOY_IMPLICATIONS)
    assert toy.infer({"PyTorch", "FAISS"}) == {
        "Deep Learning": {"PyTorch"},
        "Machine Learning": {"PyTorch"},
        "Vector Databases": {"FAISS"},
    }


def test_infer_skips_skills_already_explicit():
    toy = SkillExtractor.from_mapping(TOY_TAXONOMY, TOY_IMPLICATIONS)
    assert toy.infer({"PyTorch", "Deep Learning"}) == {
        "Machine Learning": {"PyTorch", "Deep Learning"}
    }


def test_infer_tolerates_cycles():
    cyclic = {"FAISS": ["Vector Databases"], "Vector Databases": ["FAISS"]}
    toy = SkillExtractor.from_mapping(TOY_TAXONOMY, cyclic)
    assert toy.infer({"Vector Databases"}) == {"FAISS": {"Vector Databases"}}


def test_infer_without_implications_returns_nothing():
    toy = SkillExtractor.from_mapping(TOY_TAXONOMY)
    assert toy.infer({"PyTorch"}) == {}


def test_rejects_implication_to_unknown_skill():
    with pytest.raises(ValueError, match="unknown skill"):
        SkillExtractor.from_mapping(TOY_TAXONOMY, {"FAISS": ["Quantum Computing"]})


def test_rejects_self_implication():
    with pytest.raises(ValueError, match="cannot imply itself"):
        SkillExtractor.from_mapping(TOY_TAXONOMY, {"FAISS": ["FAISS"]})


def test_bundled_implications_load_and_apply(extractor):
    inferred = extractor.infer(extractor.extract("Built search with FAISS and PyTorch"))
    assert inferred["Vector Databases"] == {"FAISS"}
    assert inferred["Machine Learning"] == {"PyTorch"}


def test_from_json_with_custom_files(tmp_path):
    taxonomy = tmp_path / "skills.json"
    implications = tmp_path / "implications.json"
    taxonomy.write_text(json.dumps(TOY_TAXONOMY), encoding="utf-8")
    implications.write_text(json.dumps(TOY_IMPLICATIONS), encoding="utf-8")

    without = SkillExtractor.from_json(taxonomy)
    with_graph = SkillExtractor.from_json(taxonomy, implications)

    assert without.infer({"FAISS"}) == {}
    assert with_graph.infer({"FAISS"}) == {"Vector Databases": {"FAISS"}}
