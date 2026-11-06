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
