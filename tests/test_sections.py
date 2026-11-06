from joblens.sections import split_requirements

JOB = """\
## Machine Learning Engineer

### Requirements
- 3+ years of Python
- Experience with PyTorch and Docker
- Familiarity with Kubernetes is an asset

### Nice to Have
- Experience with LangChain
- Python packaging (already required above)

### Responsibilities
- Deploy models on AWS
"""


def test_splits_required_and_preferred_by_heading():
    reqs = split_requirements(JOB)
    assert reqs.required == {"Python", "PyTorch", "Docker", "AWS", "Machine Learning"}
    assert reqs.preferred == {"Kubernetes", "LangChain"}


def test_inline_asset_phrase_marks_line_as_preferred():
    reqs = split_requirements("Requirements:\n- SQL\n- Knowledge of Tableau is an asset")
    assert reqs.required == {"SQL"}
    assert reqs.preferred == {"Tableau"}


def test_required_wins_over_preferred():
    reqs = split_requirements("Must have:\n- Python\nBonus:\n- Python, Rust")
    assert reqs.required == {"Python"}
    assert reqs.preferred == {"Rust"}


def test_text_without_headings_is_all_required():
    reqs = split_requirements("We use Python, Kafka and Snowflake every day.")
    assert reqs.required == {"Python", "Kafka", "Snowflake"}
    assert reqs.preferred == frozenset()


def test_new_heading_resets_preferred_section():
    text = "Preferred qualifications:\n- Experience\nRequirements:\n- Built Spark jobs"
    reqs = split_requirements(text)
    assert reqs.required == {"Apache Spark"}
