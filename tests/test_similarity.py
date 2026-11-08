import sys
import zlib

import numpy as np
import pytest

from joblens.similarity import (
    BackendUnavailableError,
    EmbeddingSimilarity,
    TfidfSimilarity,
    _load_encoder,
    chunk_text,
    get_backend,
    tfidf_similarity,
)


class HashingEncoder:
    """Deterministic bag-of-words encoder, so tests need no model download."""

    dimensions = 512

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> np.ndarray:
        assert normalize_embeddings
        self.calls.append(list(sentences))
        vectors = np.zeros((len(sentences), self.dimensions))
        for row, sentence in enumerate(sentences):
            for word in sentence.lower().split():
                vectors[row, zlib.crc32(word.encode()) % self.dimensions] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


class OppositeVectorsEncoder:
    """Returns opposite vectors on consecutive calls (cosine similarity of -1)."""

    def __init__(self) -> None:
        self.sign = 1.0

    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> np.ndarray:
        vectors = np.full((len(sentences), 1), self.sign)
        self.sign = -self.sign
        return vectors


JOB_LINE = "Build data pipelines with Python daily"
OTHER_LINES = "Cooked many tasty dishes for hungry guests\nPlayed competitive chess at local clubs"


def test_chunk_text_removes_markers_and_keeps_headings_as_context():
    text = "## Requirements\n- Build data pipelines with Python\n\n1. Deploy models to production"
    assert chunk_text(text) == [
        "Requirements Build data pipelines with Python",
        "Deploy models to production",
    ]


def test_chunk_text_splits_long_lines_into_windows():
    chunks = chunk_text(" ".join(["word"] * 450), max_words=200)
    assert [len(chunk.split()) for chunk in chunks] == [200, 200, 50]


def test_chunk_text_keeps_trailing_short_text():
    assert chunk_text("machine learning") == ["machine learning"]


@pytest.mark.parametrize("text", ["", "  \n\n  "])
def test_chunk_text_empty(text):
    assert chunk_text(text) == []


def test_embedding_similarity_of_identical_text_is_one():
    backend = EmbeddingSimilarity(encoder=HashingEncoder())
    assert backend(JOB_LINE, JOB_LINE) == pytest.approx(1.0)


@pytest.mark.parametrize(("resume", "job"), [("", JOB_LINE), (JOB_LINE, ""), ("\n", "\n")])
def test_embedding_similarity_skips_encoder_for_empty_text(resume, job):
    encoder = HashingEncoder()
    assert EmbeddingSimilarity(encoder=encoder)(resume, job) == 0.0
    assert encoder.calls == []


def test_unrelated_resume_content_does_not_lower_the_score():
    backend = EmbeddingSimilarity(encoder=HashingEncoder())
    assert backend(f"{JOB_LINE}\n{OTHER_LINES}", JOB_LINE) == pytest.approx(1.0)


def test_uncovered_job_content_lowers_the_score():
    backend = EmbeddingSimilarity(encoder=HashingEncoder())
    assert backend(JOB_LINE, f"{JOB_LINE}\n{OTHER_LINES}") < 0.5


def test_negative_similarity_is_clipped_to_zero():
    backend = EmbeddingSimilarity(encoder=OppositeVectorsEncoder())
    assert backend("some resume text here", "some job text here") == 0.0


def test_missing_dependency_gives_install_hint(monkeypatch):
    _load_encoder.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(BackendUnavailableError, match=r"joblens\[embeddings\]"):
        EmbeddingSimilarity()("some resume text here", "some job text here")


def test_get_backend_by_name():
    assert get_backend("tfidf").name == "tfidf"
    assert get_backend("embeddings").name == "embeddings:all-MiniLM-L6-v2"
    assert get_backend("embeddings", model="org/my-model").name == "embeddings:my-model"


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown similarity backend"):
        get_backend("bm25")


def test_tfidf_backend_delegates_to_function():
    a, b = "machine learning engineer", "senior machine learning scientist"
    assert TfidfSimilarity()(a, b) == tfidf_similarity(a, b)


@pytest.mark.integration
def test_real_model_captures_paraphrases():
    pytest.importorskip("sentence_transformers")
    job = "Build LLM-powered assistants that retrieve answers from internal knowledge bases"
    paraphrase = "Developed GPT-based chatbots answering customer questions from company documents"
    unrelated = "Prepared sourdough bread and managed a busy bakery kitchen every morning"

    backend = get_backend("embeddings")

    assert backend(paraphrase, job) > backend(unrelated, job) + 0.2
    assert tfidf_similarity(paraphrase, job) < 0.05
