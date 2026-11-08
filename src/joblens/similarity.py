"""Text similarity between a resume and a job description.

Two interchangeable backends are available:

- ``tfidf`` (default): fast and dependency-light, but only sees shared words.
- ``embeddings``: sentence-transformer embeddings that capture meaning, so
  "built GPT-based chatbots" relates to "develop LLM-powered assistants" even
  with no words in common. Requires ``pip install 'joblens[embeddings]'``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BACKEND_NAMES = ("embeddings", "tfidf")

# all-MiniLM-L6-v2 truncates input at 256 tokens, roughly 200 English words.
_MAX_CHUNK_WORDS = 200
# Shorter lines are usually headings; they are kept as context for the next line.
_MIN_CHUNK_WORDS = 4
_LINE_PREFIX_RE = re.compile(r"^\s*(?:#+|[-*•▪◦]|\d+[.)])\s+")


class BackendUnavailableError(RuntimeError):
    """Raised when a similarity backend cannot be loaded."""


class SimilarityBackend(Protocol):
    """Scores how similar a resume is to a job description, from 0 to 1."""

    @property
    def name(self) -> str: ...

    def __call__(self, resume_text: str, job_text: str) -> float: ...


class Encoder(Protocol):
    """The subset of ``SentenceTransformer`` that JobLens relies on."""

    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> np.ndarray: ...


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity of TF-IDF vectors (unigrams + bigrams), in ``[0, 1]``.

    Returns ``0.0`` when either text is empty or contains only stop words.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:  # empty vocabulary after stop-word removal
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


class TfidfSimilarity:
    """Lexical similarity over the two whole documents."""

    name = "tfidf"

    def __call__(self, resume_text: str, job_text: str) -> float:
        return tfidf_similarity(resume_text, job_text)


def chunk_text(text: str, max_words: int = _MAX_CHUNK_WORDS) -> list[str]:
    """Split a document into line-level chunks small enough to embed.

    Bullets and Markdown markers are removed, short lines are merged into the
    following line so headings provide context, and lines over ``max_words``
    are split into consecutive windows.
    """
    chunks: list[str] = []
    pending: list[str] = []
    for line in text.splitlines():
        words = _LINE_PREFIX_RE.sub("", line).split()
        if not words:
            continue
        words = pending + words
        if len(words) < _MIN_CHUNK_WORDS:
            pending = words
            continue
        pending = []
        chunks += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
    if pending:
        chunks.append(" ".join(pending))
    return chunks


@lru_cache(maxsize=2)
def _load_encoder(model_name: str) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise BackendUnavailableError(
            "The embeddings backend requires: pip install 'joblens[embeddings]'"
        ) from exc
    try:
        return SentenceTransformer(model_name)
    except OSError as exc:
        raise BackendUnavailableError(
            f"Could not load embedding model {model_name!r}: {exc}"
        ) from exc


class EmbeddingSimilarity:
    """Semantic similarity using sentence-transformer embeddings.

    Each job chunk is compared with its closest resume chunk and the best
    scores are averaged. The result answers "how much of this job does the
    resume cover?", so unrelated extra resume content does not lower it. The
    measure is therefore asymmetric: argument order matters.
    """

    def __init__(
        self, model_name: str = DEFAULT_EMBEDDING_MODEL, encoder: Encoder | None = None
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder

    @property
    def name(self) -> str:
        return f"embeddings:{self.model_name.rsplit('/', 1)[-1]}"

    def __call__(self, resume_text: str, job_text: str) -> float:
        resume_chunks = chunk_text(resume_text)
        job_chunks = chunk_text(job_text)
        if not resume_chunks or not job_chunks:
            return 0.0

        encoder = self._encoder or _load_encoder(self.model_name)
        resume_vectors = np.asarray(encoder.encode(resume_chunks, normalize_embeddings=True))
        job_vectors = np.asarray(encoder.encode(job_chunks, normalize_embeddings=True))

        best_per_job_chunk = (job_vectors @ resume_vectors.T).max(axis=1)
        return float(np.clip(best_per_job_chunk.mean(), 0.0, 1.0))


def get_backend(name: str, *, model: str | None = None) -> SimilarityBackend:
    """Return a similarity backend by name (``model`` applies to embeddings)."""
    if name == "tfidf":
        return TfidfSimilarity()
    if name == "embeddings":
        return EmbeddingSimilarity(model or DEFAULT_EMBEDDING_MODEL)
    raise ValueError(f"Unknown similarity backend {name!r} (choose from {BACKEND_NAMES})")
