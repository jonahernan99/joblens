"""Text similarity between a resume and a job description."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
