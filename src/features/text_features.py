"""
Text feature extraction and TF-IDF vectorization for SmartHire.
"""
import joblib
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_VECTORIZER_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)


def get_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    stop_words: str = "english",
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Instantiate a tuned TfidfVectorizer instance."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        sublinear_tf=sublinear_tf,
    )


def fit_and_save_vectorizer(
    corpus: Union[List[str], pd.Series],
    save_path: Optional[Path] = TFIDF_VECTORIZER_PATH,
    **kwargs,
) -> TfidfVectorizer:
    """Fit a TfidfVectorizer on text corpus and serialize to disk."""
    vectorizer = get_tfidf_vectorizer(**kwargs)
    vectorizer.fit(corpus)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, save_path)
    return vectorizer


def load_vectorizer(filepath: Optional[Union[str, Path]] = None) -> TfidfVectorizer:
    """Load serialized TfidfVectorizer."""
    path = Path(filepath) if filepath else TFIDF_VECTORIZER_PATH
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at: {path}")
    return joblib.load(path)
