"""
preprocess.py

Handles resume text preprocessing and feature transformation.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    """
    Create TF-IDF vectorizer for resume text.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500
    )
    return vectorizer
