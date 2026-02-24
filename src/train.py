"""
train.py

Handles model training.
"""

from sklearn.linear_model import LogisticRegression


def build_model():
    """
    Create baseline classification model.
    """
    model = LogisticRegression(max_iter=1000)
    return model
