"""
data_loader.py

Handles dataset loading and basic dataset validation.
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from given file path.

    Args:
        path (str): Path to dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
