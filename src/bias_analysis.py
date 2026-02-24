"""
bias_analysis.py

Provides basic bias / fairness diagnostics by group.
"""

import pandas as pd


def group_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str) -> pd.DataFrame:
    """
    Compute basic group-level metrics:
    - selection_rate: P(y_pred=1)
    - true_positive_rate: P(y_pred=1 | y_true=1)
    - false_positive_rate: P(y_pred=1 | y_true=0)

    Args:
        df: dataframe containing true labels, predictions, and group column
        y_true_col: name of true label column
        y_pred_col: name of predicted label column
        group_col: name of sensitive/group attribute column

    Returns:
        pd.DataFrame with metrics per group
    """
    rows = []

    for g, sub in df.groupby(group_col):
        y_true = sub[y_true_col]
        y_pred = sub[y_pred_col]

        selection_rate = (y_pred == 1).mean()

        positives = sub[sub[y_true_col] == 1]
        negatives = sub[sub[y_true_col] == 0]

        tpr = (positives[y_pred_col] == 1).mean() if len(positives) > 0 else 0.0
        fpr = (negatives[y_pred_col] == 1).mean() if len(negatives) > 0 else 0.0

        rows.append({
            group_col: g,
            "count": len(sub),
            "selection_rate": selection_rate,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
        })

    return pd.DataFrame(rows).sort_values(by=group_col)
