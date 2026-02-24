"""
main.py

Baseline end-to-end pipeline:
1) Build a small synthetic resume dataset
2) Train a baseline classifier
3) Evaluate performance
4) Run group bias diagnostics
5) Save a simple report to reports/
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import build_vectorizer
from train import build_model
from evaluate import evaluate_model
from bias_analysis import group_metrics


def build_synthetic_dataset() -> pd.DataFrame:
    """
    Build a tiny synthetic dataset for quick end-to-end testing.

    Columns:
    - resume_text: str
    - label: 1 means "shortlist", 0 means "reject"
    - group: a proxy group attribute for bias diagnostics (e.g., "GroupA"/"GroupB")
    """
    data = [
        ("experienced python developer, machine learning, teamwork", 1, "GroupA"),
        ("java backend, microservices, cloud, leadership", 1, "GroupA"),
        ("customer service, retail, cashier, friendly", 0, "GroupA"),
        ("internship python, data analysis, github projects", 1, "GroupB"),
        ("entry level, fast learner, motivated, volunteer", 0, "GroupB"),
        ("project management, stakeholder, communication", 0, "GroupB"),
        ("computer science student, algorithms, leetcode, python", 1, "GroupA"),
        ("no experience, looking for opportunity", 0, "GroupB"),
        ("data science, statistics, pandas, scikit-learn", 1, "GroupB"),
        ("manual labor, warehouse, forklift", 0, "GroupA"),
    ]
    return pd.DataFrame(data, columns=["resume_text", "label", "group"])


def ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def main():
    ensure_reports_dir()

    df = build_synthetic_dataset()

    X_train_text, X_test_text, y_train, y_test, g_train, g_test = train_test_split(
        df["resume_text"],
        df["label"],
        df["group"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = build_model()
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)

    # Prepare bias diagnostics dataframe
    y_pred = model.predict(X_test)
    diag_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "group": g_test.values
    })

    bias_df = group_metrics(diag_df, "y_true", "y_pred", "group")

    # Write report
    report_path = os.path.join("reports", "baseline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Baseline Model Report ===\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results["report"])
        f.write("\n\n=== Group Bias Diagnostics ===\n")
        f.write(bias_df.to_string(index=False))
        f.write("\n")

    print("Pipeline finished.")
    print(f"Report saved to: {report_path}")
    print("\nGroup diagnostics:")
    print(bias_df)


if __name__ == "__main__":
    main()
