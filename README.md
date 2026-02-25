# AI Resume Screening Simulator & Decision Analysis

## Project Overview

This project builds a simplified resume screening simulator to analyze how keyword-based and model-based filtering systems make decisions.

The objective is to:

- Implement a baseline screening pipeline
- Evaluate decision thresholds
- Compare different scoring strategies
- Identify potential error patterns
- Analyze basic group-level bias patterns

---

## System Architecture

The system consists of:

- Resume preprocessing (text cleaning + TF-IDF vectorization)
- Baseline keyword scoring module
- Logistic regression classifier
- Threshold-based ranking mechanism
- Evaluation and reporting module

---

## Baseline Strategy

The initial implementation includes:

- TF-IDF feature extraction
- Logistic regression classifier
- Fixed decision threshold for shortlist selection

Performance metrics:

- Accuracy
- False Positive Rate (FPR)
- False Negative Rate (FNR)

---

## Improvement Experiments

Planned experiments include:

- Threshold sensitivity analysis
- Weighted keyword scoring comparison
- Proxy feature removal experiment
- Error pattern and misclassification analysis
- Basic group-level fairness comparison

---

## Example Output

(Results screenshots will be added here)

---

## Technical Stack

- Python
- scikit-learn
- TF-IDF
- Logistic Regression
- Basic fairness diagnostics
