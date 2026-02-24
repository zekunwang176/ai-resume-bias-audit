# AI Resume Bias Audit & Decision Framework

## Project Overview

This project explores how AI-based resume screening systems make decisions and where bias may emerge in the process.

Rather than only building a classifier, this project analyzes:

- What features influence hiring decisions
- How proxy variables may introduce unintended bias
- How threshold choices affect fairness and performance
- How to design more transparent and controllable screening logic

The goal is to move beyond model accuracy and focus on decision structure and risk awareness.

---

## Problem Statement

AI resume screening systems are increasingly used to filter job applicants.  
However, these systems may unintentionally learn biased patterns from historical data.

Key concerns include:

- Gender and ethnicity proxy effects
- University ranking bias
- Keyword-driven filtering logic
- Threshold-based decision rigidity

This project investigates these risks and proposes structural improvements.

---

## Project Structure
ai-resume-bias-audit/
│
├── src/        # Core model implementation
├── data/       # Raw and processed resume datasets
├── models/     # Saved trained models
├── analysis/   # Bias evaluation logic
├── reports/    # Final analysis reports
---

## Technical Stack

- Python
- scikit-learn
- NLP preprocessing (TF-IDF)
- Basic classification models (Logistic Regression / Random Forest)
- Fairness-aware evaluation metrics

---

## Development Phases

Phase 1:
- Construct synthetic resume dataset
- Implement preprocessing pipeline
- Train baseline classifier

Phase 2:
- Conduct bias analysis
- Evaluate proxy variable influence
- Analyze false positive / false negative patterns

Phase 3:
- Design threshold control system
- Add explainability layer
- Propose structural fairness improvements

---

## Core Objective

This project is not about maximizing accuracy.

It is about understanding how decision systems operate and how to improve their structural integrity.
