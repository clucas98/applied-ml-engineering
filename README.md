# Applied ML Engineering

A hands-on portfolio for **Applied Machine Learning Engineer** roles in France.

## Project 01 — Sensor anomaly detection

Learn an end-to-end supervised ML workflow using **synthetic**, non-proprietary sensor measurements. The initial model predicts whether a measurement window is anomalous. It is a learning baseline, not a production safety system.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
python -m pytest -q
```

Artifacts are written to `artifacts/` (ignored by Git).

### What you will learn

1. Generate a reproducible labeled dataset.
2. Split train/test data without preprocessing leakage.
3. Fit a scikit-learn Pipeline and evaluate precision, recall, F1 and ROC-AUC.
4. Inspect false positives/false negatives.
5. Later: compare models, cross-validation, PyTorch, API serving and MLOps.

### Repository layout

```
src/data.py        # reproducible synthetic dataset
src/train.py       # model, training and evaluation
tests/             # starter unit tests
docs/roadmap.md    # milestones
```

**Limitations:** data is simulated, not real-world labeled sensor observations; random sample splitting is appropriate only for this toy independent-sample dataset. For real time-series data, use grouping and temporal holdouts.
