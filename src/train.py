"""Train a supervised learning baseline without training/test leakage."""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, make_dataset


def train(seed: int = 42, output_dir: str = "artifacts") -> dict:
    df = make_dataset(seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["anomaly"], test_size=0.25, random_state=seed, stratify=df["anomaly"]
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "sensor_anomaly_model.joblib")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
