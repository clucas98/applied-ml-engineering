"""Generate an illustrative, reproducible synthetic sensor dataset."""
import numpy as np
import pandas as pd

FEATURES = ["acceleration_rms", "temperature_c", "vibration_rms", "signal_quality"]


def make_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    rng = np.random.default_rng(seed)
    anomaly = rng.binomial(1, 0.2, size=n_samples)
    df = pd.DataFrame({
        "acceleration_rms": rng.normal(0.5 + 0.65 * anomaly, 0.2, n_samples),
        "temperature_c": rng.normal(25 + 6 * anomaly, 3, n_samples),
        "vibration_rms": rng.normal(0.3 + 0.5 * anomaly, 0.16, n_samples),
        "signal_quality": rng.normal(0.85 - 0.22 * anomaly, 0.09, n_samples),
        "anomaly": anomaly,
    })
    return df
