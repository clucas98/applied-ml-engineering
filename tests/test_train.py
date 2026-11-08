from src.train import train


def test_train_creates_model_and_metrics(tmp_path):
    metrics = train(output_dir=str(tmp_path))
    assert (tmp_path / "sensor_anomaly_model.joblib").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert 0 <= metrics["f1"] <= 1
