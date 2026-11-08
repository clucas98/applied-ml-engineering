from src.data import FEATURES, make_dataset


def test_dataset_is_reproducible():
    assert make_dataset(seed=7).equals(make_dataset(seed=7))


def test_dataset_schema():
    data = make_dataset(100)
    assert len(data) == 100
    assert set(FEATURES + ["anomaly"]) == set(data.columns)
    assert set(data.anomaly.unique()).issubset({0, 1})
