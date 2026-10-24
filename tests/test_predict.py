
import pytest

from src.data import FEATURE_COLUMNS, load_data, split_features_target
from src.predict import predict_cover_type


def test_prediction_returns_valid_class():
    df = load_data()
    X, _ = split_features_target(df)

    sample = X.iloc[0].to_dict()
    prediction = predict_cover_type(sample)

    assert prediction in {1, 2, 3, 4, 5, 6, 7}


def test_prediction_rejects_missing_features():
    sample = {column: 0 for column in FEATURE_COLUMNS}
    sample.pop(FEATURE_COLUMNS[0])

    with pytest.raises(ValueError, match="Missing features"):
        predict_cover_type(sample)


def test_prediction_rejects_extra_features():
    sample = {column: 0 for column in FEATURE_COLUMNS}
    sample["Unexpected_Feature"] = 0

    with pytest.raises(ValueError, match="Unexpected features"):
        predict_cover_type(sample)
