
from src.data import load_data, split_features_target


def test_dataset_shape():
    df = load_data()
    assert df.shape == (581012, 55)


def test_no_missing_values():
    df = load_data()
    assert not df.isna().any().any()


def test_feature_target_split():
    df = load_data()
    X, y = split_features_target(df)

    assert X.shape == (581012, 54)
    assert y.shape == (581012,)
    assert set(y.unique()) == {1, 2, 3, 4, 5, 6, 7}
