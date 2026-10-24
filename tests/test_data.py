
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


def test_target_contains_valid_classes():
    df = load_data()

    assert df["Cover_Type"].between(1, 7).all()


def test_wilderness_indicators_are_valid():
    df = load_data()
    columns = [f"Wilderness_Area{i}" for i in range(1, 5)]

    assert df[columns].isin([0, 1]).all().all()
    assert (df[columns].sum(axis=1) == 1).all()


def test_soil_indicators_are_valid():
    df = load_data()
    columns = [f"Soil_Type{i}" for i in range(1, 41)]

    assert df[columns].isin([0, 1]).all().all()
    assert (df[columns].sum(axis=1) == 1).all()
