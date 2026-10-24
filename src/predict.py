
import joblib
import pandas as pd

from src.data import FEATURE_COLUMNS

MODEL_PATH = "models/random_forest.joblib"


def predict_cover_type(features):
    """
    Predict the forest cover type from one record containing 54 features.

    features can be a dictionary keyed by feature name or a pandas Series.
    """
    model = joblib.load(MODEL_PATH)

    if isinstance(features, dict):
        features = pd.DataFrame([features])
    elif isinstance(features, pd.Series):
        features = features.to_frame().T
    elif isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("Provide exactly one record for prediction.")
    else:
        raise TypeError(
            "features must be a dictionary, pandas Series, or DataFrame."
        )

    missing = set(FEATURE_COLUMNS) - set(features.columns)
    extra = set(features.columns) - set(FEATURE_COLUMNS)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if extra:
        raise ValueError(f"Unexpected features: {sorted(extra)}")

    features = features[FEATURE_COLUMNS]

    prediction = int(model.predict(features)[0])
    return prediction


def main():
    # Use a real record from the dataset to demonstrate the prediction workflow.
    # This is a demonstration, not an independent real-world evaluation.
    from src.data import load_data, split_features_target

    df = load_data()
    X, _ = split_features_target(df)

    sample = X.iloc[0].to_dict()
    prediction = predict_cover_type(sample)

    print("Prediction test using the first dataset record")
    print(f"Predicted cover type: {prediction}")


if __name__ == "__main__":
    main()
