
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.data import load_data, split_features_target


def main():
    print("Loading dataset...")
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )

    model_path = Path("models/random_forest.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    print("\nTesting saved model...")
    loaded_model = joblib.load(model_path)
    reloaded_predictions = loaded_model.predict(X_test)

    same_predictions = (predictions == reloaded_predictions).all()
    print(f"Predictions identical after reload: {same_predictions}")

    if not same_predictions:
        raise RuntimeError("Saved model predictions differ from original predictions.")

    print("Model save-and-reload test passed.")


if __name__ == "__main__":
    main()
