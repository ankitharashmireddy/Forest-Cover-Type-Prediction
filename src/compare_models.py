
import time

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=2,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=2,
        ),
        "Balanced Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=2,
            class_weight="balanced_subsample",
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        elapsed = time.perf_counter() - start
        accuracy = accuracy_score(y_test, predictions)
        balanced = balanced_accuracy_score(y_test, predictions)

        recalls = recall_score(
            y_test,
            predictions,
            labels=[4, 5],
            average=None,
            zero_division=0,
        )
        recall_4, recall_5 = recalls

        results.append(
            (name, accuracy, balanced, recall_4, recall_5, elapsed)
        )

        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced:.4f}")
        print(f"Class 4 recall:    {recall_4:.4f}")
        print(f"Class 5 recall:    {recall_5:.4f}")
        print(f"Training + predict time: {elapsed:.1f} seconds")

    print("\nMODEL COMPARISON")
    print("-" * 100)
    print(
        f"{'Model':<25} {'Accuracy':>10} {'Balanced Acc.':>15} "
        f"{'Class 4 Recall':>15} {'Class 5 Recall':>15} {'Time (s)':>10}"
    )

    for name, accuracy, balanced, recall_4, recall_5, elapsed in results:
        print(
            f"{name:<25} {accuracy:>10.4f} {balanced:>15.4f} "
            f"{recall_4:>15.4f} {recall_5:>15.4f} {elapsed:>10.1f}"
        )


if __name__ == "__main__":
    main()
