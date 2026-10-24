
import time

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=2,
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

        results.append((name, accuracy, balanced, elapsed))

        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced:.4f}")
        print(f"Training + predict time: {elapsed:.1f} seconds")

    print("\nMODEL COMPARISON")
    print("-" * 75)
    print(
        f"{'Model':<20} {'Accuracy':>12} "
        f"{'Balanced Acc.':>15} {'Time (s)':>12}"
    )

    for name, accuracy, balanced, elapsed in results:
        print(
            f"{name:<20} {accuracy:>12.4f} "
            f"{balanced:>15.4f} {elapsed:>12.1f}"
        )


if __name__ == "__main__":
    main()
