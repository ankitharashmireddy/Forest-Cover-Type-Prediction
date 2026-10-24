
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

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

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[1, 2, 3, 4, 5, 6, 7],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[1, 2, 3, 4, 5, 6, 7],
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[1, 2, 3, 4, 5, 6, 7],
        yticklabels=[1, 2, 3, 4, 5, 6, 7],
    )
    plt.xlabel("Predicted cover type")
    plt.ylabel("Actual cover type")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/random_forest_confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to models/random_forest_confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
