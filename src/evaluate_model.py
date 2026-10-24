
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data import load_data, split_features_target


OUTPUT_DIR = Path("models")
COVER_TYPES = [1, 2, 3, 4, 5, 6, 7]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            labels=COVER_TYPES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=COVER_TYPES,
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=COVER_TYPES,
        yticklabels=COVER_TYPES,
    )
    plt.xlabel("Predicted cover type")
    plt.ylabel("Actual cover type")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "random_forest_confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {confusion_path}")

    recalls = recall_score(
        y_test,
        predictions,
        labels=COVER_TYPES,
        average=None,
        zero_division=0,
    )

    plt.figure(figsize=(9, 5))
    bars = plt.bar(COVER_TYPES, recalls)
    plt.xticks(COVER_TYPES)
    plt.ylim(0, 1)
    plt.xlabel("Forest cover type")
    plt.ylabel("Recall")
    plt.title("Random Forest Recall by Cover Type")
    plt.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()

    recall_path = OUTPUT_DIR / "random_forest_class_recall.png"
    plt.savefig(recall_path, dpi=150)
    plt.close()
    print(f"Class recall chart saved to {recall_path}")


if __name__ == "__main__":
    main()
