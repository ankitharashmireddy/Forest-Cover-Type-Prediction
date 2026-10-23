
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/covtype.csv")

FEATURE_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

FEATURE_COLUMNS += [
    f"Wilderness_Area{i}" for i in range(1, 5)
]
FEATURE_COLUMNS += [
    f"Soil_Type{i}" for i in range(1, 41)
]

TARGET_COLUMN = "Cover_Type"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_data(path=DATA_PATH):
    """Load the dataset and validate its basic structure."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, header=0, names=ALL_COLUMNS)

    if df.shape != (581012, 55):
        raise ValueError(f"Unexpected dataset shape: {df.shape}")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if not df[TARGET_COLUMN].isin(range(1, 8)).all():
        raise ValueError("Cover_Type must contain values from 1 to 7.")

    return df


def split_features_target(df):
    """Separate model inputs (X) from the target (y)."""
    missing_columns = set(ALL_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y
