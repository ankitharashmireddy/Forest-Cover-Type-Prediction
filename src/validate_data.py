from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/covtype.csv")

COLUMNS = [
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
COLUMNS += [f"Wilderness_Area{i}" for i in range(1, 5)]
COLUMNS += [f"Soil_Type{i}" for i in range(1, 41)]
COLUMNS += ["Cover_Type"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset with corrected column names...")

    # Replace the malformed header; do not discard the first data row.
    df = pd.read_csv(DATA_PATH, header=0, names=COLUMNS)

    print("\nDATASET SUMMARY")
    print("----------------")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\nFIRST THREE FEATURES")
    print("--------------------")
    print(df[["Elevation", "Aspect", "Slope"]].head())

    print("\nCOVER TYPE DISTRIBUTION")
    print("-----------------------")
    print(df["Cover_Type"].value_counts().sort_index().to_string())

    print("\nVALIDATION CHECKS")
    print("-----------------")

    assert len(df.columns) == 55, "Expected 55 columns."
    assert len(df) == 581012, "Unexpected row count."
    assert df.isna().sum().sum() == 0, "Missing values found."
    assert set(df["Cover_Type"].unique()) == set(range(1, 8)), (
        "Expected cover types 1 through 7."
    )

    wilderness_columns = [
        f"Wilderness_Area{i}" for i in range(1, 5)
    ]
    soil_columns = [f"Soil_Type{i}" for i in range(1, 41)]

    for column in wilderness_columns + soil_columns:
        assert set(df[column].unique()).issubset({0, 1}), (
            f"{column} contains values other than 0 and 1."
        )

    assert (df[wilderness_columns].sum(axis=1) == 1).all(), (
        "Some rows do not have exactly one wilderness area."
    )
    assert (df[soil_columns].sum(axis=1) == 1).all(), (
        "Some rows do not have exactly one soil type."
    )

    print("PASS: Expected dimensions and row count.")
    print("PASS: No missing values.")
    print("PASS: All seven cover types present.")
    print("PASS: Wilderness and soil indicators are valid.")
    print("\nDataset validation completed successfully.")
    print("The original CSV was not modified.")


if __name__ == "__main__":
    main()
