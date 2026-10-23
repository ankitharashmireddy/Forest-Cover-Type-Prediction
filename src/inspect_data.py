import csv
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/covtype.csv")

expected_columns = [
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
expected_columns += [f"Wilderness_Area{i}" for i in range(1, 5)]
expected_columns += [f"Soil_Type{i}" for i in range(1, 41)]
expected_columns += ["Cover_Type"]

row_count = 0
class_counts = Counter()
bad_lengths = 0
invalid_targets = 0
first_row = None

with DATA_PATH.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)
    original_header = next(reader, None)

    for row in reader:
        row_count += 1

        if first_row is None:
            first_row = row

        if len(row) != len(expected_columns):
            bad_lengths += 1
            continue

        target = row[-1]
        if target not in {"1", "2", "3", "4", "5", "6", "7"}:
            invalid_targets += 1
        else:
            class_counts[target] += 1

print("DATASET INSPECTION")
print("------------------")
print(f"File: {DATA_PATH}")
print(f"Original header fields: {len(original_header or [])}")
print(f"Expected data columns: {len(expected_columns)}")
print(f"Data rows: {row_count}")
print(f"Rows with an unexpected field count: {bad_lengths}")
print(f"Rows with an invalid target: {invalid_targets}")
print(f"Total target counts: {sum(class_counts.values())}")
print("\nCover_Type distribution:")

for cover_type in sorted(class_counts):
    print(f"  Class {cover_type}: {class_counts[cover_type]:,}")

print("\nFirst row mapped to expected column names:")
if first_row and len(first_row) == len(expected_columns):
    for name, value in zip(expected_columns[:14], first_row[:14]):
        print(f"  {name}: {value}")
else:
    print("First row does not have the expected number of fields.")

print("\nNote: Expected column names are used for inspection only.")
print("The original CSV has not been modified.")
