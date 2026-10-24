
@'
# Forest Cover Type Prediction

A machine learning project that predicts forest cover types from cartographic and environmental features using the Covertype dataset.

## Project Overview

This project implements a supervised machine learning workflow, from dataset validation and model comparison to model evaluation and reusable predictions.

## Objectives

- Inspect and validate the dataset.
- Train and compare classification algorithms.
- Evaluate models using accuracy and balanced accuracy.
- Save and reload a trained model.
- Implement reusable predictions.
- Validate the workflow with automated tests.

## Dataset

- **Records:** 581,012
- **Columns:** 55
- **Input features:** 54
- **Target column:** `Cover_Type`
- **Target classes:** 1 through 7

Features include elevation, aspect, slope, distances to hydrology and roads, hillshade measurements, wilderness-area indicators, and soil-type indicators.

The dataset must be placed locally at `data/covtype.csv`. It is not included in this repository.

## Machine Learning Workflow

1. Load and validate the dataset.
2. Separate the 54 input features from the target.
3. Create an 80/20 stratified train-test split using `random_state=42`.
4. Train and compare Decision Tree, Random Forest, and Extra Trees classifiers.
5. Evaluate accuracy, balanced accuracy, and class-level metrics.
6. Save and reload the Random Forest model.
7. Verify that predictions remain consistent after reloading.
8. Run predictions and automated tests.

## Model Comparison

Results from the current experiment using the same stratified train-test split:

| Model | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Decision Tree | 93.89% | 90.38% |
| Random Forest | 95.33% | 90.53% |
| Extra Trees | 95.24% | 90.82% |

Random Forest achieved the highest overall accuracy. Extra Trees achieved the highest balanced accuracy in this comparison.

Execution times depend on the computer and available resources.

## Random Forest Evaluation

- **Accuracy:** 95.33%
- **Balanced accuracy:** 90.53%

Class-level evaluation showed that class 5 had 77% recall. This indicates that some actual class 5 examples were missed by the model. The imbalance in target-class frequencies makes this an area for further investigation.

## Project Structure

```text
Forest-Cover-Type-Prediction/
├── data/                  # Local dataset; not committed
├── models/                # Generated model and evaluation artifacts
├── src/
│   ├── __init__.py
│   ├── inspect_data.py
│   ├── validate_data.py
│   ├── data.py
│   ├── train.py
│   ├── compare_models.py
│   ├── evaluate_model.py
│   ├── train_final.py
│   └── predict.py
├── tests/
│   ├── test_data.py
│   └── test_predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.12
- Packages listed in `requirements.txt`

## Setup

Clone the repository:

```powershell
git clone https://github.com/ankitharashmireddy/Forest-Cover-Type-Prediction.git
cd Forest-Cover-Type-Prediction
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the dataset at `data/covtype.csv` before running the project scripts.

## Run the Project

Inspect the dataset:

```powershell
python -m src.inspect_data
```

Validate the dataset:

```powershell
python -m src.validate_data
```

Compare models:

```powershell
python -m src.compare_models
```

Evaluate the Random Forest model:

```powershell
python -m src.evaluate_model
```

Train and save the final Random Forest model:

```powershell
python -m src.train_final
```

Run the prediction demonstration:

```powershell
python -m src.predict
```

Run automated tests:

```powershell
python -m pytest -v
```

## Tests

The test suite contains six tests covering dataset dimensions, missing values, feature-target separation, valid prediction classes, and rejection of missing or unexpected features.

**Latest recorded test result:** 6 passed.

## Limitations and Future Improvements

- Investigate lower recall for class 5.
- Explore methods for handling class imbalance.
- Tune model hyperparameters using a validation strategy.
- Consider cross-validation for more robust model comparison.
- Add tests for invalid values and model-file errors.
- Develop an interface or API for interactive predictions.

## Disclaimer

These results reflect evaluation on the test split used in this experiment. Performance on new geographic areas may differ.
'@ | Set-Content -Path README.md -Encoding utf8
