# Forest Cover Type Prediction

<<<<<<< HEAD
A machine learning project that predicts forest cover types from cartographic and environmental features using the U.S. Forest Service Covertype dataset.

## Project Overview

Forest cover type depends on environmental and geographic characteristics such as elevation, slope, distances to water and roads, hillshade measurements, wilderness areas, and soil types.

This project explores supervised machine learning to classify a forest location into one of seven cover types.

## Objectives

- Inspect and validate the dataset.
- Train and compare multiple classification algorithms.
- Evaluate model performance using accuracy and balanced accuracy.
- Save a trained model for reuse.
- Provide a reusable prediction function.
- Test the data-processing and prediction workflows.

## Dataset

The dataset contains 581,012 records and 55 columns:

- 54 input features.
- 1 target column: `Cover_Type`.
- 7 target classes, numbered 1 through 7.

The dataset file is kept locally and is not included in this public repository.

## Machine Learning Workflow

1. Load the dataset and assign the expected column names.
2. Validate the dataset structure, missing values, target labels, and categorical indicator columns.
3. Split the features and target.
4. Create an 80/20 train-test split with stratification and `random_state=42`.
5. Train and compare Decision Tree, Random Forest, and Extra Trees classifiers.
6. Evaluate performance using accuracy, balanced accuracy, a classification report, and a confusion matrix.
7. Save the Random Forest model and verify that predictions remain identical after reloading it.
8. Run predictions using the saved model.
9. Execute automated tests.

## Model Comparison

Results from the current experiment:
=======
A machine learning project that predicts forest cover types from cartographic and environmental features using the Covertype dataset.

## Overview

Forest cover type classification uses geographic and environmental measurements to predict which type of forest cover is present at a location.

This project implements a supervised machine learning workflow, from data validation and model comparison to evaluation and reusable predictions.

## Dataset

- **Records:** 581,012
- **Columns:** 55
- **Input features:** 54
- **Target:** `Cover_Type`
- **Classes:** 1–7

The dataset includes elevation, aspect, slope, distances to hydrology and roads, hillshade measurements, wilderness-area indicators, and soil-type indicators.

The dataset CSV is stored locally and is not included in this repository.

## Models Compared

Three classification algorithms were evaluated using the same stratified 80/20 train-test split with `random_state=42`.
>>>>>>> 0c8b20f (Add model comparison, evaluation, prediction, and tests)

| Model | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Decision Tree | 93.89% | 90.38% |
| Random Forest | 95.33% | 90.53% |
| Extra Trees | 95.24% | 90.82% |

<<<<<<< HEAD
Random Forest achieved the highest overall accuracy in this comparison. Extra Trees achieved the highest balanced accuracy.

These results come from the same stratified 20% test split with `random_state=42`. Runtime depends on the machine and its system load.
=======
Random Forest achieved the highest overall accuracy, while Extra Trees achieved the highest balanced accuracy in this experiment.
>>>>>>> 0c8b20f (Add model comparison, evaluation, prediction, and tests)

## Random Forest Evaluation

The Random Forest model achieved:

<<<<<<< HEAD
- Accuracy: 95.33%
- Balanced accuracy: 90.53%

The classification report showed that recall varied across classes. In particular, class 5 had 77% recall, indicating that there is room to improve recognition of this class.
=======
- **Accuracy:** 95.33%
- **Balanced accuracy:** 90.53%

Class-level evaluation showed lower recall for cover type 5 (77%), indicating an opportunity for further improvement.

## Workflow

1. Load and validate the dataset.
2. Separate the 54 input features from the target.
3. Split the data into training and test sets.
4. Train and compare Decision Tree, Random Forest, and Extra Trees.
5. Evaluate model performance using classification metrics.
6. Save and reload the Random Forest model.
7. Make predictions using the saved model.
8. Run automated tests.
>>>>>>> 0c8b20f (Add model comparison, evaluation, prediction, and tests)

## Project Structure

```text
Forest-Cover-Type-Prediction/
├── data/
├── models/
├── notebooks/
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
- Dependencies listed in `requirements.txt`

## Setup

Clone the repository:

```bash
git clone https://github.com/ankitharashmireddy/Forest-Cover-Type-Prediction.git
cd Forest-Cover-Type-Prediction
```

<<<<<<< HEAD
Create and activate a virtual environment on Windows:
=======
Create a virtual environment on Windows:
>>>>>>> 0c8b20f (Add model comparison, evaluation, prediction, and tests)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

<<<<<<< HEAD
Place the Covertype CSV file at `data/covtype.csv`. The dataset is not included in the repository.
=======
Place the Covertype CSV at `data/covtype.csv` before running the scripts. The dataset is not included in this repository.
>>>>>>> 0c8b20f (Add model comparison, evaluation, prediction, and tests)

## Run the Project

Validate the dataset:

```powershell
python -m src.validate_data
```

<<<<<<< HEAD
Compare the models:
=======
Compare models:
>>>>>>> 0c8b20f (Add model comparison, evaluation, prediction, and tests)

```powershell
python -m src.compare_models
```

Evaluate Random Forest:

```powershell
python -m src.evaluate_model
```

Train and save the Random Forest model:

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

<<<<<<< HEAD
## Testing

The current automated test suite contains six tests covering:

- Dataset dimensions.
- Missing values.
- Feature-target separation.
- Valid prediction classes.
- Missing-feature rejection.
- Unexpected-feature rejection.

Latest recorded test run: **6 passed**.

## Limitations and Future Improvements

- Investigate the lower recall for cover type 5.
- Explore class-specific evaluation and methods for handling class imbalance.
- Tune model hyperparameters using an appropriate validation strategy.
- Consider cross-validation for more robust model comparison.
- Add a user interface or API for interactive predictions.
- Add more tests for invalid values and model-file errors.

## Disclaimer

The model's predictions are based on the dataset and features used during training. Performance on new geographic areas may differ from the reported test results.
=======
## Tests

The current test suite contains six tests covering dataset shape, missing values, feature-target separation, valid prediction classes, missing features, and unexpected features.

Latest recorded test run: **6 passed**.

## Future Improvements

- Improve recall for underperforming cover types.
- Tune model hyperparameters.
- Evaluate models with cross-validation.
- Explore approaches to class imbalance.
- Develop a simple interactive prediction interface.

## Disclaimer

Performance on new geographic areas may differ from the reported test results.
>>>>>>> 0c8b20f (Add model comparison, evaluation, prediction, and tests)
