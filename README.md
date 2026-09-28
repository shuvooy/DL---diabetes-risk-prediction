# Diabetes Prediction

Predicts whether a patient has diabetes from 8 health measurements, using a neural network (PyTorch) and a Random Forest (scikit-learn).

> This is a learning project, not a medical tool. Don't use it for real diagnoses.

## The data

100,000 patient records from [this Kaggle dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

Features: gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level.
Target: `diabetes` (0 = no, 1 = yes). Only about 8.5% of patients have diabetes.

The CSV is not included in this repo. Download it and put it in the same folder as `main.ipynb`.

## What the notebook does

1. Loads and checks the data
2. Removes duplicate rows
3. Splits it into train / validation / test (70% / 15% / 15%)
4. Preprocesses it (fills missing values, scales numbers, one-hot encodes text)
5. Trains a neural network and a Random Forest
6. Picks the best decision threshold on the validation set
7. Compares both models on the test set
8. Saves the models and predicts on new patients

## Results (Random Forest)

Tested on 14,422 patients (1,272 with diabetes):

| Metric | Score |
|---|---|
| ROC-AUC | 0.977 |
| Recall | 0.911 |
| Precision | 0.471 |
| Accuracy | 0.902 |

The model finds about 91% of diabetic patients, but about half of its "diabetes" predictions are false alarms.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

Then run all the cells.

## Files

- `main.ipynb` — the full project
- `models/` — the saved models (`ann_model.pt`, `random_forest.joblib`, `preprocessor.joblib`)
- `requirements.txt` — Python packages needed

## Tools used

Python, pandas, NumPy, scikit-learn, PyTorch
