# Student Performance Predictor (ML Pipeline)

Predicts whether a student will **pass or fail** using classification models
trained on academic and lifestyle features (study hours, attendance,
previous scores, parental support, sleep, etc.).

This is the machine learning counterpart to the web-based Student
Performance Predictor — this script trains and evaluates real models on
data, rather than using a fixed scoring formula.

## Pipeline

1. **Load data** — uses a realistic synthetic dataset (1,000 students) so
   the whole pipeline runs end-to-end with no external file needed. Swap in
   a real CSV (e.g. the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance))
   by passing `csv_path` to `load_data()`.
2. **Preprocess** — fills missing values, label-encodes categorical
   features (parental support, internet access, extracurriculars).
3. **EDA** — saves a feature correlation heatmap and a score distribution
   plot.
4. **Train models** — Logistic Regression and Random Forest classifiers,
   compared on accuracy, ROC-AUC, and 5-fold cross-validation.
5. **Evaluate** — confusion matrix and feature importance plots for the
   winning model.
6. **Save** — best model and scaler are pickled with `joblib` for reuse.
7. **Predict** — example inference on a new, unseen student profile.

## Results (this run)

| Model | Accuracy | ROC-AUC | CV Mean |
|---|---|---|---|
| Logistic Regression | 0.840 | 0.873 | 0.819 |
| Random Forest | 0.830 | 0.850 | 0.818 |

Best model: **Logistic Regression**. `previous_score` and `study_hours`
are the strongest predictors of `final_score` and `passed` (see
`correlation_heatmap.png` and `feature_importance.png`).

*(Since the dataset is randomly generated each run, exact numbers will
vary slightly run to run — set a real CSV via `csv_path` for reproducible,
meaningful results.)*

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python student_prediction.py
```

## Output files

Running the script produces:
- `correlation_heatmap.png` — feature correlations
- `score_distribution.png` — distribution of final scores
- `confusion_matrix.png` — confusion matrix for the best model
- `feature_importance.png` — Random Forest feature importances
- `best_model.pkl` — trained model, saved with joblib
- `scaler.pkl` — fitted StandardScaler, saved with joblib

## Using a real dataset

Replace the synthetic generator with your own CSV:

```python
df_raw = load_data(csv_path="your_dataset.csv")
```

Your CSV should include a `final_score` (numeric) and/or `passed`
(0/1) target column, plus whatever feature columns you have — the
pipeline auto-detects numeric vs. categorical columns during
preprocessing.

## Reusing the saved model

```python
import joblib
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")  # only needed if the best model was Logistic Regression
```

## Tech stack

Python · pandas · NumPy · scikit-learn (Logistic Regression, Random
Forest) · matplotlib · seaborn · joblib
