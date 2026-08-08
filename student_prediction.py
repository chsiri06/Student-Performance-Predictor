
"""
Student Performance Prediction Project
========================================
Predicts whether a student will PASS or FAIL (classification) and their
final score (regression) based on features like study hours, attendance,
previous scores, parental support, etc.

Pipeline:
1. Load data (synthetic generator included, or plug in a real CSV)
2. Clean & preprocess (missing values, encoding, scaling)
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Train/test split
6. Train models (Logistic Regression, Random Forest)
7. Evaluate (accuracy, precision/recall, confusion matrix, feature importance)
8. Save the trained model for reuse

Run:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib --break-system-packages
    python student_prediction.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    If csv_path is given, load a real dataset (e.g. UCI Student Performance
    dataset: https://archive.ics.uci.edu/dataset/320/student+performance).

    Otherwise, generate a realistic synthetic dataset so the pipeline is
    runnable end-to-end without any external file.
    """
    if csv_path:
        return pd.read_csv(csv_path)

    n = 1000
    study_hours = np.random.normal(5, 2, n).clip(0, 12)
    attendance = np.random.normal(80, 12, n).clip(30, 100)
    previous_score = np.random.normal(65, 15, n).clip(0, 100)
    parental_support = np.random.choice(["Low", "Medium", "High"], n, p=[0.25, 0.45, 0.30])
    internet_access = np.random.choice(["Yes", "No"], n, p=[0.8, 0.2])
    extracurricular = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
    sleep_hours = np.random.normal(7, 1.2, n).clip(3, 10)

    support_bonus = {"Low": -5, "Medium": 0, "High": 5}
    final_score = (
        0.35 * previous_score
        + 3.0 * study_hours
        + 0.25 * attendance
        + np.array([support_bonus[s] for s in parental_support])
        + np.where(internet_access == "Yes", 3, -2)
        + np.random.normal(0, 8, n)
    )
    final_score = np.clip(final_score, 0, 100)
    passed = (final_score >= 50).astype(int)

    df = pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous_score.round(1),
        "parental_support": parental_support,
        "internet_access": internet_access,
        "extracurricular": extracurricular,
        "sleep_hours": sleep_hours.round(1),
        "final_score": final_score.round(1),
        "passed": passed,
    })
    return df


# -----------------------------------------------------------------------
# 2. PREPROCESSING
# -----------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Handle missing values (if any real-world data has them)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# -----------------------------------------------------------------------
# 3. EDA (saves plots instead of showing them interactively)
# -----------------------------------------------------------------------
def run_eda(df: pd.DataFrame, out_dir: str = "."):
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["final_score"], kde=True, bins=25)
    plt.title("Distribution of Final Scores")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/score_distribution.png", dpi=150)
    plt.close()

    print("EDA plots saved: correlation_heatmap.png, score_distribution.png")


# -----------------------------------------------------------------------
# 4. TRAIN + EVALUATE
# -----------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame, out_dir: str = "."):
    feature_cols = [c for c in df.columns if c not in ("final_score", "passed")]
    X = df[feature_cols]
    y = df["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f} | CV mean: {cv_scores.mean():.3f}")
        print(classification_report(y_test, preds, target_names=["Fail", "Pass"]))

        results[name] = {
            "model": model, "accuracy": acc, "auc": auc,
            "preds": preds, "probs": probs
        }

    # Confusion matrix for the best model (by accuracy)
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best = results[best_name]
    cm = confusion_matrix(y_test, best["preds"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (Random Forest)
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=150)
    plt.close()

    print(f"\nBest model: {best_name} (Accuracy: {best['accuracy']:.3f})")
    print("Saved plots: confusion_matrix.png, feature_importance.png")

    # Save the best model + scaler for later reuse
    joblib.dump(best["model"], f"{out_dir}/best_model.pkl")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    print(f"Saved trained model -> {out_dir}/best_model.pkl")

    return results, feature_cols


# -----------------------------------------------------------------------
# 5. PREDICT ON NEW STUDENT DATA
# -----------------------------------------------------------------------
def predict_new_student(model, scaler, feature_cols, sample: dict, uses_scaling: bool):
    row = pd.DataFrame([sample])[feature_cols]
    if uses_scaling:
        row = scaler.transform(row)
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    return ("Pass" if pred == 1 else "Fail"), prob


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    df_raw = load_data(csv_path=None)  # replace with "your_dataset.csv" for real data
    print(df_raw.head())

    print("\nPreprocessing...")
    df_processed, encoders = preprocess(df_raw)

    print("\nRunning EDA...")
    run_eda(df_processed)

    print("\nTraining models...")
    results, feature_cols = train_and_evaluate(df_processed)

    print("\nExample prediction for a new student:")
    sample_student = {
        "study_hours": 6.5,
        "attendance": 88,
        "previous_score": 72,
        "parental_support": encoders["parental_support"].transform(["High"])[0],
        "internet_access": encoders["internet_access"].transform(["Yes"])[0],
        "extracurricular": encoders["extracurricular"].transform(["No"])[0],
        "sleep_hours": 7.0,
    }
    rf_model = results["Random Forest"]["model"]
    outcome, prob = predict_new_student(rf_model, None, feature_cols, sample_student, uses_scaling=False)
    print(f"Prediction: {outcome} (Probability of passing: {prob:.2%})")