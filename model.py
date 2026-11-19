"""
model.py
--------
Student Result Predictor: model training, evaluation and inference.

Reads : data.csv  (produced by data_prep.py)
Saves : models/model.joblib
        models/metrics.json

Public API (importing this module has no side effects):
    load_data()
    get_summary_stats(df)
    train_and_evaluate()
    model_status()
    load_model(auto_retrain=True)
    predict(study_hours, attendance, previous_score) -> (label, prob_of_pass)
"""

import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Constants (paths are absolute so the module works from any working directory)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURES = ["Study_Hours", "Attendance", "Previous_Score"]
TARGET = "Result"
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# Prefer the simpler model if it is within this margin of the best CV score.
SIMPLE_MODEL = "Logistic Regression"
PREFERENCE_MARGIN = 0.02  # 2 percentage points


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset from CSV.

    Parameters
    ----------
    path : str
        Path to data.csv.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run data_prep.py first to generate it."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Missing-value handling
# ---------------------------------------------------------------------------
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report and handle missing values.

    Numeric features are imputed with the column median; rows with a missing
    Result are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    print("\n--- Missing Value Report ---")
    null_counts = df.isnull().sum()

    if null_counts.sum() == 0:
        print("  No missing values found. Dataset is complete.")
        return df

    for col, cnt in null_counts[null_counts > 0].items():
        print(f"  '{col}': {cnt} missing value(s)")

    for col in FEATURES:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  -> Imputed '{col}' with median = {median_val:.2f}")

    before = len(df)
    df = df.dropna(subset=[TARGET])
    if before - len(df):
        print(f"  -> Dropped {before - len(df)} row(s) with missing '{TARGET}'.")
    return df


# ---------------------------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------------------------
def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute and print descriptive statistics for the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Averages, class counts and row count.
    """
    print("\n--- Summary Statistics ---")

    avg_study = float(np.mean(df["Study_Hours"].values))
    avg_attend = float(np.mean(df["Attendance"].values))
    avg_prev = float(np.mean(df["Previous_Score"].values))
    pass_count = int((df[TARGET] == "Pass").sum())
    fail_count = int((df[TARGET] == "Fail").sum())

    print(f"  Average Study Hours    : {avg_study:.2f}")
    print(f"  Average Attendance (%) : {avg_attend:.2f}")
    print(f"  Average Previous Score : {avg_prev:.2f}")
    print(f"  Pass Count             : {pass_count}")
    print(f"  Fail Count             : {fail_count}")
    print(f"  Pass Rate              : {100 * pass_count / len(df):.1f} %")

    return {
        "avg_study_hours": round(avg_study, 4),
        "avg_attendance": round(avg_attend, 4),
        "avg_previous_score": round(avg_prev, 4),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "total_rows": len(df),
    }


def get_feature_ranges(df: pd.DataFrame) -> dict:
    """Return {feature: {"min": x, "max": y}} for each model feature."""
    return {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in FEATURES
    }


# ---------------------------------------------------------------------------
# 4. Candidate models
# ---------------------------------------------------------------------------
def _build_candidates() -> dict:
    """
    Return a dict of named, unfitted estimators to compare.

    Logistic Regression is wrapped in a StandardScaler pipeline because it is
    sensitive to feature scale. Tree models are scale-invariant.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }


# ---------------------------------------------------------------------------
# 5. Cross-validation comparison
# ---------------------------------------------------------------------------
def compare_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Run stratified k-fold cross-validation for each candidate model.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels (Pass=1, Fail=0).

    Returns
    -------
    dict[str, dict]
        Per-model mean, std, min, max and individual fold scores.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    print(f"\n--- {CV_FOLDS}-Fold Stratified Cross-Validation ---")
    print(f"  {'Model':<22} {'Mean Acc':>9} {'Std Dev':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 60)

    for name, model in _build_candidates().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "scores": [round(float(s), 4) for s in scores],
        }
        print(
            f"  {name:<22} {scores.mean():>8.4f}  {scores.std():>7.4f}  "
            f"{scores.min():>7.4f}  {scores.max():>7.4f}"
        )

    print("  " + "-" * 60)
    return cv_results


# ---------------------------------------------------------------------------
# 6. Model selection
# ---------------------------------------------------------------------------
def _pct(score: float) -> str:
    """Format a 0-1 score as a percentage with one decimal place."""
    return f"{score * 100:.1f}%"


def select_model(cv_results: dict) -> tuple[str, str]:
    """
    Select the final model from cross-validation results.

    Rule
    ----
    1. Take the model with the highest mean CV accuracy.
    2. If that is not Logistic Regression but Logistic Regression is within
       PREFERENCE_MARGIN of it, choose Logistic Regression instead, because
       it is simpler, interpretable and gives calibrated probabilities.

    The justification text describes whichever rule actually decided.

    Parameters
    ----------
    cv_results : dict
        Output of compare_models().

    Returns
    -------
    tuple[str, str]
        (chosen model name, plain-English justification)
    """
    ranked = sorted(cv_results, key=lambda k: cv_results[k]["mean"], reverse=True)
    best_name = ranked[0]
    best_score = cv_results[best_name]["mean"]
    simple_score = cv_results[SIMPLE_MODEL]["mean"]

    if best_name != SIMPLE_MODEL and best_score - simple_score <= PREFERENCE_MARGIN:
        chosen = SIMPLE_MODEL
        reason = (
            f"{best_name} had the highest mean cross-validation accuracy "
            f"({_pct(best_score)}), but {SIMPLE_MODEL} ({_pct(simple_score)}) was "
            f"within {PREFERENCE_MARGIN * 100:.0f} percentage points of it, so the "
            f"simpler model was preferred. {SIMPLE_MODEL} is interpretable and "
            "gives well-calibrated probabilities, which suits a small dataset "
            "with three numeric features."
        )
    else:
        chosen = best_name
        others = [f"{n} ({_pct(cv_results[n]['mean'])})" for n in ranked[1:]]
        if len(others) > 1:
            others_text = ", ".join(others[:-1]) + f" and {others[-1]}"
        else:
            others_text = others[0] if others else ""
        reason = (
            f"{chosen} had the highest mean cross-validation accuracy "
            f"({_pct(best_score)})"
            + (f", ahead of {others_text}." if others_text else ".")
        )
        if chosen == SIMPLE_MODEL:
            reason += (
                " It is also simple, interpretable and gives well-calibrated "
                "probabilities, which suits a small dataset with three numeric "
                "features."
            )

    print(f"\n  -> Selected model : {chosen}")
    print(f"  -> Reason         : {reason}")
    return chosen, reason


# ---------------------------------------------------------------------------
# 7. Full train-and-evaluate pipeline
# ---------------------------------------------------------------------------
def train_and_evaluate() -> dict:
    """
    End-to-end pipeline: load -> clean -> split -> CV -> train -> evaluate -> save.

    If data.csv is missing it is generated with data_prep.prepare_data().

    Returns
    -------
    dict
        Metrics dictionary (also written to metrics.json).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        import data_prep
        data_prep.prepare_data()

    df = load_data()
    print("\nDataset loaded from data.csv")
    print(f"  Shape  : {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")

    df = handle_missing(df)
    summary = get_summary_stats(df)

    X = df[FEATURES].values
    y = (df[TARGET] == "Pass").astype(int).values  # Pass=1, Fail=0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("\n--- Train / Test Split ---")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")

    cv_results = compare_models(X_train, y_train)
    chosen_name, justification = select_model(cv_results)

    final_model = _build_candidates()[chosen_name]
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Fail", "Pass"], output_dict=True, zero_division=0
    )

    print("\n--- Final Model Evaluation on Test Set ---")
    print(f"  Model     : {chosen_name}")
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f} %)")
    print("\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    print("               Fail  Pass")
    print(f"    Actual Fail  {cm[0, 0]:3d}   {cm[0, 1]:3d}")
    print(f"    Actual Pass  {cm[1, 0]:3d}   {cm[1, 1]:3d}")
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred, target_names=["Fail", "Pass"], zero_division=0
    ))

    joblib.dump(final_model, MODEL_PATH)
    print("[OK] Model saved to models/model.joblib")

    metrics = {
        "model_name": chosen_name,
        "feature_names": FEATURES,
        "sklearn_version": sklearn.__version__,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_accuracy": round(float(acc), 4),
        "cv_results": cv_results,
        "selected_model_cv": cv_results[chosen_name],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "justification": justification,
        "summary_stats": summary,
        "feature_ranges": get_feature_ranges(df),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[OK] Metrics saved to models/metrics.json")

    return metrics


# ---------------------------------------------------------------------------
# 8. Loading and inference
# ---------------------------------------------------------------------------
def load_metrics() -> dict:
    """Load metrics.json as a dict."""
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def model_status() -> str | None:
    """
    Check whether the saved model can be used as-is.

    Returns
    -------
    str | None
        None if the model is usable, otherwise a short reason why it must be
        retrained (missing file, missing metrics, or scikit-learn mismatch).
    """
    if not os.path.exists(MODEL_PATH):
        return "model file is missing"
    try:
        saved_version = load_metrics().get("sklearn_version")
    except (OSError, ValueError):
        return "metrics file is missing or unreadable"
    if saved_version != sklearn.__version__:
        return (
            f"model was trained with scikit-learn {saved_version}, "
            f"installed version is {sklearn.__version__}"
        )
    return None


def load_model(auto_retrain: bool = True):
    """
    Load the trained model from disk, retraining it when necessary.

    The model is retrained (and re-saved) if the file is missing, cannot be
    unpickled, or was saved with a different scikit-learn version. Training is
    deterministic (random_state=42), so a retrained model matches the original.

    Parameters
    ----------
    auto_retrain : bool
        If False, raise instead of retraining.

    Returns
    -------
    sklearn estimator
    """
    reason = model_status()
    if reason is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return joblib.load(MODEL_PATH)
        except Exception as exc:  # corrupted or incompatible pickle
            reason = f"model file could not be loaded ({exc})"

    if not auto_retrain:
        raise RuntimeError(f"Model unavailable: {reason}.")

    print(f"[INFO] Retraining model: {reason}.")
    train_and_evaluate()
    return joblib.load(MODEL_PATH)


def predict(
    study_hours: float,
    attendance: float,
    previous_score: float,
    model=None,
) -> tuple[str, float]:
    """
    Predict Pass/Fail for a single student.

    Parameters
    ----------
    study_hours : float
        Hours studied per day, 0-24.
    attendance : float
        Attendance percentage, 0-100.
    previous_score : float
        Previous exam score, 0-100.
    model : estimator, optional
        A loaded model. If omitted, load_model() is called.

    Returns
    -------
    tuple[str, float]
        ("Pass" or "Fail", probability of passing in [0, 1]).

    Raises
    ------
    ValueError
        If an input is outside its valid range.
    """
    if not 0 <= study_hours <= 24:
        raise ValueError(f"study_hours={study_hours} must be in [0, 24].")
    if not 0 <= attendance <= 100:
        raise ValueError(f"attendance={attendance} must be in [0, 100].")
    if not 0 <= previous_score <= 100:
        raise ValueError(f"previous_score={previous_score} must be in [0, 100].")

    if model is None:
        model = load_model()

    X_input = np.array([[study_hours, attendance, previous_score]], dtype=float)
    prob_pass = float(model.predict_proba(X_input)[0][1])
    label = "Pass" if model.predict(X_input)[0] == 1 else "Fail"
    return label, round(prob_pass, 4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_evaluate()

    print("\n--- Predict Function Smoke-Test ---")
    print(f"  {'Study Hrs':>10} {'Attend':>7} {'Prev Sc':>8} {'Label':>6} {'P(Pass)':>8}")
    print("  " + "-" * 46)
    trained = load_model()
    for sh, att, ps in [(8.0, 90.0, 80.0), (1.0, 45.0, 25.0), (5.0, 70.0, 55.0)]:
        label, prob = predict(sh, att, ps, model=trained)
        print(f"  {sh:>10.1f} {att:>7.1f} {ps:>8.1f} {label:>6} {prob:>8.4f}")
