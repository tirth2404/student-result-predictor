"""
data_prep.py
------------
Student Result Predictor: data preparation pipeline.

Reads  : data/raw/student_exam_scores.csv  (Kaggle dataset)
Writes : data.csv  (clean, labelled, leakage-free)

Columns in data.csv
  Student_ID     – unique identifier
  Study_Hours    – hours studied per day
  Attendance     – attendance percentage
  Previous_Score – previous exam score
  Result         – "Pass" if exam_score >= 35 else "Fail"

Note: exam_score is intentionally dropped to prevent label leakage.
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "student_exam_scores.csv")
OUT_PATH = os.path.join(BASE_DIR, "data.csv")
PASS_THRESHOLD = 35
SYNTHETIC_ROWS = 200
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Column mapping (Kaggle → project schema)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "student_id": "Student_ID",
    "hours_studied": "Study_Hours",
    "attendance_percent": "Attendance",
    "previous_scores": "Previous_Score",
    "exam_score": "_exam_score",   # renamed temporarily to derive Result
}


# ---------------------------------------------------------------------------
# Helper: derive Result label
# ---------------------------------------------------------------------------
def _label(exam_score: float) -> str:
    """Return 'Pass' if exam_score >= PASS_THRESHOLD, else 'Fail'."""
    return "Pass" if exam_score >= PASS_THRESHOLD else "Fail"


# ---------------------------------------------------------------------------
# Source 1: Real Kaggle CSV
# ---------------------------------------------------------------------------
def load_from_csv(path: str) -> pd.DataFrame:
    """
    Load and clean the Kaggle student-exam-scores CSV.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns:
        Student_ID, Study_Hours, Attendance, Previous_Score, Result.
    """
    df = pd.read_csv(path)

    # Validate required columns exist
    required = set(COLUMN_MAP.keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing columns: {missing}")

    # Rename to project schema; drop sleep_hours and exam_score
    df = df.rename(columns=COLUMN_MAP)
    df["Result"] = df["_exam_score"].apply(_label)
    keep = ["Student_ID", "Study_Hours", "Attendance", "Previous_Score", "Result"]
    df = df[keep]

    return df


# ---------------------------------------------------------------------------
# Source 2: Synthetic fallback
# ---------------------------------------------------------------------------
def generate_synthetic(n: int = SYNTHETIC_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset when the raw CSV is unavailable.

    Features are correlated with the outcome to mimic real-world patterns.
    A small fraction (~5 %) of rows are intentional noise cases (high study
    hours but Fail, or low but Pass) to prevent a trivially separable dataset.

    Parameters
    ----------
    n    : int  – number of rows to generate (default 200)
    seed : int  – NumPy random seed for reproducibility

    Returns
    -------
    pd.DataFrame with Student_ID, Study_Hours, Attendance,
    Previous_Score, Result.
    """
    rng = np.random.default_rng(seed)

    study_hours    = rng.uniform(0.5, 10.0, n).round(1)
    attendance     = rng.uniform(40.0, 100.0, n).round(1)
    previous_score = rng.uniform(20.0, 100.0, n).round(1)

    # Composite score: weighted combination of features
    composite = (
        study_hours    * 2.5
        + attendance   * 0.3
        + previous_score * 0.4
    )

    # Normalise to [0, 1] then scale to exam_score-like range [10, 70]
    comp_min, comp_max = composite.min(), composite.max()
    exam_scores = 10 + 60 * (composite - comp_min) / (comp_max - comp_min)
    exam_scores = exam_scores.round(1)

    # Inject ~5 % noise by flipping borderline labels
    noise_idx = rng.choice(n, size=max(1, n // 20), replace=False)
    exam_scores[noise_idx] = rng.uniform(28, 42, len(noise_idx)).round(1)

    results    = ["Pass" if s >= PASS_THRESHOLD else "Fail" for s in exam_scores]
    student_ids = [f"S{str(i+1).zfill(3)}" for i in range(n)]

    df = pd.DataFrame({
        "Student_ID":     student_ids,
        "Study_Hours":    study_hours,
        "Attendance":     attendance,
        "Previous_Score": previous_score,
        "Result":         results,
    })
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def prepare_data() -> pd.DataFrame:
    """
    Full data-preparation pipeline.

    Tries to read the real CSV first; falls back to synthetic generation.
    Saves the result to OUT_PATH and returns the dataframe.

    Returns
    -------
    pd.DataFrame – the cleaned / generated dataset.
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    # --- Attempt to read real data ---
    if os.path.exists(RAW_PATH):
        try:
            df = load_from_csv(RAW_PATH)
            source = "REAL (Kaggle CSV)"
        except Exception as exc:
            print(f"[WARNING] Could not parse the raw CSV: {exc}")
            print("[INFO]    Falling back to synthetic dataset.")
            df = generate_synthetic()
            source = "SYNTHETIC (fallback)"
    else:
        print("[INFO] Raw file not found at data/raw/student_exam_scores.csv.")
        print("[INFO] Generating synthetic dataset instead.")
        df = generate_synthetic()
        source = "SYNTHETIC (fallback)"

    # --- Validate minimum row count ---
    if len(df) < 30:
        raise RuntimeError(
            f"Dataset has only {len(df)} rows – minimum 30 required. "
            "Check the raw CSV or increase SYNTHETIC_ROWS."
        )

    # --- Print diagnostic info ---
    print("=" * 60)
    print(f"  DATA SOURCE : {source}")
    print(f"  Rows        : {len(df)}")
    print(f"  Columns     : {list(df.columns)}")
    print("-" * 60)
    print("  CLASS BALANCE")
    balance = df["Result"].value_counts()
    for label, count in balance.items():
        pct = 100 * count / len(df)
        print(f"    {label:5s} : {count:4d}  ({pct:.1f} %)")
    print("=" * 60)

    # --- Save ---
    df.to_csv(OUT_PATH, index=False)
    print("\n[OK] Saved cleaned data to data.csv\n")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prepare_data()
