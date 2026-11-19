"""
Tests for the Student Result Predictor data, model and app.

Run from the project root:
    pytest -v
"""

import json
import os
import shutil
import sys

import pytest
import sklearn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import model as ml  # noqa: E402

REQUIRED_COLUMNS = {"Student_ID", "Study_Hours", "Attendance", "Previous_Score", "Result"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def df():
    """The cleaned dataset."""
    return ml.load_data()


@pytest.fixture(scope="session")
def loaded_model():
    """The trained model (retrained if needed)."""
    return ml.load_model(auto_retrain=True)


@pytest.fixture
def backup_model_files(tmp_path):
    """Back up model.joblib and metrics.json and always restore them afterwards."""
    ml.load_model(auto_retrain=True)  # make sure both files exist
    saved = {}
    for path in (ml.MODEL_PATH, ml.METRICS_PATH):
        dest = tmp_path / os.path.basename(path)
        shutil.copy(path, dest)
        saved[path] = dest
    yield
    for path, dest in saved.items():
        shutil.copy(dest, path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestDataset:
    def test_minimum_row_count(self, df):
        assert len(df) >= 30

    def test_required_columns(self, df):
        assert REQUIRED_COLUMNS.issubset(df.columns)

    def test_no_missing_result(self, df):
        assert df["Result"].notna().all()

    def test_both_classes_present(self, df):
        assert set(df["Result"].unique()) == {"Pass", "Fail"}

    def test_feature_ranges_plausible(self, df):
        assert df["Study_Hours"].between(0, 24).all()
        assert df["Attendance"].between(0, 100).all()
        assert df["Previous_Score"].between(0, 100).all()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class TestPredict:
    def test_returns_valid_label_and_probability(self, loaded_model):
        label, prob = ml.predict(5.0, 70.0, 60.0, model=loaded_model)
        assert label in ("Pass", "Fail")
        assert 0.0 <= prob <= 1.0

    def test_strong_student_passes(self, loaded_model):
        label, prob = ml.predict(9.0, 95.0, 90.0, model=loaded_model)
        assert label == "Pass"
        assert prob > 0.5

    def test_weak_student_fails(self, loaded_model):
        label, prob = ml.predict(1.0, 45.0, 40.0, model=loaded_model)
        assert label == "Fail"
        assert prob < 0.5

    def test_predict_without_model_argument(self):
        label, prob = ml.predict(9.0, 95.0, 90.0)
        assert label == "Pass" and 0.0 <= prob <= 1.0

    @pytest.mark.parametrize(
        "args, field",
        [((25.0, 70.0, 60.0), "study_hours"),
         ((5.0, 101.0, 60.0), "attendance"),
         ((5.0, 70.0, -1.0), "previous_score")],
    )
    def test_invalid_input_raises(self, args, field):
        with pytest.raises(ValueError, match=field):
            ml.predict(*args)


# ---------------------------------------------------------------------------
# Model selection justification
# ---------------------------------------------------------------------------
class TestSelectModel:
    @staticmethod
    def _cv(lr, dt, rf):
        return {"Logistic Regression": {"mean": lr},
                "Decision Tree": {"mean": dt},
                "Random Forest": {"mean": rf}}

    def test_best_model_justification(self):
        name, reason = ml.select_model(self._cv(0.8938, 0.7688, 0.8438))
        assert name == "Logistic Regression"
        assert reason.startswith(
            "Logistic Regression had the highest mean cross-validation accuracy (89.4%), "
            "ahead of Random Forest (84.4%) and Decision Tree (76.9%)."
        )
        assert "within" not in reason and "below" not in reason

    def test_margin_rule_mentioned_only_when_it_decides(self):
        name, reason = ml.select_model(self._cv(0.85, 0.70, 0.86))
        assert name == "Logistic Regression"
        assert "within 2 percentage points" in reason

    def test_other_model_wins_outside_margin(self):
        name, reason = ml.select_model(self._cv(0.80, 0.70, 0.90))
        assert name == "Random Forest"
        assert reason.startswith("Random Forest had the highest")


# ---------------------------------------------------------------------------
# Persistence and auto-retraining
# ---------------------------------------------------------------------------
class TestModelPersistence:
    def test_metrics_json_has_required_keys(self, loaded_model):
        metrics = ml.load_metrics()
        required = {
            "model_name", "feature_names", "sklearn_version", "training_date",
            "test_accuracy", "cv_results", "confusion_matrix",
            "classification_report", "justification", "summary_stats",
            "feature_ranges",
        }
        assert required.issubset(metrics)
        assert metrics["sklearn_version"] == sklearn.__version__

    def test_retrains_when_model_file_deleted(self, backup_model_files):
        os.remove(ml.MODEL_PATH)
        assert ml.model_status() is not None

        retrained = ml.load_model(auto_retrain=True)

        assert os.path.exists(ml.MODEL_PATH)
        assert ml.model_status() is None
        label, prob = ml.predict(9.0, 95.0, 90.0, model=retrained)
        assert label == "Pass" and 0.0 <= prob <= 1.0
        # Retraining is deterministic (random_state=42).
        assert ml.load_metrics()["test_accuracy"] == pytest.approx(0.925)

    def test_retrains_on_sklearn_version_mismatch(self, backup_model_files):
        metrics = ml.load_metrics()
        metrics["sklearn_version"] = "0.0.0"
        with open(ml.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        assert "scikit-learn" in ml.model_status()

        ml.load_model(auto_retrain=True)

        assert ml.load_metrics()["sklearn_version"] == sklearn.__version__

    def test_retrains_on_corrupted_model_file(self, backup_model_files):
        with open(ml.MODEL_PATH, "wb") as f:
            f.write(b"not a model")

        retrained = ml.load_model(auto_retrain=True)

        assert hasattr(retrained, "predict_proba")

    def test_no_retrain_raises_when_missing(self, backup_model_files):
        os.remove(ml.MODEL_PATH)
        with pytest.raises(RuntimeError):
            ml.load_model(auto_retrain=False)


# ---------------------------------------------------------------------------
# Streamlit app smoke test (headless)
# ---------------------------------------------------------------------------
class TestApp:
    def test_app_renders_all_tabs_without_errors(self, loaded_model):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
        at.run()

        assert not at.exception
        assert not at.error
        assert len(at.tabs) == 3
        captions = " ".join(c.value for c in at.caption)
        assert "trained on study hours 1-12, attendance 50.3-100% and previous scores 40-95" in captions
        assert "Test set = 40 students" in captions
        assert "standardised features" in captions

    def test_app_predicts_pass_and_fail(self, loaded_model):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
        at.run()
        at.slider[0].set_value(9.0)
        at.slider[1].set_value(95)
        at.slider[2].set_value(90)
        at.button[0].click().run()
        assert any("Likely to PASS" in s.value for s in at.success)

        at.slider[0].set_value(1.0)
        at.slider[1].set_value(45)
        at.slider[2].set_value(40)
        at.button[0].click().run()
        assert any("Likely to FAIL" in e.value for e in at.error)
