"""
app.py
------
Student Result Predictor: Streamlit web application.

Tabs:
  Dashboard         - dataset overview, charts, raw data
  Predict Result    - interactive prediction form
  Model Performance - metrics, confusion matrix, feature influence

Run:
  streamlit run app.py
"""

import matplotlib

matplotlib.use("Agg")  # headless backend for servers

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import streamlit as st

import model as ml

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Colours and chart styling (green = Pass, red = Fail everywhere)
# ---------------------------------------------------------------------------
CLR_PASS = "#2ecc71"
CLR_FAIL = "#e74c3c"
CLR_NEUTRAL = "#3498db"
RESULT_COLOURS = [("Pass", CLR_PASS), ("Fail", CLR_FAIL)]
TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 9


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def _load_df() -> pd.DataFrame:
    """Load data.csv, generating it with data_prep if it is missing."""
    try:
        return ml.load_data()
    except FileNotFoundError:
        import data_prep
        data_prep.prepare_data()
        return ml.load_data()


@st.cache_resource(show_spinner=False)
def _load_model_and_metrics():
    """
    Return (model, metrics).

    Retrains first if the saved model is missing, unreadable or was saved with
    a different scikit-learn version, so a cold start never crashes.
    """
    with st.spinner("Preparing model..."):
        model = ml.load_model(auto_retrain=True)
    return model, ml.load_metrics()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _fmt_prob(prob: float) -> str:
    """Format a pass probability; never show exactly 0% or 100%."""
    pct = prob * 100
    if pct < 1:
        return "<1%"
    if pct > 99:
        return ">99%"
    return f"{pct:.1f}%"


def _progress_val(prob: float) -> float:
    """Clamp a probability to [0.01, 0.99] for st.progress."""
    return max(0.01, min(0.99, prob))


def _fmt_num(value: float) -> str:
    """Format a number without trailing zeros (12.0 -> '12', 50.3 -> '50.3')."""
    return f"{value:g}"


def _show(fig: plt.Figure) -> None:
    """Render a matplotlib figure and free its memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent title, label and spine styling."""
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=TICK_SIZE)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _chart_pass_fail(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of Pass vs Fail counts."""
    counts = df["Result"].value_counts().reindex(["Pass", "Fail"], fill_value=0)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    bars = ax.bar(counts.index, counts.values,
                  color=[CLR_PASS, CLR_FAIL], width=0.5, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom",
                fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)
    _style_axes(ax, "Pass vs Fail Distribution", ylabel="Number of Students")
    fig.tight_layout()
    return fig


def _chart_scatter(df: pd.DataFrame) -> plt.Figure:
    """Scatter of Study Hours vs Previous Score, coloured by Result."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for label, clr in RESULT_COLOURS:
        sub = df[df["Result"] == label]
        ax.scatter(sub["Study_Hours"], sub["Previous_Score"],
                   c=clr, label=label, alpha=0.7,
                   edgecolors="#333333", linewidths=0.3, s=36)
    _style_axes(ax, "Study Hours vs Previous Score",
                "Daily Study Hours", "Previous Score")
    ax.legend(title="Result", fontsize=TICK_SIZE, title_fontsize=TICK_SIZE,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    return fig


def _chart_attendance(df: pd.DataFrame) -> plt.Figure:
    """Grouped (side-by-side) bars of attendance bands for Pass and Fail."""
    edges = np.histogram_bin_edges(df["Attendance"], bins=10)
    centres = (edges[:-1] + edges[1:]) / 2
    bar_w = (edges[1] - edges[0]) * 0.4

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for offset, (label, clr) in zip((-bar_w / 2, bar_w / 2), RESULT_COLOURS):
        counts, _ = np.histogram(df.loc[df["Result"] == label, "Attendance"], bins=edges)
        ax.bar(centres + offset, counts, width=bar_w, color=clr,
               label=label, edgecolor="white", linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax, "Attendance by Result", "Attendance (%)", "Number of Students")
    ax.legend(title="Result", fontsize=TICK_SIZE, title_fontsize=TICK_SIZE,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    return fig


def _chart_confusion(cm: list) -> plt.Figure:
    """Annotated confusion-matrix heatmap."""
    cm_arr = np.array(cm)
    labels = ["Fail", "Pass"]
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(cm_arr, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels, fontsize=LABEL_SIZE)
    ax.set_yticks(range(len(labels)), labels, fontsize=LABEL_SIZE)
    ax.set_xlabel("Predicted", fontsize=LABEL_SIZE)
    ax.set_ylabel("Actual", fontsize=LABEL_SIZE)
    ax.set_title("Confusion Matrix", fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    thresh = cm_arr.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm_arr[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def _chart_feature_influence(model, feature_names: list) -> plt.Figure | None:
    """
    Horizontal bar chart of |coefficient| (Logistic Regression) or feature
    importance (tree models). Returns None if neither is available.
    """
    clf = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model

    if hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        xlabel = "|Coefficient| (standardised features)"
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        xlabel = "Feature Importance"
    else:
        return None

    order = np.argsort(importances)
    names = [feature_names[i].replace("_", " ") for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.barh(names, values, color=CLR_NEUTRAL, edgecolor="white")
    pad = values.max() * 0.02
    for bar, val in zip(bars, values):
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=TICK_SIZE)
    ax.set_xlim(0, values.max() * 1.25)  # room for value labels
    _style_axes(ax, "Feature Influence", xlabel)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fail tip
# ---------------------------------------------------------------------------
def _fail_tip(study: float, attend: float, prev: float, stats: dict) -> str:
    """Return an improvement tip for the input furthest below the dataset average."""
    gaps = {
        "study": (stats["avg_study_hours"] - study) / stats["avg_study_hours"],
        "attendance": (stats["avg_attendance"] - attend) / stats["avg_attendance"],
        "previous": (stats["avg_previous_score"] - prev) / stats["avg_previous_score"],
    }
    weakest = max(gaps, key=gaps.get)
    tips = {
        "study": (
            f"💡 **Tip:** Study hours ({study}h/day) are below the class average "
            f"({stats['avg_study_hours']:.1f}h). Even 1-2 extra hours a day can "
            "make a significant difference."
        ),
        "attendance": (
            f"💡 **Tip:** Attendance ({attend}%) is below the class average "
            f"({stats['avg_attendance']:.1f}%). Attending more classes improves "
            "understanding and exam readiness."
        ),
        "previous": (
            f"💡 **Tip:** The previous score ({prev}) is below the class average "
            f"({stats['avg_previous_score']:.1f}). Reviewing past material and "
            "practising past papers can boost performance."
        ),
    }
    return tips[weakest]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def _render_header() -> None:
    st.title("🎓 Student Result Predictor")
    st.markdown(
        "_Predict whether a student is likely to **Pass** or **Fail** from "
        "study hours, attendance and previous score, using machine learning._"
    )
    st.divider()


# ---------------------------------------------------------------------------
# Tab: Dashboard
# ---------------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame, stats: dict) -> None:
    st.subheader("📊 Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Students", stats["total_rows"])
    c2.metric("✅ Passed", stats["pass_count"])
    c3.metric("❌ Failed", stats["fail_count"])
    c4.metric("📅 Avg Attendance", f"{stats['avg_attendance']:.1f}%")
    c5.metric("📝 Avg Previous Score", f"{stats['avg_previous_score']:.1f}")

    st.markdown("")

    col1, col2, col3 = st.columns([1, 1.15, 1.15])
    with col1:
        _show(_chart_pass_fail(df))
    with col2:
        _show(_chart_scatter(df))
    with col3:
        _show(_chart_attendance(df))

    st.markdown("")

    with st.expander("📋 View Dataset", expanded=False):
        st.dataframe(df, width="stretch", height=320)
        st.download_button(
            label="⬇️ Download data.csv",
            data=df.to_csv(index=False),
            file_name="student_data.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Tab: Predict Result
# ---------------------------------------------------------------------------
def render_predict(df: pd.DataFrame, model, stats: dict) -> None:
    st.subheader("🎯 Predict a Student's Result")
    st.markdown("Fill in the student's details below and click **Predict Result**.")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        with st.form("prediction_form"):
            study_hours = st.slider(
                "📚 Daily Study Hours", min_value=0.0, max_value=12.0,
                value=5.0, step=0.5,
                help="Average hours the student studies per day.",
            )
            attendance = st.slider(
                "📅 Attendance (%)", min_value=0, max_value=100, value=75,
                help="Percentage of classes attended.",
            )
            previous_score = st.slider(
                "📝 Previous Exam Score", min_value=0, max_value=100, value=60,
                help="Score obtained in the previous exam (out of 100).",
            )
            submitted = st.form_submit_button(
                "Predict Result", type="primary", width="stretch"
            )

        hours, att, prev = (df[c] for c in ml.FEATURES)
        st.caption(
            f"The model was trained on study hours {_fmt_num(hours.min())}-"
            f"{_fmt_num(hours.max())}, attendance {_fmt_num(att.min())}-"
            f"{_fmt_num(att.max())}% and previous scores {_fmt_num(prev.min())}-"
            f"{_fmt_num(prev.max())}. Values outside these ranges are "
            "extrapolations and less reliable."
        )

    with right:
        if not submitted:
            st.info("👈 Adjust the sliders and click **Predict Result** to see the prediction.")
            return

        try:
            label, prob = ml.predict(study_hours, attendance, previous_score, model=model)
        except ValueError as exc:
            st.error(f"Input error: {exc}")
            return

        prob_display = _fmt_prob(prob)
        if label == "Pass":
            st.success("## ✅ Likely to PASS")
        else:
            st.error("## ⚠️ Likely to FAIL")

        st.markdown(f"**Pass Probability: {prob_display}**")
        st.progress(_progress_val(prob))
        st.markdown("")

        if label == "Pass":
            st.markdown(
                f"The model estimates a **{prob_display} chance of passing** "
                "for these details, so this student looks on track. Keep it up! 🎉"
            )
        else:
            st.markdown(
                f"The model estimates a **{prob_display} chance of passing**. "
                "There is still time to improve; see the tip below."
            )
            st.markdown(_fail_tip(study_hours, attendance, previous_score, stats))

        st.markdown("")
        st.markdown("**Entered Values vs Dataset Average**")
        st.dataframe(
            pd.DataFrame({
                "Feature": ["Study Hours / day", "Attendance", "Previous Score"],
                "Value": [f"{study_hours}h", f"{attendance}%", str(previous_score)],
                "Dataset Avg": [
                    f"{stats['avg_study_hours']:.1f}h",
                    f"{stats['avg_attendance']:.1f}%",
                    f"{stats['avg_previous_score']:.1f}",
                ],
            }),
            hide_index=True,
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Tab: Model Performance
# ---------------------------------------------------------------------------
def render_performance(model, metrics: dict) -> None:
    st.subheader("📈 Model Performance")

    model_name = metrics["model_name"]
    cv_results = metrics["cv_results"]
    cm = metrics["confusion_matrix"]
    clf_report = metrics["classification_report"]
    n_test = int(np.sum(cm))
    cv_mean = cv_results[model_name]["mean"]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("🏆 Selected Model", model_name)
    mc2.metric("🎯 Test Accuracy", f"{metrics['test_accuracy'] * 100:.1f}%")
    mc2.caption(
        f"Test set = {n_test} students, so one student changes this by "
        f"{100 / n_test:g} points. The {ml.CV_FOLDS}-fold cross-validation mean "
        f"(≈{cv_mean * 100:.0f}%) is the more reliable estimate."
    )
    mc3.metric("🗓️ Trained On", metrics.get("training_date", "-")[:10])

    st.markdown("")

    st.markdown(f"#### Cross-Validation Comparison ({ml.CV_FOLDS}-Fold Stratified)")
    st.dataframe(
        pd.DataFrame([
            {
                "Model": name,
                "Mean Acc": f"{res['mean'] * 100:.2f}%",
                "Std Dev": f"±{res['std'] * 100:.2f}%",
                "Min": f"{res['min'] * 100:.2f}%",
                "Max": f"{res['max'] * 100:.2f}%",
                "Selected": "✅" if name == model_name else "",
            }
            for name, res in cv_results.items()
        ]),
        hide_index=True,
        width="stretch",
    )

    st.markdown("")

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown("#### Confusion Matrix")
        _show(_chart_confusion(cm))
    with col_fi:
        st.markdown("#### Feature Influence")
        fig = _chart_feature_influence(model, metrics["feature_names"])
        if fig is None:
            st.info("Feature influence is not available for this model type.")
        else:
            _show(fig)
            st.caption(
                "Coefficients are for standardised features, so they are directly "
                "comparable. Larger = stronger effect on the result."
            )

    st.markdown("")

    st.markdown("#### Classification Report (Test Set)")
    st.dataframe(
        pd.DataFrame([
            {
                "Class": cls,
                "Precision": f"{clf_report[cls]['precision']:.3f}",
                "Recall": f"{clf_report[cls]['recall']:.3f}",
                "F1-Score": f"{clf_report[cls]['f1-score']:.3f}",
                "Support": int(clf_report[cls]["support"]),
            }
            for cls in ["Fail", "Pass", "macro avg", "weighted avg"]
            if cls in clf_report
        ]),
        hide_index=True,
        width="stretch",
    )

    st.markdown("")

    with st.expander("🤔 Why was this model selected?", expanded=True):
        st.markdown(metrics["justification"])


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
def _render_footer() -> None:
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
        "Student Result Predictor | Built with Python, Scikit-learn and Streamlit | "
        "Dataset: Kaggle 'Student Exam Scores' (200 records; exam score excluded "
        "from features to prevent label leakage)"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _render_header()

    try:
        df = _load_df()
    except Exception as exc:
        st.error(f"❌ Could not load dataset: {exc}")
        st.stop()

    try:
        model, metrics = _load_model_and_metrics()
    except Exception as exc:
        st.error(f"❌ Could not prepare the model: {exc}")
        st.stop()

    stats = metrics["summary_stats"]

    tab_dash, tab_pred, tab_perf = st.tabs(
        ["📊 Dashboard", "🎯 Predict Result", "📈 Model Performance"]
    )
    with tab_dash:
        render_dashboard(df, stats)
    with tab_pred:
        render_predict(df, model, stats)
    with tab_perf:
        render_performance(model, metrics)

    _render_footer()


if __name__ == "__main__":
    main()
