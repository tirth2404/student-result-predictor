# Submission: Student Result Predictor

| Item | Link |
|---|---|
| **Live application** | https://student-result-predictor-tirth.streamlit.app/ |
| **GitHub repository** | https://github.com/tirth2404/student-result-predictor |
| **Demo video (2 min)** | https://drive.google.com/file/d/1jGB6W1aztAg-mPyQKjPWEziiggWElBL6/view?usp=drive_link |

---

## Submission Explanation

Student Result Predictor is a Streamlit web app that predicts whether a student is likely to **Pass** or **Fail** from their daily study hours, attendance percentage and previous exam score. It is deployed on Streamlit Community Cloud, and the full source code, dataset and documentation are in the GitHub repository.

How each requirement is covered:

| Requirement | Where / how |
|---|---|
| **Dataset (≥ 30 records)** | `data.csv`: 200 students from the Kaggle "Student Exam Scores" dataset with `Student_ID`, `Study_Hours`, `Attendance`, `Previous_Score` and `Result` (Pass if exam score ≥ 35). `data_prep.py` builds it and falls back to a NumPy-generated dataset if the raw file is missing. |
| **Data analysis (Pandas & NumPy)** | `model.py` loads and displays the data, checks for missing values (median imputation if any), calculates average study hours (6.33), attendance (74.8%) and previous score (66.8), and counts passes (91) and fails (109). |
| **Machine learning model** | `model.py` separates features and target, uses a stratified 80/20 train/test split, compares Logistic Regression, Decision Tree and Random Forest with 5-fold cross-validation, and evaluates the chosen model with an accuracy score, confusion matrix and classification report. |
| **Streamlit app** | `app.py` has three tabs: **Dashboard** (5 summary cards and 3 charts), **Predict Result** (sliders and a *Predict Result* button showing Pass/Fail with probability) and **Model Performance** (model name, accuracy, confusion matrix, classification report and model comparison). |
| **Code structure** | Separate files for data preparation (`data_prep.py`), ML logic (`model.py`) and UI (`app.py`), plus `tests/`, `models/`, `assets/`, `requirements.txt`, `README.md` and `.gitignore`. |
| **Deployment** | Streamlit Community Cloud. If the saved model is missing or was built with a different scikit-learn version, the app retrains it automatically on start-up, so it never crashes on a cold start. |
| **Quality** | 22 automated tests (pytest) cover the data, predictions, model selection, auto-retraining and a headless run of the app. |

---

## Technical Explanation

I built a model that predicts whether a student will pass or fail from three inputs: daily study hours, attendance percentage and previous exam score. I used the Kaggle "Student Exam Scores" dataset of 200 students and labelled each student as Pass if their exam score was 35 or more. I removed the exam score from the inputs so the model could not simply copy the answer.

I split the data 80/20 (160 students for training, 40 for testing) and compared three models with 5-fold cross-validation: Logistic Regression (89.4%), Random Forest (84.4%) and Decision Tree (76.9%). I chose Logistic Regression because it scored highest. It is also simple, easy to explain and gives reliable probabilities, which suits a small dataset with only three numeric features.

To make a prediction, the app first scales the three inputs using the averages learned during training. The model then multiplies each value by a learned weight, adds them up, and converts the total into a probability of passing with the logistic function. If the probability is 50% or more, it shows "Likely to Pass"; otherwise it shows "Likely to Fail". Study hours have the strongest effect.

The model scored 92.5% on the 40 test students. Cross-validation (about 89%) is the more reliable estimate.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Selected model | Logistic Regression (StandardScaler + LogisticRegression) |
| 5-fold CV accuracy | 89.4% (± 4.2%) |
| Test accuracy | 92.5% (37 of 40 correct) |
| Confusion matrix (test) | Fail: 20 correct, 2 wrong · Pass: 17 correct, 1 wrong |
| Strongest feature | Study hours |

See [README.md](README.md) for setup instructions, the full model explanation and known limitations.
