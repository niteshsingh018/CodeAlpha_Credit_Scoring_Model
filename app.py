from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Page setup and paths (paths are anchored to this file)
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "credit_risk_features.csv"
CLEAN_FILE = ROOT / "data" / "credit_risk_cleaned.csv"
MODEL_FILE = ROOT / "models" / "best_credit_model.pkl"
EVALUATION_FILE = ROOT / "results" / "model_evaluation.csv"
COMPARISON_FILE = ROOT / "results" / "model_comparison.csv"
BEST_INFO_FILE = ROOT / "results" / "best_model_info.csv"
FEATURES_FILE = ROOT / "data" / "X_train.csv"

st.set_page_config(
    page_title="CrediScope | Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #10213a; }
    [data-testid="stSidebar"] * { color: #f3f6fc !important; }
    .hero { padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
            background: linear-gradient(120deg, #10213a 0%, #175d78 100%); margin-bottom: 1rem; }
    .hero h1 { color: white; margin: 0; font-size: 2.1rem; }
    .hero p { color: #d8e8f3; margin: .45rem 0 0 0; font-size: 1rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e3e9f2;
      padding: 1rem; border-radius: 14px; box-shadow: 0 3px 14px rgba(18,40,70,.04); }
    div[data-testid="stForm"] { background: white; padding: 1rem 1.2rem; border-radius: 14px;
      border: 1px solid #e3e9f2; }
    .small-note { color: #66758a; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Safe cached loaders
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def metric_value(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def fmt_pct(value):
    return "—" if value is None or not np.isfinite(value) else f"{value * 100:.2f}%"


def make_features(values, expected_features):
    """Create the exact 31-column feature row expected by the trained model."""
    income = max(float(values["annual_income"]), 1.0)
    debt = float(values["current_debt"])
    loan = float(values["loan_amount"])
    age = max(float(values["age"]), 1.0)
    history = float(values["credit_history_years"])
    employment = float(values["years_employed"])
    savings = float(values["savings_assets"])

    row = {
        "age": age,
        "years_employed": employment,
        "annual_income": income,
        "credit_score": float(values["credit_score"]),
        "credit_history_years": history,
        "savings_assets": savings,
        "current_debt": debt,
        "delinquencies_last_2yrs": float(values["delinquencies_last_2yrs"]),
        "derogatory_marks": float(values["derogatory_marks"]),
        "loan_amount": loan,
        "interest_rate": float(values["interest_rate"]),
        "debt_to_income_ratio": float(values["debt_to_income_ratio"]),
        "loan_to_income_ratio": loan / income,
        "payment_to_income_ratio": float(values["payment_to_income_ratio"]),
        "income_to_loan_ratio": income / (loan + 1),
        "income_to_debt_ratio": income / (debt + 1),
        "savings_to_debt_ratio": savings / (debt + 1),
        "credit_history_to_age_ratio": history / (age + 1),
        "employment_to_age_ratio": employment / (age + 1),
        "total_debt_burden": debt + loan,
        "financial_stress_ratio": (debt + loan) / (income + 1),
        "occupation_status_Self-Employed": int(values["occupation_status"] == "Self-Employed"),
        "occupation_status_Student": int(values["occupation_status"] == "Student"),
        "defaults_on_file_1": int(values["defaults_on_file"] == "Yes"),
        "product_type_Line of Credit": int(values["product_type"] == "Line of Credit"),
        "product_type_Personal Loan": int(values["product_type"] == "Personal Loan"),
        "loan_intent_Debt Consolidation": int(values["loan_intent"] == "Debt Consolidation"),
        "loan_intent_Education": int(values["loan_intent"] == "Education"),
        "loan_intent_Home Improvement": int(values["loan_intent"] == "Home Improvement"),
        "loan_intent_Medical": int(values["loan_intent"] == "Medical"),
        "loan_intent_Personal": int(values["loan_intent"] == "Personal"),
    }
    return pd.DataFrame([{name: row.get(name, 0) for name in expected_features}], columns=expected_features)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------
st.sidebar.markdown("# 💳 CrediScope")
st.sidebar.caption("CREDIT RISK INTELLIGENCE")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Predict Credit Risk", "Model Performance", "Dataset Insights", "About Project"],
    label_visibility="visible",
)
st.sidebar.divider()
st.sidebar.caption("Academic machine-learning demonstration")

# ------------------------------------------------------------
# Shared data/model state
# ------------------------------------------------------------
data = read_csv(str(DATA_FILE))
clean_data = read_csv(str(CLEAN_FILE))
evaluation = read_csv(str(EVALUATION_FILE))
comparison = read_csv(str(COMPARISON_FILE))
model = load_model(str(MODEL_FILE))
feature_frame = read_csv(str(FEATURES_FILE))
expected_features = list(feature_frame.columns) if feature_frame is not None else list(getattr(model, "feature_names_in_", []))

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if page == "Overview":
    hero("Credit Risk Intelligence", "A machine-learning interface for exploring applicant data, comparing classifiers, and testing credit-risk predictions.")

    if data is None:
        st.error("Feature dataset is missing. Expected: data/credit_risk_features.csv")
    else:
        n_records = len(data)
        n_features = len(expected_features) if expected_features else max(len(data.columns) - 1, 0)
        positive_rate = data["loan_status"].mean() if "loan_status" in data else np.nan
        best_name = "Not available"
        best_auc = None
        if comparison is not None and not comparison.empty and "Model" in comparison:
            best_name = str(comparison.iloc[0]["Model"])
            best_auc = metric_value(comparison.iloc[0], "ROC-AUC")
        elif model is not None:
            best_name = "Saved classifier"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset records", f"{n_records:,}")
        c2.metric("Model input features", f"{n_features}")
        c3.metric("Positive target class", fmt_pct(positive_rate))
        c4.metric("Top model ROC-AUC", fmt_pct(best_auc), help="Calculated from the saved evaluation results when available.")

        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Loan-status distribution")
            if "loan_status" in data:
                counts = data["loan_status"].value_counts().sort_index()
                counts.index = [f"Class {int(i)}" for i in counts.index]
                st.bar_chart(counts, color="#1d7894")
                st.caption("Class labels follow the dataset. Class 1 is treated as the positive outcome by the current project code.")
        with right:
            st.subheader("Model performance at a glance")
            if evaluation is not None and not evaluation.empty:
                metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"] if c in evaluation.columns]
                if "Model" in evaluation.columns and metric_cols:
                    chart = evaluation.set_index("Model")[metric_cols]
                    st.bar_chart(chart, y=metric_cols, color=["#1d7894", "#57a6a1", "#f2b36d", "#6f83bd", "#9a82c4"][:len(metric_cols)])
            else:
                st.info("Evaluation CSV not found. The Model Performance page explains which result files are expected.")

        st.subheader("Applicant dataset preview")
        st.dataframe(data.drop(columns=["loan_status"], errors="ignore").head(8), use_container_width=True, hide_index=True)
        if model is not None:
            st.success(f"Saved prediction model loaded successfully · {type(model).__name__}")
        else:
            st.error("Saved model is missing or could not be loaded. Check models/best_credit_model.pkl and the package versions.")

# ------------------------------------------------------------
# Prediction page
# ------------------------------------------------------------
elif page == "Predict Credit Risk":
    hero("Applicant Assessment", "Enter the applicant's financial profile to receive a model prediction. This is a classroom demonstration, not a lending decision.")
    if model is None or not expected_features:
        st.error("Prediction is unavailable because the saved model or its expected feature list could not be loaded.")
    else:
        with st.form("credit_prediction_form"):
            st.subheader("Personal and employment profile")
            a, b, c = st.columns(3)
            with a:
                age = st.number_input("Age (years)", min_value=18, max_value=100, value=32, step=1)
                occupation = st.selectbox("Occupation status", ["Employed", "Self-Employed", "Student"])
            with b:
                years_employed = st.number_input("Years employed", min_value=0.0, max_value=70.0, value=5.0, step=0.5)
                annual_income = st.number_input("Annual income", min_value=1.0, value=60000.0, step=1000.0)
            with c:
                credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=680, step=1)
                history_years = st.number_input("Credit history (years)", min_value=0.0, max_value=80.0, value=6.0, step=0.5)

            st.subheader("Financial obligations")
            a, b, c = st.columns(3)
            with a:
                savings = st.number_input("Savings / assets", min_value=0.0, value=5000.0, step=500.0)
                current_debt = st.number_input("Current debt", min_value=0.0, value=10000.0, step=500.0)
            with b:
                loan_amount = st.number_input("Requested loan amount", min_value=0.0, value=15000.0, step=500.0)
                interest_rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.25)
            with c:
                debt_ratio_default = min(current_debt / max(annual_income, 1), 10.0)
                dti = st.number_input("Debt-to-income ratio", min_value=0.0, max_value=10.0, value=float(debt_ratio_default), step=0.01, help="Current debt divided by annual income. Adjust if your dataset defines this ratio differently.")
                payment_ratio = st.number_input("Payment-to-income ratio", min_value=0.0, max_value=10.0, value=0.10, step=0.01, help="Enter the payment-to-income ratio using the same definition as your dataset.")

            st.subheader("Credit history and loan details")
            a, b, c = st.columns(3)
            with a:
                delinquencies = st.number_input("Delinquencies in last 2 years", min_value=0, max_value=50, value=0, step=1)
                derogatory = st.number_input("Derogatory marks", min_value=0, max_value=50, value=0, step=1)
            with b:
                default_file = st.selectbox("Previous default on file?", ["No", "Yes"])
                product_type = st.selectbox("Product type", ["Credit Card", "Line of Credit", "Personal Loan"])
            with c:
                loan_intent = st.selectbox("Loan purpose", ["Business", "Debt Consolidation", "Education", "Home Improvement", "Medical", "Personal"])
                st.caption("Loan-to-income and engineered ratios are calculated automatically.")

            submitted = st.form_submit_button("Run Credit Risk Prediction", type="primary", use_container_width=True)

        if submitted:
            if years_employed > age:
                st.error("Years employed cannot be greater than age. Please correct the value and try again.")
            elif history_years > age:
                st.error("Credit history cannot exceed age. Please correct the value and try again.")
            else:
                inputs = {
                    "age": age,
                    "years_employed": years_employed,
                    "annual_income": annual_income,
                    "credit_score": credit_score,
                    "credit_history_years": history_years,
                    "savings_assets": savings,
                    "current_debt": current_debt,
                    "delinquencies_last_2yrs": delinquencies,
                    "derogatory_marks": derogatory,
                    "loan_amount": loan_amount,
                    "interest_rate": interest_rate,
                    "debt_to_income_ratio": dti,
                    "payment_to_income_ratio": payment_ratio,
                    "occupation_status": occupation,
                    "defaults_on_file": default_file,
                    "product_type": product_type,
                    "loan_intent": loan_intent,
                }
                try:
                    x_input = make_features(inputs, expected_features)
                    prediction = int(model.predict(x_input)[0])
                    probabilities = model.predict_proba(x_input)[0] if hasattr(model, "predict_proba") else None
                    positive_probability = None
                    if probabilities is not None:
                        classes = list(getattr(model, "classes_", [0, 1]))
                        if 1 in classes:
                            positive_probability = float(probabilities[classes.index(1)])

                    st.divider()
                    st.subheader("Prediction result")
                    if prediction == 1:
                        st.success("Predicted outcome: Class 1 — positive outcome in the training dataset")
                    else:
                        st.warning("Predicted outcome: Class 0 — negative outcome in the training dataset")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Predicted class", str(prediction))
                    r2.metric("Class 1 model probability", fmt_pct(positive_probability))
                    r3.metric("Requested loan", f"{loan_amount:,.0f}")
                    st.caption("The class meaning should be confirmed against the original dataset documentation. Probability is a model estimate, not a guarantee or a calibrated lending decision.")
                    with st.expander("View generated model features"):
                        st.dataframe(x_input, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.info("Check that the saved model and X_train.csv were created from the same feature-engineering pipeline.")

# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------
elif page == "Model Performance":
    hero("Model Evaluation Lab", "Compare the classifiers using the evaluation files generated by your training workflow.")
    if evaluation is not None and not evaluation.empty:
        st.subheader("Evaluation metrics")
        st.dataframe(evaluation.style.format({c: "{:.4f}" for c in evaluation.columns if c != "Model"}), use_container_width=True, hide_index=True)
        metrics = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"] if c in evaluation.columns]
        if "Model" in evaluation.columns and metrics:
            st.subheader("Metric comparison")
            st.bar_chart(evaluation.set_index("Model")[metrics])
    else:
        st.warning("results/model_evaluation.csv is missing or empty. Recreate it from all three prediction files before submission.")

    if comparison is not None and not comparison.empty:
        st.subheader("Model ranking")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if "Model" in comparison and "ROC-AUC" in comparison:
            best_row = comparison.sort_values("ROC-AUC", ascending=False).iloc[0]
            st.info(f"Highest ROC-AUC in the stored comparison: **{best_row['Model']}** ({float(best_row['ROC-AUC']):.4f}). The model file should match this selection.")

    st.subheader("Confusion matrix images")
    img_cols = st.columns(3)
    for col, title, filename in zip(img_cols, ["Logistic Regression", "Decision Tree", "Random Forest"], ["logistic_regression_confusion_matrix.png", "decision_tree_confusion_matrix.png", "random_forest_confusion_matrix.png"]):
        with col:
            image_path = ROOT / "results" / filename
            st.markdown(f"**{title}**")
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.caption("Image not available yet.")

# ------------------------------------------------------------
# Dataset insights
# ------------------------------------------------------------
elif page == "Dataset Insights":
    hero("Dataset Explorer", "Review the cleaned data, target balance, and available feature columns.")
    source = clean_data if clean_data is not None else data
    if source is None:
        st.error("No dataset was found under the data/ folder.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(source):,}")
        c2.metric("Columns", f"{len(source.columns):,}")
        c3.metric("Missing cells", f"{int(source.isna().sum().sum()):,}")
        if "loan_status" in source:
            st.subheader("Target class balance")
            st.bar_chart(source["loan_status"].value_counts().sort_index(), color="#1d7894")
        st.subheader("Column summary")
        summary = pd.DataFrame({"Column": source.columns, "Data type": [str(t) for t in source.dtypes], "Missing values": [int(source[c].isna().sum()) for c in source.columns], "Unique values": [int(source[c].nunique(dropna=True)) for c in source.columns]})
        st.dataframe(summary, use_container_width=True, hide_index=True)
        with st.expander("Preview dataset rows"):
            st.dataframe(source.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download dataset preview (CSV)", source.head(1000).to_csv(index=False).encode("utf-8"), "credit_dataset_preview.csv", "text/csv")

# ------------------------------------------------------------
# About
# ------------------------------------------------------------
else:
    hero("About CrediScope", "Creditworthiness prediction using supervised classification.")
    st.markdown("""
    ### Project objective
    Estimate the target class associated with an applicant's financial profile using historical financial data.

    ### Algorithms
    - **Logistic Regression** — linear classification baseline.
    - **Decision Tree** — rule-based classification model.
    - **Random Forest** — ensemble of decision trees.

    ### Workflow
    Dataset collection → data cleaning → exploratory analysis → feature engineering → train/test split → model training → evaluation → model comparison → saved model → interactive prediction.

    ### Important limitations
    - This is an academic demonstration, not a real credit decision service.
    - Confirm the meaning of target labels from the original dataset source before describing class 0 or class 1 as approved/rejected.
    - Do not use this prototype to approve or deny real applicants. Production credit decisions require validation, fairness testing, privacy/security controls, explainability, and compliance review.
    """)

st.sidebar.divider()
st.sidebar.caption("CrediScope · Machine Learning Project")
