# app.py — Employee Attrition Risk Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Attrition Risk Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────
# LOAD MODELS
# ─────────────────────────────
@st.cache_resource
def load_artifacts():
    models = {
        "Random Forest": joblib.load("random_forest.pkl"),
        "Random Forest (Tuned)": joblib.load("random_forest_tuned.pkl"),
        "Logistic Regression": joblib.load("logistic_regression.pkl"),
        "Decision Tree": joblib.load("decision_tree.pkl")
    }
    scaler = joblib.load("scaler.pkl")
    return models, scaler

models, scaler = load_artifacts()

FEATURE_ORDER = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EmployeeCount", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# ─────────────────────────────
# SESSION HISTORY
# ─────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────
# HEADER
# ─────────────────────────────
st.title("🧭 Employee Attrition Risk Dashboard")

st.caption(
    "Employee Attrition Prediction Using ML — Chaithrali Parab"
)

# ─────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────
st.sidebar.header("👤 Employee Profile")

model_choice = st.sidebar.selectbox(
    "🤖 Select Model",
    list(models.keys())
)

selected_model = models[model_choice]

age = st.sidebar.slider("Age", 18, 60, 30)
job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
num_companies = st.sidebar.slider("Companies Worked", 0, 9, 2)
distance = st.sidebar.slider("Distance From Home", 1, 30, 10)

monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000, 500)
years_company = st.sidebar.slider("Years at Company", 0, 40, 5)
salary_hike = st.sidebar.slider("Salary Hike %", 11, 25, 14)

overtime = st.sidebar.radio("Overtime", ["No", "Yes"])
job_sat = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
env_sat = st.sidebar.slider("Env Satisfaction", 1, 4, 3)
wlb = st.sidebar.slider("Work-Life Balance", 1, 4, 3)

overtime_enc = 1 if overtime == "Yes" else 0

# ─────────────────────────────
# INPUT ROW
# ─────────────────────────────
input_row = {
    "Age": age,
    "BusinessTravel": 1,
    "DailyRate": 800,
    "Department": 2,
    "DistanceFromHome": distance,
    "Education": 3,
    "EducationField": 3,
    "EmployeeCount": 1,
    "EnvironmentSatisfaction": env_sat,
    "Gender": 1,
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": job_level,
    "JobRole": 5,
    "JobSatisfaction": job_sat,
    "MaritalStatus": 1,
    "MonthlyIncome": monthly_income,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": num_companies,
    "OverTime": overtime_enc,
    "PercentSalaryHike": salary_hike,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": max(age - 22, 1),
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": wlb,
    "YearsAtCompany": years_company,
    "YearsInCurrentRole": max(years_company - 2, 0),
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
}

input_df = pd.DataFrame([input_row])[FEATURE_ORDER]

# ─────────────────────────────
# LAYOUT
# ─────────────────────────────
left, mid, right = st.columns([1.2, 1, 1])

# ─────────────────────────────
# PREDICTION
# ─────────────────────────────
with left:
    st.subheader("Risk Assessment")

    predict_clicked = st.button("Run Prediction", type="primary")

    if predict_clicked:
        scaled = scaler.transform(input_df)

        pred = selected_model.predict(scaled)[0]
        proba = selected_model.predict_proba(scaled)[0][1] * 100

        tier = "Low Risk" if proba < 30 else "Moderate Risk" if proba < 60 else "High Risk"

        st.metric("Attrition Probability", f"{proba:.1f}%", delta=tier)
        st.progress(min(int(proba), 100) / 100)

        if pred == 1:
            st.error(f"⚠️ {tier} — likely to leave")
        else:
            st.success(f"✅ {tier} — likely to stay")

        st.session_state.history.append({
            "Age": age,
            "JobLevel": job_level,
            "Income": monthly_income,
            "Probability": proba,
            "Risk": tier
        })

    else:
        st.info("Click Run Prediction")

# ─────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────
with mid:
    st.subheader("Top Features")

    if hasattr(selected_model, "feature_importances_"):
        imp = selected_model.feature_importances_

        df = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "Importance": imp
        }).sort_values("Importance", ascending=True).tail(7)

        fig, ax = plt.subplots()
        ax.barh(df["Feature"], df["Importance"])
        st.pyplot(fig)
    else:
        st.info("Not available for this model")

# ─────────────────────────────
# SNAPSHOT
# ─────────────────────────────
with right:
    st.subheader("Profile")

    st.write({
        "Age": age,
        "Job Level": job_level,
        "Income": monthly_income,
        "Overtime": overtime
    })

# ─────────────────────────────
# HISTORY TAB
# ─────────────────────────────
st.divider()
st.subheader("Prediction History")

if len(st.session_state.history) == 0:
    st.info("No history yet")
else:
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist)
    st.bar_chart(hist["Probability"])

if st.button("Clear History"):
    st.session_state.history = []
    st.rerun()