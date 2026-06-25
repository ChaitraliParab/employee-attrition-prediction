# app.py — Employee Attrition Risk Dashboard
# Author: Chaithrali Chandrashekhar Parab (MMS, Aditya Institute of Management Studies and Research)
# Internship Project: Employee Attrition Prediction Using Machine Learning
# DCyber TechLab Pvt Ltd

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ───────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attrition Risk Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────
# LOAD MODEL + SCALER (cached so it only loads once)
# ───────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────
# SESSION STATE — keeps a running log of predictions made
# ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ───────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────
st.title("🧭 Employee Attrition Risk Dashboard")
st.caption(
    "Employee Attrition Prediction Using Machine Learning  ·  "
    "Built by **Chaithrali Chandrashekhar Parab**, MMS, "
    "Aditya Institute of Management Studies and Research"
)

# ── Model metric strip (always visible at the top) ──────────
hm1, hm2, hm3, hm4, hm5 = st.columns(5)
hm1.metric("Model", "Random Forest")
hm2.metric("Trees", "200")
hm3.metric("Accuracy", "84.0%")
hm4.metric("F1-Score", "33.8%")
hm5.metric("Recall", "25.5%")

st.divider()

# ───────────────────────────────────────────────────────────
# TABBED LAYOUT
# ───────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_history, tab_about = st.tabs(
    ["🔮 Risk Predictor", "📊 Workforce Insights", "🕒 Prediction Log", "ℹ️ About"]
)

# =============================================================
# TAB 1 — RISK PREDICTOR
# =============================================================
with tab_predict:

    st.sidebar.header("👤 Employee Profile")
model_choice = st.sidebar.selectbox(
    "🤖 Select Model",
    list(models.keys())
)
st.sidebar.caption("Adjust the sliders to describe the employee")

with st.sidebar.expander("Personal & Role", expanded=True):
        age = st.slider("Age", 18, 60, 30)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
        num_companies = st.slider("Companies Worked Previously", 0, 9, 2)
        distance = st.slider("Distance from Home (km)", 1, 30, 10)
    
with st.sidebar.expander("Compensation & Tenure", expanded=True):
        monthly_income = st.number_input(
            "Monthly Income (₹)", min_value=1000, max_value=20000, value=5000, step=500
        )
        years_company = st.slider("Years at Company", 0, 40, 5)
        salary_hike = st.slider("Last Salary Hike (%)", 11, 25, 14)
    
with st.sidebar.expander("Work Environment", expanded=True):
        overtime = st.radio("Works Overtime?", ["No", "Yes"], horizontal=True)
        job_sat = st.slider("Job Satisfaction", 1, 4, 3, help="1 = Low, 4 = Very High")
        env_sat = st.slider("Environment Satisfaction", 1, 4, 3, help="1 = Low, 4 = Very High")
        wlb = st.slider("Work-Life Balance", 1, 4, 3, help="1 = Bad, 4 = Excellent")

overtime_enc = 1 if overtime == "Yes" else 0

input_row = {
        "Age": age, "BusinessTravel": 1, "DailyRate": 800,
        "Department": 2, "DistanceFromHome": distance,
        "Education": 3, "EducationField": 3,
        "EmployeeCount": 1, "EnvironmentSatisfaction": env_sat,
        "Gender": 1, "HourlyRate": 65, "JobInvolvement": 3,
        "JobLevel": job_level, "JobRole": 5,
        "JobSatisfaction": job_sat, "MaritalStatus": 1,
        "MonthlyIncome": monthly_income, "MonthlyRate": 14000,
        "NumCompaniesWorked": num_companies, "OverTime": overtime_enc,
        "PercentSalaryHike": salary_hike, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
        "TotalWorkingYears": max(age - 22, 1), "TrainingTimesLastYear": 3,
        "WorkLifeBalance": wlb, "YearsAtCompany": years_company,
        "YearsInCurrentRole": max(years_company - 2, 0),
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
}
input_df = pd.DataFrame([input_row])[FEATURE_ORDER]

left, mid, right = st.columns([1.1, 1, 1])

with left:
        st.subheader("Risk Assessment")
        predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

if predict_clicked:
    scaled = scaler.transform(input_df)
    pred = selected_model.predict(scaled)[0]
    proba = selected_model.predict_proba(scaled)[0][1] * 100

    # Risk tier
    if proba < 30:
        tier = "Low Risk"
    elif proba < 60:
        tier = "Moderate Risk"
    else:
        tier = "High Risk"

    st.metric("Attrition Probability", f"{proba:.1f}%", delta=tier)
    st.progress(min(int(proba), 100) / 100)

    if pred == 1:
        st.error(
            f"⚠️ **{tier}** — model flags this employee as likely to leave "
            f"({proba:.1f}% probability)."
        )
        st.markdown(
            "**Suggested next steps:**\n"
            "- Schedule a 1:1 check-in within 2-4 weeks\n"
            "- Review workload and overtime\n"
            "- Discuss career growth opportunities"
        )
    else:
        st.success(
            f"✅ **{tier}** — model predicts this employee is likely to stay "
            f"({proba:.1f}% probability of leaving)."
        )
        st.markdown("No immediate action required.")

    # log entry
    log_entry = input_row.copy()
    log_entry["Predicted Probability (%)"] = round(proba, 1)
    log_entry["Risk Tier"] = tier
    st.session_state.history.append(log_entry)

else:
    st.info("Set the employee's details in the sidebar, then click **Run Prediction**.")

# 👇 THESE MUST BE OUTSIDE if/else (IMPORTANT)
with mid:
    st.subheader("Top Attrition Predictors")
    st.caption("Feature importance from the Random Forest model")

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(7)

    fig_imp, ax_imp = plt.subplots(figsize=(4, 3.2))
    ax_imp.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#2E74B5")
    ax_imp.set_xlabel("Importance")
    for i, (val, name) in enumerate(zip(imp_df["Importance"][::-1], imp_df["Feature"][::-1])):
        ax_imp.text(val, i, f" {val*100:.1f}%", va="center", fontsize=8)
    st.pyplot(fig_imp, use_container_width=True)

with right:
        st.subheader("Profile Snapshot")
        snapshot = pd.DataFrame({
            "Attribute": [
                "Age", "Job Level", "Monthly Income", "Years at Company",
                "Overtime", "Job Satisfaction", "Environment Satisfaction",
                "Work-Life Balance", "Distance from Home", "Companies Worked",
            ],
            "Value": [
                age, job_level, f"₹{monthly_income:,}", years_company,
                overtime, f"{job_sat}/4", f"{env_sat}/4",
                f"{wlb}/4", f"{distance} km", num_companies,
            ]
        })
        st.dataframe(snapshot, hide_index=True, use_container_width=True)

        # Quick bar comparison of the 3 satisfaction scores
        st.caption("Satisfaction profile (1 = Low, 4 = High)")
        sat_df = pd.DataFrame({
            "Dimension": ["Job Satisfaction", "Environment", "Work-Life Balance"],
            "Score": [job_sat, env_sat, wlb]
        })
        fig, ax = plt.subplots(figsize=(4, 2.2))
        bars = ax.barh(sat_df["Dimension"], sat_df["Score"], color="#2E74B5")
        ax.set_xlim(0, 4)
        ax.set_xlabel("Score")
for bar, val in zip(bars, sat_df["Score"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, str(val), va="center")
        st.pyplot(fig, use_container_width=True)


# =============================================================
# TAB 2 — WORKFORCE INSIGHTS (dataset-level analytics)
# =============================================================
with tab_insights:
    st.subheader("Dataset-Level Attrition Patterns")
    st.caption("Charts generated from the IBM HR Analytics dataset used to train this model.")

    try:
        raw = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

        k1, k2, k3, k4 = st.columns(4)
        total = len(raw)
        left_count = (raw["Attrition"] == "Yes").sum()
        k1.metric("Total Employees", f"{total:,}")
        k2.metric("Attrition Count", f"{left_count}")
        k3.metric("Attrition Rate", f"{left_count/total*100:.1f}%")
        k4.metric("Avg. Monthly Income", f"₹{raw['MonthlyIncome'].mean():,.0f}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            dept = (raw.groupby("Department")["Attrition"]
                       .value_counts(normalize=True).unstack() * 100)
            dept["Yes"].sort_values().plot(kind="barh", ax=ax, color="#C0392B")
            ax.set_title("Attrition Rate by Department (%)", fontweight="bold")
            ax.set_xlabel("Attrition Rate (%)")
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(x="OverTime", hue="Attrition", data=raw,
                           palette=["#2E74B5", "#C0392B"], ax=ax2)
            ax2.set_title("Overtime vs Attrition", fontweight="bold")
            st.pyplot(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Attrition", y="MonthlyIncome", data=raw,
                        palette=["#2E74B5", "#C0392B"], ax=ax3)
            ax3.set_title("Monthly Income vs Attrition", fontweight="bold")
            st.pyplot(fig3, use_container_width=True)
        with c4:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            tenure = (raw.groupby("YearsAtCompany")["Attrition"]
                         .value_counts(normalize=True).unstack() * 100)
            tenure["Yes"].plot(kind="bar", ax=ax4, color="#C0392B")
            ax4.set_title("Attrition Rate (%) by Years at Company", fontweight="bold")
            ax4.set_xlabel("Years at Company")
            ax4.set_ylabel("Attrition Rate (%)")
            st.pyplot(fig4, use_container_width=True)

except FileNotFoundError:
        st.warning(
            "Dataset file `WA_Fn-UseC_-HR-Employee-Attrition.csv` not found in this folder. "
            "Add it alongside app.py to enable the workforce insight charts."
        )


# =============================================================
# TAB 3 — PREDICTION LOG (session history)
# =============================================================
with tab_history:
    st.subheader("Predictions Made This Session")
if len(st.session_state.history) == 0:
        st.info("No predictions yet. Run a prediction from the **Risk Predictor** tab to see it logged here.")
else:
        hist_df = pd.DataFrame(st.session_state.history)
        display_cols = [
            "Age", "JobLevel", "MonthlyIncome", "OverTime", "YearsAtCompany",
            "JobSatisfaction", "WorkLifeBalance",
            "Predicted Probability (%)", "Risk Tier"
        ]
        st.dataframe(hist_df[display_cols], use_container_width=True, hide_index=True)

        st.bar_chart(hist_df["Predicted Probability (%)"])

if st.button("Clear Log"):
            st.session_state.history = []
            st.rerun()


# =============================================================
# TAB 4 — ABOUT
# =============================================================
with tab_about:
    st.subheader("About This Dashboard")
    st.markdown(
        """
This dashboard is the final deliverable of the **Employee Attrition Prediction Using
Machine Learning** internship project, built on the IBM HR Analytics Employee
Attrition dataset.

**Pipeline summary**
- Data cleaning & encoding (Label Encoding, target conversion to 0/1)
- Feature scaling with `StandardScaler`
- Class balancing using `SMOTE` (84/16 → 50/50 on the training set)
- Models trained: Logistic Regression, Decision Tree, Random Forest (+ tuned variant)
- This dashboard uses the **Random Forest** model with its corresponding `scaler.pkl`

**How to use**
1. Go to the **Risk Predictor** tab.
2. Fill in the employee's profile in the sidebar.
3. Click **Run Prediction** to see the attrition risk and suggested actions.
4. Check the **Workforce Insights** tab for overall dataset patterns.
5. Review past predictions in the **Prediction Log** tab.
        """
    )
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "Random Forest")
    m2.metric("Accuracy", "84.0%")
    m3.metric("Precision", "50.0%")
    m4.metric("Recall", "25.5%")

st.divider()
st.caption(
    "Project: Employee Attrition Prediction Using Machine Learning  |  "
    "Chaithrali Chandrashekhar Parab — MMS, Aditya Institute of Management Studies and Research  |  "
    "DCyber TechLab Pvt Ltd, Summer Internship 2026"
)
