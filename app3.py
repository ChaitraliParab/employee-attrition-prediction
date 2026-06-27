# app.py — Employee Attrition Risk Dashboard
# Author: Chaithrali Chandrashekhar Parab
#         MMS, Aditya Institute of Management Studies and Research
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
# MODEL REGISTRY — all four trained models with their stats
# ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression": {
        "file": "logistic_regression.pkl",
        "accuracy": "76.5%", "precision": "38.3%",
        "recall": "76.6%", "f1": "51.1%", "auc": "0.804",
        "has_importance": False,
        "note": "Best F1 & Recall — recommended for catching at-risk employees",
        "color": "#5C6BC0",
    },
    "Decision Tree": {
        "file": "decision_tree.pkl",
        "accuracy": "78.9%", "precision": "35.3%",
        "recall": "38.3%", "f1": "36.7%", "auc": "0.658",
        "has_importance": True,
        "note": "Interpretable tree structure — moderate overall performance",
        "color": "#26A69A",
    },
    "Random Forest": {
        "file": "random_forest.pkl",
        "accuracy": "84.0%", "precision": "50.0%",
        "recall": "25.5%", "f1": "33.8%", "auc": "0.800",
        "has_importance": True,
        "note": "Highest accuracy — best at identifying employees who stay",
        "color": "#EF5350",
    },
    "Random Forest (Tuned)": {
        "file": "random_forest_tuned.pkl",
        "accuracy": "84.0%", "precision": "50.0%",
        "recall": "25.5%", "f1": "33.8%", "auc": "0.800",
        "has_importance": True,
        "note": "GridSearchCV-tuned variant — identical results to base RF on this dataset",
        "color": "#AB47BC",
    },
}

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
# LOAD ALL MODELS + SCALER (cached)
# ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    scaler = joblib.load("scaler.pkl")
    models = {}
    for name, info in MODEL_REGISTRY.items():
        try:
            models[name] = joblib.load(info["file"])
        except FileNotFoundError:
            models[name] = None
    return scaler, models

scaler, all_models = load_all()

# ───────────────────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ───────────────────────────────────────────────────────────
# SIDEBAR — model selector + employee profile
# ───────────────────────────────────────────────────────────
st.sidebar.title("🧭 Dashboard Controls")

st.sidebar.markdown("### 🤖 Choose Model")
selected_model_name = st.sidebar.selectbox(
    "Active prediction model",
    list(MODEL_REGISTRY.keys()),
    index=0,
    help="Switch between all four trained models to compare predictions."
)
selected_info = MODEL_REGISTRY[selected_model_name]
active_model = all_models[selected_model_name]

st.sidebar.caption(f"💡 {selected_info['note']}")
st.sidebar.divider()

st.sidebar.markdown("### 👤 Employee Profile")
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

# ───────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────
st.title("🧭 Employee Attrition Risk Dashboard")
st.caption(
    "Employee Attrition Prediction Using Machine Learning  ·  "
    "Built by **Chaithrali Chandrashekhar Parab**, MMS, "
    "Aditya Institute of Management Studies and Research  ·  DCyber TechLab Pvt Ltd"
)

# ── Dynamic model metric strip (updates with model choice) ──
h1, h2, h3, h4, h5, h6 = st.columns(6)
h1.metric("Active Model", selected_model_name.replace(" (Tuned)", " ✨"))
h2.metric("Accuracy", selected_info["accuracy"])
h3.metric("Precision", selected_info["precision"])
h4.metric("Recall", selected_info["recall"])
h5.metric("F1-Score", selected_info["f1"])
h6.metric("AUC-ROC", selected_info["auc"])

st.divider()

# ───────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────
tab_predict, tab_compare, tab_insights, tab_history, tab_about = st.tabs(
    ["🔮 Risk Predictor", "📈 Model Comparison", "📊 Workforce Insights", "🕒 Prediction Log", "ℹ️ About"]
)

# =============================================================
# TAB 1 — RISK PREDICTOR
# =============================================================
with tab_predict:

    if active_model is None:
        st.error(f"`{selected_info['file']}` not found. Make sure all model .pkl files are in the same folder as app.py.")
    else:
        left, mid, right = st.columns([1.1, 1, 1])

        with left:
            st.subheader(f"Risk Assessment — {selected_model_name}")
            predict_clicked = st.button("▶ Run Prediction", type="primary", use_container_width=True)

            if predict_clicked:
                scaled = scaler.transform(input_df)
                pred = active_model.predict(scaled)[0]
                proba = active_model.predict_proba(scaled)[0][1] * 100

                if proba < 30:
                    tier, tier_color = "Low Risk", "green"
                elif proba < 60:
                    tier, tier_color = "Moderate Risk", "orange"
                else:
                    tier, tier_color = "High Risk", "red"

                st.metric("Attrition Probability", f"{proba:.1f}%", delta=tier)
                st.progress(min(int(proba), 100) / 100)

                if pred == 1:
                    st.error(f"⚠️ **{tier}** — {selected_model_name} flags this employee as likely to leave ({proba:.1f}% probability).")
                    st.markdown(
                        "**Suggested HR actions:**\n"
                        "- Schedule a 1:1 check-in within the next 2–4 weeks\n"
                        "- Review workload and overtime patterns\n"
                        "- Discuss career growth and recognition options\n"
                        "- Consider compensation review if income is below market"
                    )
                else:
                    st.success(f"✅ **{tier}** — {selected_model_name} predicts this employee is likely to stay ({proba:.1f}% probability of leaving).")
                    st.markdown("No immediate intervention needed — continue regular check-ins.")

                log_entry = input_row.copy()
                log_entry["Model Used"] = selected_model_name
                log_entry["Predicted Probability (%)"] = round(proba, 1)
                log_entry["Risk Tier"] = tier
                st.session_state.history.append(log_entry)
            else:
                st.info("Configure the employee profile in the sidebar, choose a model, then click **▶ Run Prediction**.")

        with mid:
            if selected_info["has_importance"] and hasattr(active_model, "feature_importances_"):
                st.subheader("Top Attrition Predictors")
                st.caption(f"Feature importance — {selected_model_name}")
                imp_df = pd.DataFrame({
                    "Feature": FEATURE_ORDER,
                    "Importance": active_model.feature_importances_
                }).sort_values("Importance", ascending=False).head(8)

                fig_imp, ax_imp = plt.subplots(figsize=(4, 3.5))
                colors_imp = [selected_info["color"]] * len(imp_df)
                ax_imp.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color=colors_imp)
                ax_imp.set_xlabel("Importance Score")
                for i, (val, _) in enumerate(zip(imp_df["Importance"][::-1], imp_df["Feature"][::-1])):
                    ax_imp.text(val, i, f" {val*100:.1f}%", va="center", fontsize=8)
                ax_imp.set_title(f"Top 8 Features", fontsize=10, fontweight="bold")
                st.pyplot(fig_imp, use_container_width=True)
            else:
                st.subheader("Model Coefficients")
                st.caption("Logistic Regression — top features by coefficient magnitude")
                if hasattr(active_model, "coef_"):
                    coef_df = pd.DataFrame({
                        "Feature": FEATURE_ORDER,
                        "Coefficient": np.abs(active_model.coef_[0])
                    }).sort_values("Coefficient", ascending=False).head(8)
                    fig_c, ax_c = plt.subplots(figsize=(4, 3.5))
                    ax_c.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1], color=selected_info["color"])
                    ax_c.set_xlabel("|Coefficient|")
                    ax_c.set_title("Top 8 Features", fontsize=10, fontweight="bold")
                    st.pyplot(fig_c, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")

        with right:
            st.subheader("Profile Snapshot")
            snapshot = pd.DataFrame({
                "Attribute": [
                    "Age", "Job Level", "Monthly Income", "Years at Company",
                    "Overtime", "Job Satisfaction", "Env. Satisfaction",
                    "Work-Life Balance", "Distance from Home", "Companies Worked",
                ],
                "Value": [
                    age, job_level, f"₹{monthly_income:,}", years_company,
                    overtime, f"{job_sat}/4", f"{env_sat}/4",
                    f"{wlb}/4", f"{distance} km", num_companies,
                ]
            })
            st.dataframe(snapshot, hide_index=True, use_container_width=True)

            st.caption("Satisfaction profile (1 = Low, 4 = High)")
            sat_df = pd.DataFrame({
                "Dimension": ["Job Satisfaction", "Environment", "Work-Life Balance"],
                "Score": [job_sat, env_sat, wlb]
            })
            fig_s, ax_s = plt.subplots(figsize=(4, 2.2))
            bars = ax_s.barh(sat_df["Dimension"], sat_df["Score"], color=selected_info["color"])
            ax_s.set_xlim(0, 4)
            ax_s.set_xlabel("Score")
            for bar, val in zip(bars, sat_df["Score"]):
                ax_s.text(val + 0.05, bar.get_y() + bar.get_height()/2, str(val), va="center")
            st.pyplot(fig_s, use_container_width=True)


# =============================================================
# TAB 2 — MODEL COMPARISON
# =============================================================
with tab_compare:
    st.subheader("All Models — Side-by-Side Comparison")
    st.caption("Evaluated on the same held-out test set (294 employees, real-world 84/16 class distribution).")

    comp_data = {
        "Model": list(MODEL_REGISTRY.keys()),
        "Accuracy": [v["accuracy"] for v in MODEL_REGISTRY.values()],
        "Precision": [v["precision"] for v in MODEL_REGISTRY.values()],
        "Recall": [v["recall"] for v in MODEL_REGISTRY.values()],
        "F1-Score": [v["f1"] for v in MODEL_REGISTRY.values()],
        "AUC-ROC": [v["auc"] for v in MODEL_REGISTRY.values()],
    }
    comp_df = pd.DataFrame(comp_data).set_index("Model")
    st.dataframe(
        comp_df.style.highlight_max(axis=0, color="#d4edda").highlight_min(axis=0, color="#f8d7da"),
        use_container_width=True
    )

    st.divider()
    st.markdown("#### F1-Score and Recall Comparison (key metrics for attrition use case)")

    c1, c2 = st.columns(2)
    with c1:
        fig_f1, ax_f1 = plt.subplots(figsize=(6, 3.5))
        models_list = list(MODEL_REGISTRY.keys())
        f1_vals = [float(v["f1"].replace("%",""))/100 for v in MODEL_REGISTRY.values()]
        colors_bar = [v["color"] for v in MODEL_REGISTRY.values()]
        bars_f1 = ax_f1.bar(models_list, f1_vals, color=colors_bar, edgecolor="white")
        ax_f1.set_title("F1-Score by Model", fontweight="bold")
        ax_f1.set_ylabel("F1-Score")
        ax_f1.set_ylim(0, 0.7)
        ax_f1.set_xticklabels(models_list, rotation=15, ha="right", fontsize=9)
        for bar, val in zip(bars_f1, f1_vals):
            ax_f1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val*100:.1f}%", ha="center", fontsize=9)
        st.pyplot(fig_f1, use_container_width=True)

    with c2:
        fig_rec, ax_rec = plt.subplots(figsize=(6, 3.5))
        rec_vals = [float(v["recall"].replace("%",""))/100 for v in MODEL_REGISTRY.values()]
        bars_rec = ax_rec.bar(models_list, rec_vals, color=colors_bar, edgecolor="white")
        ax_rec.set_title("Recall by Model", fontweight="bold")
        ax_rec.set_ylabel("Recall")
        ax_rec.set_ylim(0, 1.0)
        ax_rec.set_xticklabels(models_list, rotation=15, ha="right", fontsize=9)
        for bar, val in zip(bars_rec, rec_vals):
            ax_rec.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val*100:.1f}%", ha="center", fontsize=9)
        st.pyplot(fig_rec, use_container_width=True)

    st.info(
        "💡 **Why Logistic Regression is the recommended model for this dashboard:** "
        "In an attrition prediction tool, missing an at-risk employee (false negative) is more costly than a false alarm. "
        "Logistic Regression's recall of 76.6% means it catches far more at-risk employees than any other model, "
        "and its F1-Score of 51.1% is the strongest overall balance between precision and recall. "
        "You can still switch models above to see how predictions change."
    )


# =============================================================
# TAB 3 — WORKFORCE INSIGHTS
# =============================================================
with tab_insights:
    st.subheader("Dataset-Level Attrition Patterns")
    st.caption("Charts from the IBM HR Analytics dataset used to train all models.")

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
            dept = raw.groupby("Department")["Attrition"].value_counts(normalize=True).unstack() * 100
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
            tenure = raw.groupby("YearsAtCompany")["Attrition"].value_counts(normalize=True).unstack() * 100
            tenure["Yes"].plot(kind="bar", ax=ax4, color="#C0392B")
            ax4.set_title("Attrition Rate (%) by Years at Company", fontweight="bold")
            ax4.set_xlabel("Years at Company")
            ax4.set_ylabel("Attrition Rate (%)")
            st.pyplot(fig4, use_container_width=True)

    except FileNotFoundError:
        st.warning(
            "Dataset file `WA_Fn-UseC_-HR-Employee-Attrition.csv` not found. "
            "Add it to the same folder as app.py to enable these charts."
        )


# =============================================================
# TAB 4 — PREDICTION LOG
# =============================================================
with tab_history:
    st.subheader("Predictions Made This Session")
    if len(st.session_state.history) == 0:
        st.info("No predictions yet. Run a prediction from the **Risk Predictor** tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        display_cols = [
            "Model Used", "Age", "JobLevel", "MonthlyIncome", "OverTime",
            "YearsAtCompany", "JobSatisfaction", "WorkLifeBalance",
            "Predicted Probability (%)", "Risk Tier"
        ]
        available = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(hist_df[available], use_container_width=True, hide_index=True)
        st.bar_chart(hist_df["Predicted Probability (%)"])
        if st.button("🗑 Clear Log"):
            st.session_state.history = []
            st.rerun()


# =============================================================
# TAB 5 — ABOUT
# =============================================================
with tab_about:
    st.subheader("About This Dashboard")
    st.markdown(
        """
This dashboard is the Week 6 deliverable of the **Employee Attrition Prediction Using Machine Learning**
Summer Internship Project at DCyber TechLab Pvt Ltd.

**Project pipeline**
- Raw dataset: IBM HR Analytics Employee Attrition (1,470 employees × 35 columns)
- Preprocessing: Label Encoding, StandardScaler, SMOTE to fix 84/16 class imbalance
- Models trained: Logistic Regression, Decision Tree, Random Forest, Random Forest (Tuned)
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC across all four models
- Recommended model: **Logistic Regression** (highest Recall 76.6% and F1-Score 51.1%)

**Files required alongside app.py**
```
logistic_regression.pkl
decision_tree.pkl
random_forest.pkl
random_forest_tuned.pkl
scaler.pkl
WA_Fn-UseC_-HR-Employee-Attrition.csv   (optional — for Workforce Insights tab)
```

**How to use**
1. Select a model from the dropdown at the top of the sidebar.
2. Fill in the employee profile using the sliders.
3. Click **▶ Run Prediction** — see the risk level, probability, and suggested actions.
4. Switch models freely to compare how predictions differ.
5. Use the **Model Comparison** tab to see all four models' metrics side by side.
6. Use the **Prediction Log** tab to review predictions made in this session.
        """
    )
    st.divider()
    st.markdown("**All four models — quick reference**")
    ref_data = {k: {
        "Accuracy": v["accuracy"], "F1-Score": v["f1"], "Recall": v["recall"], "AUC": v["auc"]
    } for k, v in MODEL_REGISTRY.items()}
    st.dataframe(pd.DataFrame(ref_data).T, use_container_width=True)

st.divider()
st.caption(
    "Chaithrali Chandrashekhar Parab — MMS, Aditya Institute of Management Studies and Research  |  "
    "DCyber TechLab Pvt Ltd  |  Summer Internship 2026"
)