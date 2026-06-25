# app.py  –  Employee Attrition Prediction Dashboard
# Created by: Chaitrali Parab
# DCyber TechLab Pvt Ltd | Aditya Institute of Management Studies and Research

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page configuration ──────────────────────────────────
st.set_page_config(
    page_title='Employee Attrition Predictor',
    page_icon='U0001f4ca',
    layout='wide'
)
# ── Load model and scaler from same folder as app.py ────
@st.cache_resource
def load_model():
    model  = joblib.load('random_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Page title ───────────────────────────────────────────
st.title('U0001f4ca  Employee Attrition Prediction Dashboard')
st.markdown('**DCyber TechLab Pvt Ltd  |  Aditya Institute of Management Studies and Research**')
st.write('Enter employee details in the sidebar to predict attrition risk.')
st.divider()

# ── Sidebar input form ───────────────────────────────────
st.sidebar.header('U0001f464  Enter Employee Details')
st.sidebar.markdown('---')

age            = st.sidebar.slider('Age', 18, 60, 30)
monthly_income = st.sidebar.number_input('Monthly Income', 1000, 20000, 5000, step=500)
overtime       = st.sidebar.selectbox('OverTime', ['No', 'Yes'])
job_sat        = st.sidebar.slider('Job Satisfaction  (1=Low, 4=High)', 1, 4, 3)
env_sat        = st.sidebar.slider('Environment Satisfaction', 1, 4, 3)
wlb            = st.sidebar.slider('Work-Life Balance', 1, 4, 3)
years_company  = st.sidebar.slider('Years at Company', 0, 40, 5)
distance       = st.sidebar.slider('Distance from Home (km)', 1, 30, 10)
job_level      = st.sidebar.selectbox('Job Level', [1, 2, 3, 4, 5])
num_companies  = st.sidebar.slider('No. of Companies Worked', 0, 9, 2)

overtime_enc = 1 if overtime == 'Yes' else 0

# ── Build input vector (31 features matching training order) ─
input_data = pd.DataFrame([{
    'Age': age, 'BusinessTravel': 1, 'DailyRate': 800,
    'Department': 2, 'DistanceFromHome': distance,
    'Education': 3, 'EducationField': 3,
    'EmployeeCount': 1, 'EnvironmentSatisfaction': env_sat,
    'Gender': 1, 'HourlyRate': 65, 'JobInvolvement': 3,
    'JobLevel': job_level, 'JobRole': 5,
    'JobSatisfaction': job_sat, 'MaritalStatus': 1,
    'MonthlyIncome': monthly_income, 'MonthlyRate': 14000,
    'NumCompaniesWorked': num_companies, 'OverTime': overtime_enc,
    'PercentSalaryHike': 14, 'PerformanceRating': 3,
    'RelationshipSatisfaction': 3, 'StockOptionLevel': 1,
    'TotalWorkingYears': max(age - 22, 1), 'TrainingTimesLastYear': 3,
    'WorkLifeBalance': wlb, 'YearsAtCompany': years_company,
    'YearsInCurrentRole': max(years_company - 2, 0),
    'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 3,
}])

# ── Prediction section ────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader('U0001f9e0  Attrition Risk Prediction')
    if st.button('  Predict Attrition Risk  ', type='primary', use_container_width=True):
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1] * 100
        if prediction == 1:
            st.error(f'⚠️  HIGH RISK – {probability:.1f}% probability of leaving')
            st.write('This employee is at elevated risk of attrition.')
            st.write('Recommended action: HR review within 30 days.')
        else:
            st.success(f'✅  LOW RISK – Only {probability:.1f}% probability of leaving')
            st.write('This employee appears stable.')

with col2:
    st.subheader('U0001f4cb  Input Summary')
    summary = pd.DataFrame({
        'Attribute': ['Age','Monthly Income','OverTime','Job Satisfaction',
                       'Work-Life Balance','Years at Company','Distance from Home'],
        'Value':     [age, f'{monthly_income:,}', overtime,
                       f'{job_sat}/4', f'{wlb}/4', years_company, f'{distance} km']
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ── Analytics section ─────────────────────────────────────
st.divider()
st.subheader('U0001f4ca  Attrition Analytics – Dataset Overview')

try:
    raw = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,4))
        dept = raw.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()*100
        dept['Yes'].sort_values().plot(kind='barh', ax=ax, color='#C0392B')
        ax.set_title('Attrition Rate by Department (%)', fontweight='bold')
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.countplot(x='OverTime', hue='Attrition', data=raw,
                      palette=['#2E74B5','#C0392B'], ax=ax2)
        ax2.set_title('Overtime vs Attrition', fontweight='bold')
        st.pyplot(fig2)
except FileNotFoundError:
    st.info('Upload the dataset CSV to your GitHub repo to enable analytics charts.')

# ── Footer ────────────────────────────────────────────────
st.divider()
m1,m2,m3,m4 = st.columns(4)
m1.metric('Model',    'Random Forest')
m2.metric('Accuracy', '~85%')
m3.metric('F1-Score', '~82%')
m4.metric('AUC-ROC',  '~0.91')
st.caption('Chaitrali Parab | DCyber TechLab Pvt Ltd | 2026')
