
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='🔮',
    layout='wide'
)

@st.cache_resource
def load_model():
    model    = joblib.load('models/xgboost_churn_model.pkl')
    scaler   = joblib.load('models/scaler.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_model()

st.title('🔮 Customer Churn Predictor')
st.markdown('Enter a customer profile below to predict their churn risk.')
st.divider()

st.sidebar.header('Customer Profile')

gender           = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age              = st.sidebar.slider('Age', 18, 80, 35)
country          = st.sidebar.selectbox('Country', ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'India'])
city             = st.sidebar.selectbox('City', ['New York', 'London', 'Berlin', 'Paris', 'Toronto', 'Sydney', 'Mumbai'])
customer_segment = st.sidebar.selectbox('Customer Segment', ['SME', 'Individual', 'Enterprise'])
tenure_months    = st.sidebar.slider('Tenure (months)', 1, 60, 12)
signup_channel   = st.sidebar.selectbox('Signup Channel', ['Web', 'Mobile', 'Referral'])
contract_type    = st.sidebar.selectbox('Contract Type', ['Monthly', 'Quarterly', 'Yearly'])

st.sidebar.divider()

monthly_logins       = st.sidebar.slider('Monthly Logins', 0, 50, 15)
weekly_active_days   = st.sidebar.slider('Weekly Active Days', 0, 7, 3)
avg_session_time     = st.sidebar.slider('Avg Session Time (min)', 0.0, 40.0, 15.0)
features_used        = st.sidebar.slider('Features Used', 1, 15, 5)
usage_growth_rate    = st.sidebar.slider('Usage Growth Rate', -50.0, 50.0, 0.0)
last_login_days_ago  = st.sidebar.slider('Last Login (days ago)', 0, 70, 7)

st.sidebar.divider()

monthly_fee            = st.sidebar.selectbox('Monthly Fee ($)', [10, 20, 30, 40, 50, 60])
total_revenue          = st.sidebar.number_input('Total Revenue ($)', 0, 5000, 500)
payment_method         = st.sidebar.selectbox('Payment Method', ['Card', 'PayPal', 'Bank Transfer'])
payment_failures       = st.sidebar.slider('Payment Failures', 0, 5, 0)
discount_applied       = st.sidebar.selectbox('Discount Applied', ['Yes', 'No'])
price_increase_last_3m = st.sidebar.selectbox('Price Increase Last 3M', ['Yes', 'No'])

st.sidebar.divider()

support_tickets     = st.sidebar.slider('Support Tickets', 0, 7, 1)
avg_resolution_time = st.sidebar.slider('Avg Resolution Time (hrs)', 0.0, 72.0, 24.0)
complaint_type      = st.sidebar.selectbox('Complaint Type', ['No Complaint', 'Technical', 'Billing', 'Service'])
csat_score          = st.sidebar.select_slider('CSAT Score', options=[1.0, 2.0, 3.0, 4.0, 5.0], value=4.0)
escalations         = st.sidebar.slider('Escalations', 0, 4, 0)

st.sidebar.divider()

email_open_rate      = st.sidebar.slider('Email Open Rate', 0.0, 1.0, 0.4)
marketing_click_rate = st.sidebar.slider('Marketing Click Rate', 0.0, 1.0, 0.2)
nps_score            = st.sidebar.slider('NPS Score', -100, 100, 30)
survey_response      = st.sidebar.selectbox('Survey Response', ['Positive', 'Neutral', 'Negative'])
referral_count       = st.sidebar.slider('Referral Count', 0, 7, 1)

encode_map = {
    'gender':               {'Male': 1, 'Female': 0},
    'country':              {'USA': 6, 'UK': 5, 'Germany': 2, 'France': 1, 'Canada': 0, 'Australia': 3, 'India': 4},
    'city':                 {'New York': 3, 'London': 2, 'Berlin': 0, 'Paris': 4, 'Toronto': 6, 'Sydney': 5, 'Mumbai': 1},
    'customer_segment':     {'SME': 2, 'Individual': 1, 'Enterprise': 0},
    'signup_channel':       {'Web': 2, 'Mobile': 1, 'Referral': 0},
    'contract_type':        {'Monthly': 1, 'Quarterly': 2, 'Yearly': 0},
    'payment_method':       {'Card': 0, 'PayPal': 1, 'Bank Transfer': 2},
    'discount_applied':     {'Yes': 1, 'No': 0},
    'price_increase_last_3m': {'Yes': 1, 'No': 0},
    'complaint_type':       {'No Complaint': 2, 'Technical': 3, 'Billing': 0, 'Service': 1},
    'survey_response':      {'Positive': 1, 'Neutral': 0, 'Negative': 2},
}

input_data = {
    'gender':               encode_map['gender'][gender],
    'age':                  age,
    'country':              encode_map['country'][country],
    'city':                 encode_map['city'][city],
    'customer_segment':     encode_map['customer_segment'][customer_segment],
    'tenure_months':        tenure_months,
    'signup_channel':       encode_map['signup_channel'][signup_channel],
    'contract_type':        encode_map['contract_type'][contract_type],
    'monthly_logins':       monthly_logins,
    'weekly_active_days':   weekly_active_days,
    'avg_session_time':     avg_session_time,
    'features_used':        features_used,
    'usage_growth_rate':    usage_growth_rate,
    'last_login_days_ago':  last_login_days_ago,
    'monthly_fee':          monthly_fee,
    'total_revenue':        total_revenue,
    'payment_method':       encode_map['payment_method'][payment_method],
    'payment_failures':     payment_failures,
    'discount_applied':     encode_map['discount_applied'][discount_applied],
    'price_increase_last_3m': encode_map['price_increase_last_3m'][price_increase_last_3m],
    'support_tickets':      support_tickets,
    'avg_resolution_time':  avg_resolution_time,
    'complaint_type':       encode_map['complaint_type'][complaint_type],
    'csat_score':           csat_score,
    'escalations':          escalations,
    'email_open_rate':      email_open_rate,
    'marketing_click_rate': marketing_click_rate,
    'nps_score':            nps_score,
    'survey_response':      encode_map['survey_response'][survey_response],
    'referral_count':       referral_count,
}

input_df = pd.DataFrame([input_data])[feature_names]
input_sc  = scaler.transform(input_df)

THRESHOLD  = 0.099
proba      = model.predict_proba(input_sc)[0][1]
prediction = int(proba >= THRESHOLD)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Churn Probability', f'{proba:.1%}')
with col2:
    risk = 'HIGH RISK 🔴' if proba >= 0.4 else ('MEDIUM RISK 🟡' if proba >= 0.2 else 'LOW RISK 🟢')
    st.metric('Risk Level', risk)
with col3:
    st.metric('Prediction', 'Will Churn ⚠️' if prediction == 1 else 'Will Stay ✅')

st.divider()

st.subheader('Churn Risk Gauge')
fig, ax = plt.subplots(figsize=(8, 1.2))
ax.barh(['Risk'], [1], color='#e0e0e0', height=0.5)
ax.barh(['Risk'], [proba], color='#F44336' if proba >= 0.4 else ('#FF9800' if proba >= 0.2 else '#4CAF50'), height=0.5)
ax.axvline(THRESHOLD, color='black', linewidth=2, linestyle='--', label=f'Threshold ({THRESHOLD:.0%})')
ax.set_xlim(0, 1)
ax.set_xlabel('Churn Probability')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.legend(loc='upper right', fontsize=9)
ax.set_title(f'Predicted churn probability: {proba:.1%}', fontweight='bold')
plt.tight_layout()
st.pyplot(fig)

st.divider()

st.subheader('Why this prediction? — SHAP Explanation')
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_sc)
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_values[0],
        base_values   = explainer.expected_value,
        data          = input_df.iloc[0],
        feature_names = feature_names
    ),
    max_display=12,
    show=False
)
st.pyplot(plt.gcf())

st.divider()

st.subheader('Customer Profile Summary')
summary_df = pd.DataFrame({
    'Feature': ['CSAT Score', 'Payment Failures', 'Tenure (months)',
                'Monthly Logins', 'Support Tickets', 'Last Login (days ago)'],
    'Value':   [csat_score, payment_failures, tenure_months,
                monthly_logins, support_tickets, last_login_days_ago]
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)
