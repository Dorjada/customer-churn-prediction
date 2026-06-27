# 🔮 Customer Churn Prediction

A full end-to-end machine learning project that predicts customer churn using XGBoost, with a live interactive Streamlit app and SHAP explainability.

---

## 📌 Business Problem

Customer churn costs businesses significantly more than retention. This project builds a production-ready ML model that identifies at-risk customers before they leave, enabling proactive retention strategies.

**Even a 1% reduction in monthly churn can translate to millions in preserved annual revenue.**

---

## 🎯 Project Objectives

- Accurately predict which customers are likely to churn
- Identify the key business drivers behind churn
- Deliver explainable, actionable predictions for retention teams
- Deploy an interactive app for real-time risk scoring

---

## 📊 Dataset

- **Source:** Kaggle — Customer Churn Prediction Business Dataset
- **Size:** 10,000 customers x 32 features
- **Target:** Binary churn label (10.2% churn rate)

---

## 🔬 Methodology

1. **EDA** — Churn rate analysis, distributions, correlation heatmap
2. **Preprocessing** — Label encoding, SMOTE, StandardScaler
3. **Modelling** — Logistic Regression baseline, XGBoost champion with GridSearchCV
4. **Threshold Tuning** — Optimal threshold 0.099 for max F1
5. **Explainability** — SHAP global + individual prediction explanations

### Results

| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.722 |
| XGBoost initial | 0.784 |
| XGBoost tuned | 0.780 |

**Champion model at optimal threshold:**
- Recall (Churners): 77%
- AUC-ROC: 0.780
- F1 Score: 0.37

---

## 💡 Top Churn Drivers (SHAP)

1. **csat_score** — Low satisfaction = high churn risk
2. **payment_failures** — Strongest early warning signal
3. **tenure_months** — New customers churn most
4. **monthly_logins** — Disengaged customers leave first

---

## 💼 Business Recommendations

1. Contact customers with CSAT score 2 or below within 48 hours
2. Trigger retention workflow after first payment failure
3. Structured onboarding for customers in first 6 months
4. Monthly batch scoring — prioritise top 15% highest-risk customers

---

## 🚀 Run the App

pip install -r requirements.txt
streamlit run app.py

Then open http://localhost:8501

---

## 🛠️ Tech Stack

Python, XGBoost, Streamlit, SHAP, scikit-learn, pandas, matplotlib, seaborn

---

## 👤 Author

**Dorjada Halili**

LinkedIn: https://linkedin.com/in/yourprofile
GitHub: https://github.com/yourusername
