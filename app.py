# The training of model Loan Default Predictor was done in Google collab which is shared below and the retraining code is in retrain.py. The main app is in app.py.
# https://colab.research.google.com/drive/1171tWOSnkIVh1q4TIEVu2IBJDxUzm4jR?usp=sharing

# This is the main Streamlit app for predicting loan default risk.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Loan Default Risk Predictor")
st.markdown("Predicts whether a loan applicant is likely to default.")

# Load your saved files
@st.cache_resource
def load_models():
    model = joblib.load('models/current_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/encoders.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, encoders, feature_columns

@st.cache_data
def load_data():
    df = pd.read_csv('data/initial_training.csv')
    return df

try:
    model, scaler, encoders, feature_columns = load_models()
    df = load_data()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure all files are in the correct folders")
    st.stop()

# Sidebar info
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("Training Samples", f"{len(df):,}")
    st.metric("Features Used", len(feature_columns))
    default_rate = df['Default'].mean() * 100
    st.metric("Historical Default Rate", f"{default_rate:.1f}%")
    
    st.markdown("---")
    st.caption("Model: Random Forest Classifier")
    st.caption("Predicts loan default risk")

# Input form
st.subheader("📝 Loan Application Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=75000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=50000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680)

with col2:
    months_employed = st.number_input("Months Employed", min_value=0, max_value=480, value=48)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=20, value=5)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72], index=2)

with col3:
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.2f")
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

col4, col5 = st.columns(2)

with col4:
    has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
    has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with col5:
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
    has_co_signer = st.selectbox("Has Co-signer?", ["No", "Yes"])

if st.button("🔮 Predict Default Risk", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame([{
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
        'Education': education,
        'EmploymentType': employment_type,
        'MaritalStatus': marital_status,
        'HasMortgage': has_mortgage,
        'HasDependents': has_dependents,
        'LoanPurpose': loan_purpose,
        'HasCoSigner': has_co_signer
    }])
    
    # Step 1: Convert Yes/No to 0/1 for columns that are NOT in encoders
    # (These columns were numeric in training data)
    yes_no_columns = ['HasMortgage', 'HasDependents', 'HasCoSigner']
    for col in yes_no_columns:
        if col in input_data.columns and col not in encoders:
            input_data[col] = input_data[col].map({'No': 0, 'Yes': 1})
            st.write(f"Converted {col}: {input_data[col].values[0]}")  # Debug
    
    # Step 2: Apply encoding for categorical columns that ARE in encoders
    for col, encoder in encoders.items():
        if col in input_data.columns:
            value = input_data[col].iloc[0]
            
            # Check if the value exists in the encoder's classes
            if value not in encoder.classes_:
                st.error(f"Value '{value}' for {col} is not recognized.")
                st.write(f"Expected one of: {list(encoder.classes_)}")
                st.stop()
            
            # Transform the value
            input_data[col] = encoder.transform([value])[0]
            st.write(f"Encoded {col}: {input_data[col].values[0]}")  # Debug
    
    # Step 3: Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Step 4: Reorder columns to match training order
    input_data = input_data[feature_columns]
    
    # Step 5: Convert to float and scale
    input_numeric = input_data.astype(float)
    input_scaled = scaler.transform(input_numeric)
    
    # Step 6: Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK - Likely to Default")
        else:
            st.success("✅ LOW RISK - Likely to Repay")
    
    with col2:
        st.metric("Default Probability", f"{probability[1]*100:.1f}%")
    
    with col3:
        st.metric("Repayment Probability", f"{probability[0]*100:.1f}%")
    
    # Risk meter
    risk_percent = probability[1] * 100
    st.progress(int(risk_percent))
    
    if risk_percent < 30:
        st.info("📗 Low Risk: Applicant likely to repay loan")
    elif risk_percent < 60:
        st.warning("📙 Medium Risk: Further review recommended")
    else:
        st.error("📕 High Risk: High probability of default")
    
    # Store for feedback
    st.session_state['last_prediction'] = {
        'prediction': int(prediction),
        'probability': probability[1],
        'input_data': input_data.to_dict('records')[0]
    }

# Feedback section
st.markdown("---")
st.subheader("📝 Help Improve the Model")

with st.form("feedback_form"):
    actual_outcome = st.selectbox("Actual outcome", ["Re-paid successfully", "Defaulted on loan"])
    submitted = st.form_submit_button("Submit Feedback")
    
    if submitted and 'last_prediction' in st.session_state:
        feedback = pd.DataFrame([{
            'timestamp': datetime.now(),
            'predicted': st.session_state['last_prediction']['prediction'],
            'probability': st.session_state['last_prediction']['probability'],
            'actual': 1 if actual_outcome == "Defaulted on loan" else 0
        }])
        
        try:
            existing = pd.read_csv('data/user_feedback.csv')
            updated = pd.concat([existing, feedback])
        except:
            updated = feedback
        
        updated.to_csv('data/user_feedback.csv', index=False)
        st.success("✅ Thank you! Your feedback helps improve predictions.")
        st.balloons()

# Footer
st.markdown("---")
st.caption("""
**How it works:** This model uses a Random Forest classifier trained on historical loan data.
The model considers factors like credit score, income, debt-to-income ratio, and employment history.
""")