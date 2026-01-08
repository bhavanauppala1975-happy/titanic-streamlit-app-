import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Encoding
sex = 0 if sex == "male" else 1
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_df = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S]],
    columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.success(f"Passenger Survived ✅ (Probability: {prob:.2f})")
    else:
        st.error(f"Passenger Did Not Survive ❌ (Probability: {prob:.2f})")
