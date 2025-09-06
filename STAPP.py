import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("Gaussian_model.pk1")

st.title("Customer Churn Prediction")


age = st.slider("Age", min_value=18, max_value=100, value=30)
tenure = st.slider("Tenure", min_value=0, max_value=50, value=5)
gender = st.selectbox("Gender", ["Male", "Female"])

gender_encoded = 1 if gender == "Male" else 0

input_data = np.array([[age, tenure, gender_encoded]])
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.write("This customer will Churn.")
else:
    st.write("This customer will Not Churn.")

st.write(f"Probability of Churn: {probability:.2%}")

fig, ax = plt.subplots(figsize=(4, 1.2))
ax.barh(["Churn", "Not Churn"], [probability, 1 - probability], color=["red", "green"])
ax.set_xlim(0, 1)
for i, v in enumerate([probability, 1 - probability]):
    ax.text(v + 0.02, i, f"{v:.1%}", va="center")
st.pyplot(fig)