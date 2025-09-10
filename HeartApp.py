import streamlit as st
import numpy as np
import pickle

# Load saved models and scaler
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# App Title
st.title("Heart Disease Prediction App")

# Choose model
model_choice = st.selectbox("Choose Prediction Model", ["Logistic Regression", "K-Nearest Neighbor"])

# Input fields
age = st.number_input("Age", 18, 100)

# Sex
sex_input = st.radio("Sex", ["Female", "Male"])
sex = 0 if sex_input == "Female" else 1

# Chest Pain Type
cp_dict = {
    'Typical Angina': 0,
    'Atypical Angina': 1,
    'Non-anginal Pain': 2,
    'Asymptomatic': 3
}
cp_input = st.selectbox("Chest Pain Type", list(cp_dict.keys()))
cp = cp_dict[cp_input]

trestbps = st.number_input("Resting Blood Pressure (mm Hg)")
chol = st.number_input("Serum Cholesterol (mg/dl)")

# Fasting Blood Sugar
fbs_input = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs_input == "Yes" else 0

# Resting ECG
restecg_dict = {
    'Normal': 0,
    'ST-T wave abnormality': 1,
    'Left ventricular hypertrophy': 2
}
restecg_input = st.selectbox("Resting ECG Results", list(restecg_dict.keys()))
restecg = restecg_dict[restecg_input]

thalach = st.number_input("Max Heart Rate Achieved")
exang_input = st.radio("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang_input == "Yes" else 0

oldpeak = st.number_input("ST Depression Induced by Exercise")

# Slope
slope_dict = {
    'Up sloping': 0,
    'Flat': 1,
    'Down sloping': 2
}
slope_input = st.selectbox("Slope of ST Segment", list(slope_dict.keys()))
slope = slope_dict[slope_input]

ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

# Thalassemia
thal_dict = {
    'Normal': 1,
    'Fixed Defect': 2,
    'Reversible Defect': 3
}
thal_input = st.selectbox("Thalassemia", list(thal_dict.keys()))
thal = thal_dict[thal_input]

# Prepare input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(scaled_input)
    else:
        prediction = knn_model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease")
        st.info("Please consult a cardiologist soon.")
    else:
        st.success("Low risk of heart disease")


# Footer
st.markdown("---")
st.caption("Built with Streamlit")