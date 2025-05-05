# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 13:44:57 2025
@author: NoorenMariyam
"""





import streamlit as st
import pickle
import numpy as np

# Load trained model
model_path = "https://github.com/Noorenmariyam/diabetesxgboost/raw/main/xgb_diabetes_model_joblib.pkl"
loaded_model = pickle.load(open(model_path, "rb"))

# Function for diabetes prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return "🚨 The person is diabetic 🚨" if prediction[0] == 1 else "✅ The person is NOT diabetic ✅"

# Main function
def main():
    st.title("🔥 Diabetes Prediction Web App 🔥")

    # Sliders for user input with a fixed theme
    Pregnancies = st.slider("Number of Pregnancies", 0, 20, 1)
    Glucose = st.slider("Glucose Level", 50, 200, 100)
    BloodPressure = st.slider("Blood Pressure", 40, 180, 80)
    SkinThickness = st.slider("Skin Thickness", 0, 100, 20)
    Insulin = st.slider("Insulin Level", 0, 500, 30)
    BMI = st.slider("BMI Value", 10.0, 50.0, 25.0)
    DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.slider("Age", 1, 120, 30)

    # Prediction Button
    diagnosis = ""
    if st.button("🔴 Predict Diabetes Status 🔴"):
        input_features = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction(input_features)

    # Display Result in Big, Bold Format
    st.subheader(diagnosis)

# Run Streamlit app
if __name__ == "__main__":
    main()
