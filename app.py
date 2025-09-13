import streamlit as st
import pandas as pd
import numpy as np
import pickle


best_model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Car MPG Prediction App 🚗")
st.write("Enter the car features to predict MPG:")


horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=95.0)
weight = st.number_input("Weight (lbs)", min_value=1500, max_value=6000, value=2372)
acceleration = st.number_input("Acceleration", min_value=5.0, max_value=30.0, value=15.0)
model_year = st.number_input("Model Year", min_value=70, max_value=82, value=70)
origin = st.selectbox("Origin", [1, 2, 3])  # 1=US, 2=Europe, 3=Japan

if st.button("Predict MPG"):
  
    input_df = pd.DataFrame({
        'horsepower': [np.log1p(horsepower)],
        'weight': [np.log1p(weight)],
        'acceleration': [np.log1p(acceleration)],
        'model year': [model_year],
        'origin': [origin]
    })

 
    input_scaled = scaler.transform(input_df)

  
    predicted_mpg = best_model.predict(input_scaled)[0]
    st.success(f"Predicted MPG: {predicted_mpg:.2f}")
