import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
regmodel = pickle.load(open('/Users/states/Desktop/BostonHousing/best_rf_model.pkl', 'rb'))
scaler = pickle.load(open('/Users/states/Desktop/BostonHousing/scaling.pkl', 'rb'))

st.title("Boston Housing Price Prediction")
st.write("Enter the features to predict the house price:")

# Define input fields
feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10']  # Replace with actual feature names
user_input = []

for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}", value=0.0, step=0.1)
    user_input.append(value)

if st.button("Predict Price"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(pd.DataFrame([user_input]))
    prediction = regmodel.predict(scaled_input)[0]
    
    st.success(f"Predicted Price: ${prediction:,.2f}")
