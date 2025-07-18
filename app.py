pip install streamlit scikit-learn pandas joblib

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load model and cleaned data
svm = joblib.load('svm_linear_model.pkl')  # path to your saved model
cleaned_df = pd.read_csv('cleaned_data.csv')  # or .pkl, .joblib as needed

st.title("SVM Classifier App")

# Show dataset preview
st.subheader("Cleaned Data Preview")
st.dataframe(cleaned_df.head())

# Input features dynamically from the user
st.subheader("Make a Prediction")

# Automatically generate input widgets for each feature
input_data = {}
for col in cleaned_df.drop(columns=['target']).columns:  # Replace 'target' with your label column
    val = st.number_input(f"Enter value for {col}", value=float(cleaned_df[col].mean()))
    input_data[col] = val

if st.button("Predict"):
    # Convert input to DataFrame for model
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = svm.predict(input_df)[0]
    
    st.success(f"Predicted Class: {prediction}")

# Optional: Show model accuracy
st.subheader("Model Accuracy")

# For demo purposes, re-evaluate accuracy if needed
# Load original train/test split if needed for live accuracy
# Here we just report hardcoded values (update as needed)
train_acc = 0.93  # Replace with your actual svm_train_acc
test_acc = 0.89   # Replace with your actual svm_test_acc

st.write(f"Training Accuracy: **{train_acc * 100:.2f}%**")
st.write(f"Testing Accuracy: **{test_acc * 100:.2f}%**")
