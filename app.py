
import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("models/svm_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="SVM Classification",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 SVM Classification App")

st.write(
    "Enter the input features below to predict the classification result."
)

# User inputs
feature_1 = st.number_input("Feature 1", value=0.0, step=0.1)
feature_2 = st.number_input("Feature 2", value=0.0, step=0.1)

# Prediction
if st.button("Predict"):
    input_data = np.array([[feature_1, feature_2]])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("Predicted Class: 1")
    else:
        st.info("Predicted Class: 0")
