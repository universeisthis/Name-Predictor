import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("name_classifier.pkl")
vectorizer = joblib.load("name_vectorizer.pkl")

# Streamlit UI
st.title("T Name Predictor")
st.write("Enter your name to find out the prediction")

# Input from user
name_input = st.text_input("Enter the name Son:")

# When button is clicked
if st.button("Predict"):
    if name_input.strip() == "":
        st.warning("Please enter a name.")
    else:
        # Transform and predict
        vectorized_input = vectorizer.transform([name_input])
        prediction = model.predict(vectorized_input)[0]
        st.success(f"Prediction: {'T confirmed' if prediction == 1 else 'False'}")
