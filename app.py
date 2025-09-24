import streamlit as st
import joblib
import re

# Preprocess the input text
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    return text.strip() 

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Fake News Detection")

user_input = st.text_area("Enter the news text you want to check:")

# Prediction when button is clicked
if st.button("Predict"):
    if user_input.strip():  
        processed_text = preprocess_text(user_input)  # Preprocess the input text
        vect_text = vectorizer.transform([processed_text])  
        prediction = model.predict(vect_text)[0]  # Make prediction
        
        # Display the result
        label = "Real News" if prediction == 1 else "Fake News"
        st.success(f"Prediction: {label}")
    else:
        st.warning("Please enter some news text.")  # Show warning if input is empty
