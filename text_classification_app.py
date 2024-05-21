import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model and label encoder
model = joblib.load('complete_text_processing_pipeline.pkl')
encoder = joblib.load('label_encoder.pkl')



st.title('Text Classification Application')
user_input = st.text_input("Enter your text here for classification:")


if st.button('Predict'):
    if user_input:
        # Predict using the encapsulated preprocessing and model
        prediction = model.predict([user_input])  # Use model.predict directly if using label encoder externally
        prediction_proba = model.predict_proba([user_input])
        predicted_class_indices = np.argmax(prediction_proba, axis=1)
        
        # Decode the prediction to get human-readable labels
        final_label = encoder.inverse_transform(predicted_class_indices)
        st.write(f'Predicted Class: {final_label[0]}')
    else:
        st.write("Please enter some text to classify.")

