import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import spacy
import re
from nltk.corpus import wordnet
from sklearn.base import BaseEstimator, TransformerMixin
nltk_resources = ["punkt", "wordnet", "averaged_perceptron_tagger"]
import nltk
for resource in nltk_resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

def load_spacy_model(model_name='en_core_web_md'):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"{model_name} not found, downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

nlp = load_spacy_model()

import warnings
warnings.filterwarnings("ignore")

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        # Apply preprocessing to each element in X
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        tokens = word_tokenize(text)  # Tokenize
        filtered_tokens = [token for token in tokens if token not in self.stop_words]  # Remove stopwords
        return ' '.join(filtered_tokens)  # Join tokens back into a single string



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

