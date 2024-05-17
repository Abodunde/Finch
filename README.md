The validity/performance of the model is dependent on the labelling technique. For example, a procedural NLP approach was used which is less robust than using LLMs.

LLMs are usually more robust due to the training data as well as the domain knowledge they seem to have. As at the completion of this task, the main task seemed to be in understanding the logic in labelling hence the non-usage of LLMs in the first place.

Consequently, to run a prediction on entirely new reddit comments, the following preprocessing should first be done:


STEP 1:

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    text = text.lower() #lowercase conversion
    text = re.sub(r'[^a-zA-Z\s]', '', text) #removal of special characters
    tokens = word_tokenize(text) #tokenization of the text
    
    stop_words = set(stopwords.words('english')) #load the stopword instance
    filtered_tokens = [token for token in tokens if token not in stop_words] #remove stopwords
    
    preprocessed_text = ' '.join(filtered_tokens) # Join the tokens back into a single string
    
    return preprocessed_text

The preprocess_text function tokenizes, removes special characters, and removes stopwords.


STEP 2:

VECTORIZE THE COMMENT:

First we need to load the saved vectorizer:

import joblib

# Path to your .joblib file
filename = 'path/to/your_model.joblib' #set the path to the location of the 'vectorizer.pkl' file

# Load the model using joblib
vectorizer = joblib.load(filename)

Run the following line:

comment_vectorized = vectorizer.transform(comment)

STEP 3:

ENCODE THE TARGET VARIABLE:
import joblib

# Path to your .joblib file
filename = 'path/to/your_model.joblib' #set the path to the location of the 'label_encoder.pkl' file

# Load the model using joblib
encoder = joblib.load(filename)

Run the following line:

comment_encoded = encoder.transform(label) 

IT IS IMPORTANT THAT YOUR LABELS ARE IN THE FORMAT THAT THE INITIAL MODEL WERE TRAINED ON: **THAT IS "veterinarian", "doctor" OR "others"**
STEP 4:

RUN YOUR PREDICTION

import joblib

# Path to your .joblib file
filename = 'path/to/your_model.joblib' #set the path to the location of the 'xgboost_model.pkl' file

# Load the model using joblib
model = joblib.load(filename)

Run the following line:

comment_encoded = model.predict(comment)


NOTE: To reiterate, the validity of the model itself is based on the validity of the lables. If the labels from the new reddit comments are doen through LLms, there could be a concept drift for the simple reason that LLMs are much more robust that any procedural logic.

I can in no time, implement a new model which have been labelled by LLMs. If this does not meet the 70% accuracy threshold, please let me know and I will reimplement the labelling using LLMs.

Thank you for your consideration.



