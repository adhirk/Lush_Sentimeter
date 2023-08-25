import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from my_tokenizer_module import my_tokenizer  # Import the my_tokenizer function
from PIL import Image


image = Image.open('LUSH.png')
st.image(image, '')

# Load the saved model and fitted CountVectorizer
model_path = './model.sav'
loaded_model = pickle.load(open(model_path, 'rb'))
vectorizer_path = './vectorizer.pkl'
loaded_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Streamlit app
st.title("Senti-meter")
# st.write("Enter your text below:")

# Text input
# user_input = st.text_area("### User review will be passed to this field")
# Text input
st.markdown("<h5>User review will be passed to the ML model to predict <span style='color:green;'>positive</span> / <span style='color:red;'>negative</span></h5>", unsafe_allow_html=True)
user_input = st.text_area("")

if st.button("Predict"):
    # Preprocess the user input
    processed_input = my_tokenizer(user_input)
    
    # Transform the processed input using loaded vectorizer
    input_transformed = loaded_vectorizer.transform([' '.join(processed_input)])
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_transformed)
    
    # Display the prediction with color based on positive/negative prediction
    if prediction[0] == 1:
        st.markdown("<p style='color:green;'>Prediction: Positive</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:red;'>Prediction: Negative</p>", unsafe_allow_html=True)