import streamlit as st
import joblib
import pickle
import spacy

# Load the trained pipeline and encoder

pipeline = joblib.load('model_compressed.joblib')
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load spaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # Remove stop words, punctuation, and lemmatize the text (lowercased)
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_.lower())
    
    return " ".join(filtered_tokens)

# Streamlit App Layout
st.title('Text Sentiment Prediction  😊😢😐🤷‍♂️')

st.image('senti.jpg' , width=300)
# Brief explanation of how it works
st.write("""
    This app predicts the sentiment of a given text. The possible sentiments are:
    - Positive
    - Negative
    - Neutral
    - Irrelevant
    
    Here's how it works:
    1. **Preprocessing**: The input text is cleaned by removing stop words, punctuation, and lemmatizing the words.
    2. **Prediction**: The preprocessed text is passed through a trained machine learning model.
    3. **Output**: The model predicts the sentiment, and the app shows the result.
    
    Simply type or paste any text in the box below, click **"Predict Sentiment"**, and see the result!
""")
# Input Text Box
text = st.text_area("")

if st.button("Predict Sentiment"):
    if text:
        # Preprocess input text
        preprocessed_text = preprocess(text)
        
        # Predict the sentiment using the pipeline
        prediction = pipeline.predict([preprocessed_text])
        
        # Convert prediction from encoded value to label
        sentiment = encoder.inverse_transform(prediction)
        
        # Display the result
        st.write(f"The predicted sentiment is: {sentiment[0]}")
    else:
        st.write("Please enter some text to analyze.")
