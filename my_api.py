from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import sequence
from pyvi.ViTokenizer import ViTokenizer
import numpy as np
import pickle
from keras import models
import re

app = Flask(__name__)

# Load stopwords
STOPWORDS = 'vietnamese-stopwords-dash.txt'
with open(STOPWORDS, "r", encoding="utf8") as ins:
    stopwords = set(line.strip('\n') for line in ins)

# Preprocessing functions
def filter_stop_words(sentence, stop_words):
    return ' '.join([word for word in sentence.split() if word not in stop_words])

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'', text)

def preprocess(text, tokenized=True, lowercased=True):
    text = ViTokenizer.tokenize(text) if tokenized else text
    text = filter_stop_words(text, stopwords)
    text = deEmojify(text)
    text = text.lower() if lowercased else text
    return text

def pre_process_features(X, tokenized=True, lowercased=True):
    return [preprocess(str(p), tokenized=tokenized, lowercased=lowercased) for p in X]

# Model and tokenizer loading
vocabulary_size = 10000
sequence_length = 100
embedding_dim = 300

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

with open('embeddings_index-001.pkl', 'rb') as file:
    embeddings_index = pickle.load(file)

model = models.load_model('Text_CNN_model_v13.keras')

# Feature generation
def make_features(X, tokenizer):
    X = tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(X, maxlen=sequence_length)
    return X

# Flask route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request body
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request. 'text' field is required."}), 400
        
        # Extract the input text
        input_text = data["text"]

        # Split sentences using regex for punctuation
        sentences = re.split(r'[.!?]+', input_text)
        # Remove empty strings and strip whitespace
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Preprocess sentences
        preprocessed_sentences = pre_process_features(sentences)
        features = make_features(preprocessed_sentences, tokenizer)

        # Make predictions
        predictions = model.predict(features, verbose=0)
        predictions = predictions.tolist()

        # Convert predictions to class labels
        class_labels = []
        for prediction in predictions:
            # Get the index of the highest probability
            predicted_class = np.argmax(prediction)
            # If class 0, it's "Clean", otherwise "Not Clean"
            class_labels.append("Clean" if predicted_class == 0 else "Not Clean")

        # Return results
        result = {"predictions": class_labels}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)