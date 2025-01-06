from tensorflow.keras.preprocessing import text, sequence
from pyvi.ViTokenizer import ViTokenizer
import pandas as pd
import numpy as np

input_text = "Mẹ bạn thật đẹp"

STOPWORDS = 'vietnamese-stopwords-dash.txt'
with open(STOPWORDS, "r", encoding="utf8") as ins:
    stopwords = []
    for line in ins:
        dd = line.strip('\n')
        stopwords.append(dd)
    stopwords = set(stopwords)

def filter_stop_words(train_sentences, stop_words):
    new_sent = [word for word in train_sentences.split() if word not in stop_words]
    train_sentences = ' '.join(new_sent)
    return train_sentences

import re
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def preprocess(text, tokenized = True, lowercased = True):
    text = ViTokenizer.tokenize(text) if tokenized else text
    text = filter_stop_words(text, stopwords)
    text = deEmojify(text)
    text = text.lower() if lowercased else text
    return text

def pre_process_features(X, tokenized = True, lowercased = True):
    X = [preprocess(str(p), tokenized = tokenized, lowercased = lowercased) for p in list(X)]
    for idx, ele in enumerate(X):
        if not ele:
            np.delete(X, idx)
    return X

vocabulary_size = 10000
sequence_length = 100
embedding_dim = 300

def make_features(X, tokenizer):
    X = tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(X, maxlen=sequence_length)
    return X, 

# tokenizer = text.Tokenizer(lower=False, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
# tokenizer.fit_on_texts(input_text)

import pickle

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))

with open('embeddings_index.pkl', 'rb') as file:
    embeddings_index = pickle.load(file)

for word, i in word_index.items():
    if i >= vocabulary_size:
        continue

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

input_text = make_features([input_text], tokenizer)

from keras import models
model = models.load_model('Text_CNN_model_v13.keras')
prediction = model.predict(input_text, verbose=0)
print(prediction)