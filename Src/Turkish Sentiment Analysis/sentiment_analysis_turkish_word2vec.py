# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import asarray
from nltk.corpus import stopwords
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.python.keras.layers import GRU, Embedding, CuDNNGRU, LSTM, Bidirectional, CuDNNLSTM
from tensorflow.python.keras.layers import  Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from keras.initializers import Constant
from nltk.tokenize import word_tokenize
import string
import os

file_path = '../Data/Word2Vec/'
filename = 'hepsiburada_Word2Vec_Model.txt'
dataset = pd.read_csv('../Data/hepsiburada.csv')
dataset.sample(5)
dataset.isnull().values.any()

dataset.shape
dataset.head()
dataset["Review"][25]

tokenizer_obj = Tokenizer()
total_reviews = dataset["Review"].values
tokenizer_obj.fit_on_texts(total_reviews)

max_length = max([len(s.split()) for s in total_reviews])
max_length

review_lines = list()
lines = dataset["Review"].values.tolist()
for line in lines:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('turkish'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)
len(review_lines)


def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


Word2Vec_files = find_all(filename, file_path)

if len(Word2Vec_files) >= 1:
    Word2Vec_file = open('../Data/Word2Vec/hepsiburada_Word2Vec_Model.txt', encoding="utf8")
else:
    model = Word2Vec(sentences=review_lines, size=100, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format(filename, binary=False)
    Word2Vec_file = open('../Data/Word2Vec/hepsiburada_Word2Vec_Model.txt', encoding="utf8")

embeddings_dictionary = {}
for line in Word2Vec_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
Word2Vec_file.close()

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)
word_index = tokenizer_obj.word_index
review_pad = pad_sequences(sequences, maxlen=max_length)
sentiment = dataset['Rating'].values
# sentiment = np.array(list(map(lambda x: 1 if x=="positive" else 0, sentiment)))

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

for word, index in word_index.items():
    if index > num_words:
        continue
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1_form = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1_form), tf.zeros_like(f1_form), f1_form)
    return K.mean(f1)


def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    return K.mean(precision)


def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    recall = tp / (tp + fn + K.epsilon())
    return K.mean(recall)


model = Sequential()

embedding_layer = Embedding(num_words, 100,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length, trainable=False)
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(CuDNNGRU(128)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', recall, precision, f1])
model.summary()

validation_split = 0.2
indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(validation_split * review_pad.shape[0])

X_train_pad = review_pad[:-num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test_pad = review_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]

history = model.fit(X_train_pad, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(X_test_pad, y_test))
