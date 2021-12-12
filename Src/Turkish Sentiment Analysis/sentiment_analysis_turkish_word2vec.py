# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import asarray
from nltk.corpus import stopwords
import tensorflow as tf
import keras.backend as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.python.keras.layers import GRU, Embedding, CuDNNGRU, LSTM, Bidirectional, CuDNNLSTM
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from keras.initializers import Constant
from nltk.tokenize import word_tokenize
import string
import os


class Word2VecTurkish:

    def __init__(self):
        self.file_path = '..\\..\\Data\\Word2Vec'
        self.filename = 'hepsiburada_Word2Vec_Model.txt'
        self.dataset = pd.read_csv('..\\..\\Data\\hepsiburada.csv')
        self.total_reviews = self.dataset["Review"].values
        # self.total_reviews = np.array(list(map(lambda x: 1 if x=="positive" else 0, self.total_reviews)))
        self.max_length = max([len(s.split()) for s in self.total_reviews])
        self.review_lines = self.review_lines_create()
        self.Word2Vec_files = self.find_all(self.filename, self.file_path)
        self.word2vec_file = self.word2vec_file_load()
        self.embedding_dictionary = self.get_embedding_dictionary()
        self.word_index, self.review_pad = self.padding()
        self.num_words = len(self.word_index) + 1
        self.embedding_matrix = self.get_embedding_matrix(self.embedding_dictionary)

    def tokenizer_create(self):
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(self.total_reviews)

    def review_lines_create(self):
        review_lines = []
        for line in self.total_reviews.tolist():
            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('turkish'))
            words = [w for w in words if w not in stop_words]
            review_lines.append(words)
        return review_lines

    @staticmethod
    def find_all(name, path):
        result = []
        for root, dirs, files in os.walk(path):
            if name in files:
                result.append(os.path.join(root, name))
        return result

    def word2vec_file_load(self):
        if len(self.Word2Vec_files) >= 1:
            word2vec_file = open('..\\..\\Data\\Word2Vec\\hepsiburada_Word2Vec_Model.txt', encoding="utf8")
        else:
            model = Word2Vec(sentences=self.review_lines, size=100, window=5, min_count=1, workers=4)
            model.wv.save_word2vec_format('..\\..\\Data\\Word2Vec\\' + self.filename, binary=False)
            word2vec_file = open('..\\..\\Data\\Word2Vec\\hepsiburada_Word2Vec_Model.txt', encoding="utf8")
        return word2vec_file

    def get_embedding_dictionary(self):
        embeddings_dictionary = {}
        for line in self.word2vec_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
        self.word2vec_file.close()
        return embeddings_dictionary

    def padding(self):
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(self.review_lines)
        sequences = tokenizer_obj.texts_to_sequences(self.review_lines)
        word_index = tokenizer_obj.word_index
        review_pad = pad_sequences(sequences, maxlen=self.max_length)
        return word_index, review_pad

    def get_embedding_matrix(self, embeddings_dictionary):
        embedding_matrix = np.zeros((self.num_words, 100))
        for word, index in self.word_index.items():
            if index > self.num_words:
                continue
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    @staticmethod
    def f1(y_true, y_pred):
        y_pred = k.round(y_pred)
        tp = k.sum(k.cast(y_true * y_pred, 'float'), axis=0)
        # tn = k.sum(k.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = k.sum(k.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = k.sum(k.cast(y_true * (1 - y_pred), 'float'), axis=0)
        p = tp / (tp + fp + k.epsilon())
        r = tp / (tp + fn + k.epsilon())
        f1_form = 2 * p * r / (p + r + k.epsilon())
        f1 = tf.where(tf.math.is_nan(f1_form), tf.zeros_like(f1_form), f1_form)
        return k.mean(f1)

    @staticmethod
    def precision(y_true, y_pred):
        y_pred = k.round(y_pred)
        tp = k.sum(k.cast(y_true * y_pred, 'float'), axis=0)
        fp = k.sum(k.cast((1 - y_true) * y_pred, 'float'), axis=0)
        precision = tp / (tp + fp + k.epsilon())
        return k.mean(precision)

    @staticmethod
    def recall(y_true, y_pred):
        y_pred = k.round(y_pred)
        tp = k.sum(k.cast(y_true * y_pred, 'float'), axis=0)
        fn = k.sum(k.cast(y_true * (1 - y_pred), 'float'), axis=0)
        recall = tp / (tp + fn + k.epsilon())
        return k.mean(recall)

    def run(self):
        model = Sequential()
        embedding_layer = Embedding(self.num_words, 100,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_length, trainable=False)
        model.add(embedding_layer)
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(CuDNNGRU(128)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', self.recall, self.precision,
                                                                             self.f1])
        model.summary()
        validation_split = 0.2
        indices = np.arange(self.review_pad.shape[0])
        np.random.shuffle(indices)
        num_validation_samples = int(validation_split * self.review_pad[indices].shape[0])
        x_train_pad = self.review_pad[indices][:-num_validation_samples]
        y_train = self.total_reviews[indices][:-num_validation_samples]
        x_test_pad = self.review_pad[indices][-num_validation_samples:]
        y_test = self.total_reviews[indices][-num_validation_samples:]

        history = model.fit(x_train_pad, y_train, batch_size=128, epochs=5, verbose=1,
                            validation_data=(x_test_pad, y_test))
        return history


if __name__ == '__main__':
    w2v_tr = Word2VecTurkish()
    training = w2v_tr.run()
