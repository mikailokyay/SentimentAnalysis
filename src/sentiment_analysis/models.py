"""This Module is using for crating training model and adding embedding layer"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Bidirectional, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.initializers.initializers_v2 import Constant
from metrics import recall, precision, f1_score
from model_constants import get_embedding_constants, LANG


def cnn_model(train_model):
    """
    This function is using for adding cnn layers if model type is cnn or hybrid
    :param train_model:
        Training model which cnn layers will be added
    :return:
        It returns training model with cnn layers
    """
    train_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    train_model.add(MaxPooling1D(pool_size=2))
    train_model.add(Flatten())
    train_model.add(Dense(250, activation='relu'))
    return train_model


def get_model(num_words, max_length, embedding_matrix):
    """
    This function is creating model with embedding matrix which is come from word2vec,
    glove or fasttext
    :param num_words:int
        Total number of words
    :param max_length:int
        Maximum length of sentences
    :param embedding_matrix:ndarray
        Embedding matrix of word2vec, glove or fasttext
    :return:
        It returns training model
    """
    embedding_constants = get_embedding_constants()
    train_model = Sequential()
    if embedding_constants["embedding_model"] == "glove" and LANG == "tr":
        embedding_dim = 300
    else:
        embedding_dim = 100
    embedding_layer = Embedding(num_words, embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length, trainable=False)
    train_model.add(embedding_layer)
    train_model.add(Dropout(0.25))

    if embedding_constants["model_type"] == "hybrid":
        train_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        train_model.add(MaxPooling1D(pool_size=2))
        train_model.add(Dropout(0.25))

        if embedding_constants["bidirectional"]:
            train_model.add(Bidirectional(embedding_constants["lstm_gru_type"](128)))
        else:
            train_model.add(embedding_constants["lstm_gru_type"](128))

    elif embedding_constants["model_type"] == "cnn":
        train_model = cnn_model(train_model)

    else:
        if embedding_constants["bidirectional"]:
            train_model.add(Bidirectional(embedding_constants["lstm_gru_type"](128)))
        else:
            train_model.add(embedding_constants["lstm_gru_type"](128))
    train_model.add(Dense(1, activation='sigmoid'))
    train_model.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['acc', recall, precision, f1_score])
    return train_model
