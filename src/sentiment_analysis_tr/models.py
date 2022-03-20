from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Bidirectional, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.initializers.initializers_v2 import Constant
from metrics import recall, precision, f1
from model_constants import get_embedding_constants


def cnn_model(train_model):
    train_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    train_model.add(MaxPooling1D(pool_size=2))
    train_model.add(Flatten())
    train_model.add(Dense(250, activation='relu'))
    return train_model


def get_model(num_words, max_length, embedding_matrix):
    embedding_constants = get_embedding_constants()
    train_model = Sequential()
    embedding_layer = Embedding(num_words, 100,
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
    train_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', recall, precision, f1])
    return train_model
