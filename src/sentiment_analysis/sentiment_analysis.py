""" This module is main module for Turkish and English sentiment analysis model training """

import string
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.sentiment_analysis.models import get_model, get_elmo_model, get_transformer_model
from src.sentiment_analysis.embedding import get_embedding_matrix
from src.sentiment_analysis.model_constants import get_data_path, EMBEDDING_MODEL, LANG
from src.sentiment_analysis.model_constants import PRETRAINED_MODEL, PRETRAINED_MODEL_NAME
from sklearn.metrics import accuracy_score


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allocator_type ="BFC"
# config.gpu_options.per_process_gpu_memory_fraction = 0.90


class SentimentAnalysis:
    """
    This class is using for creating embedding matrix and train sentiment analysis model.
    """

    def __init__(self):
        self.dataset = pd.read_csv(get_data_path())
        if LANG == "en":
            self.dataset["sentiment"] = np.array(list(map(lambda x: 1 if x == "positive" else 0,
                                               self.dataset["sentiment"])))

        self.total_reviews = self.dataset["Review"].values
        self.sentiment = self.dataset['Rating'].values

        if EMBEDDING_MODEL != "elmo":
            self.max_length = max([len(s.split()) for s in self.total_reviews])
            self.review_lines = self.review_lines_create()
            self.word_index, self.review_pad = self.padding()

    def review_lines_create(self):
        """
        In this function, punctuation marks and stopwords are
        removing after splitting the words in each review.
        After that each review is adding in a review line list.

        :return:list
            It returns  List of word list for each split reviews
        """
        if LANG == "tr":
            stop_words = set(stopwords.words('turkish'))
        else:
            stop_words = set(stopwords.words('english'))
        review_lines = []
        for line in tqdm(self.total_reviews.tolist()):
            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            words = [w for w in words if w not in stop_words]
            review_lines.append(words)
        return review_lines

    def padding(self):
        """
        In this function, paddings are adding to each review.

        :return:dict,ndarray
            It returns  word index and reviews which have paddings
        """
        tokenizer_obj = tf.keras.preprocessing.text.Tokenizer()
        tokenizer_obj.fit_on_texts(self.review_lines)
        sequences = tokenizer_obj.texts_to_sequences(self.review_lines)
        word_index = tokenizer_obj.word_index
        review_pad = tf.keras.utils.pad_sequences(sequences, maxlen=self.max_length)
        return word_index, review_pad

    def get_train_test_data(self):
        """
        In this function, train and test data is splitting from whole data which is paddings added

        :return:list, list, list, list
            It returns  x_train, y_train, x_test, y_test
        """
        validation_split = 0.2
        index_array = np.arange(self.review_pad.shape[0])
        np.random.shuffle(index_array)
        num_validation_samples = int(validation_split * self.review_pad[index_array].shape[0])
        x_train_pad = self.review_pad[index_array][:-num_validation_samples]
        y_train = self.sentiment[index_array][:-num_validation_samples]
        x_test_pad = self.review_pad[index_array][-num_validation_samples:]
        y_test = self.sentiment[index_array][-num_validation_samples:]
        return x_train_pad, y_train, x_test_pad, y_test

    @property
    def train(self):
        """
       In this function, embed_model training for sentiment analysis is doing.

       :return:
           It returns  trained embed_model.
       """
        if EMBEDDING_MODEL == "elmo":
            x_train, x_test, y_train, y_test = train_test_split(self.total_reviews,
                                                                pd.get_dummies(self.sentiment).values,
                                                                test_size=0.2, random_state=0)
            embed_model = get_elmo_model()
            embed_model.fit(np.array(x_train),
                            np.array(y_train),
                            epochs=5,
                            batch_size=16,
                            validation_data=(x_test, y_test), use_multiprocessing=True, workers=8)
        elif EMBEDDING_MODEL == "transformers":

            train_df, val_df = train_test_split(self.dataset, test_size=0.2, random_state=0)
            embed_model = get_transformer_model(PRETRAINED_MODEL, PRETRAINED_MODEL_NAME, set(self.sentiment))
            embed_model.train_model(train_df, val_df, acc=accuracy_score)

        else:
            num_words = len(self.word_index) + 1
            x_train_pad, y_train, x_test_pad, y_test = self.get_train_test_data()
            embedding_matrix = get_embedding_matrix(self.review_lines, num_words, self.word_index)
            embed_model = get_model(num_words, self.max_length, embedding_matrix)
            embed_model.summary()
            embed_model.fit(x_train_pad, y_train, batch_size=8, epochs=5, verbose=1,
                            validation_data=(x_test_pad, y_test), use_multiprocessing=True, workers=-1)
        return embed_model


if __name__ == '__main__':
    sentiment_analysis = SentimentAnalysis()
    model = sentiment_analysis.train


