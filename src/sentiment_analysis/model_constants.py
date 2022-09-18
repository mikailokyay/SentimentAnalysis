"""This module is using for creating model constants"""
from tensorflow.python.keras.layers import CuDNNLSTM, CuDNNGRU

LANG = "en"
EMBEDDING_MODEL = "elmo"
RNN_LAYER_TYPE = "gru"
MODEL_TYPE = "not_cnn_or_hybrid"
BIDIRECTIONAL = True


def get_embedding_constants():
    """
    This function is using for getting embedding constants based on model constants
    :return:dict
        It returns embedding constants as dict
    """
    if RNN_LAYER_TYPE == "lstm":
        embedding_constants = {"model_type": MODEL_TYPE,
                               "bidirectional": BIDIRECTIONAL,
                               "embedding_model": EMBEDDING_MODEL,
                               "lstm_gru_type": CuDNNLSTM}
    else:
        embedding_constants = {"model_type": MODEL_TYPE,
                               "bidirectional": BIDIRECTIONAL,
                               "embedding_model": EMBEDDING_MODEL,
                               "lstm_gru_type": CuDNNGRU}
    return embedding_constants


def get_embedding_file():
    """
    This function is using for getting embedding model pretrained file path and name
    :return:dict
    It returns embedding model file path and name as dict
    """
    if LANG == "tr":
        if EMBEDDING_MODEL == "word2vec":
            embedding_model_file = {"file_path": '../../data/word2vec/',
                                    "file_name": 'hepsiburada_word2vec_model.txt'}
        elif EMBEDDING_MODEL == "glove":
            embedding_model_file = {"file_path": '../../data/glove/',
                                    "file_name": 'vectors.txt'}

        else:
            embedding_model_file = {"file_path": '../../data/fasttext/',
                                    "file_name": 'hepsiburada_fasttext_model.txt'}
    else:
        if EMBEDDING_MODEL == "word2vec":
            embedding_model_file = {"file_path": '../../data/word2vec/',
                                    "file_name": 'IMDB_word2vec_model.txt'}

        elif EMBEDDING_MODEL == "glove":
            embedding_model_file = {"file_path": '../../data/glove/',
                                    "file_name": 'glove.6B.100d.txt'}

        else:
            embedding_model_file = {"file_path": '../../data/fasttext/',
                                    "file_name": 'IMDB_fasttext_model.txt'}

    return embedding_model_file


def get_data_path():
    """
    This function is using for getting training data for English and Turkish
    :return:
    It returns data path
    """
    if LANG == "tr":
        data_path = "../../data/hepsiburada.csv"
    else:
        data_path = "../../data/IMDB_Dataset.csv"
    return data_path
