from keras.layers import CuDNNLSTM, CuDNNGRU

lang = "tr"
embedding_model = "fasttext"
rnn_model_type = "gru"
model_type = "not_cnn_or_hybrid"


def rnn_model():

    if rnn_model_type == "lstm":
        model = CuDNNLSTM
    else:
        model = CuDNNGRU
    return model


def get_embedding_constants():

    if embedding_model == "word2vec":
        model_constants = {"model_type": model_type,
                           "bidirectional": False,
                           "embedding_model": "word2vec",
                           "lstm_gru_type": rnn_model()}

    elif embedding_model == "glove":
        model_constants = {"model_type": model_type,
                           "bidirectional": True,
                           "embedding_model": "glove",
                           "lstm_gru_type": rnn_model()}
    else:
        model_constants = {"model_type": model_type,
                           "bidirectional": True,
                           "embedding_model": "fasttext",
                           "lstm_gru_type": rnn_model()}
    return model_constants


def get_embedding_file():
    if lang == "tr":
        if embedding_model == "word2vec":
            embedding_model_file = {"file_path": '../../data/word2vec/',
                                    "file_name": 'hepsiburada_word2vec_model.txt'}
        elif embedding_model == "glove":
            embedding_model_file = {"file_path": '../../data/glove/',
                                    "file_name": 'vectors.txt'}

        else:
            embedding_model_file = {"file_path": '../../data/fasttext/',
                                    "file_name": 'hepsiburada_fasttext_model.txt'}
    else:
        if embedding_model == "word2vec":
            embedding_model_file = {"file_path": '../../data/word2vec/',
                                    "file_name": 'IMDB_word2vec_model.txt'}

        elif embedding_model == "glove":
            embedding_model_file = {"file_path": '../../data/glove/',
                                    "file_name": 'glove.6B.100d.txt'}

        else:
            embedding_model_file = {"file_path": '../../data/fasttext/',
                                    "file_name": 'IMDB_fasttext_model.txt'}

    return embedding_model_file


def get_data_path():
    if lang == "tr":
        data_path = "../../data/hepsiburada.csv"
    else:
        data_path = "../../data/IMDB_Dataset.csv"
    return data_path
