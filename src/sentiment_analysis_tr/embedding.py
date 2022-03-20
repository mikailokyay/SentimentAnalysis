import os
import numpy as np
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
from model_constants import get_embedding_file, embedding_model


def find_all(path, name):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


def word2vec_file_load(review_lines):
    file = get_embedding_file()
    file_path = file["file_path"]
    file_name = file["file_name"]
    if len(find_all(file_path, file_name)) >= 1:
        word2vec_file = open(file_path + file_name, encoding="utf8")
    else:
        word2vec_model = Word2Vec(sentences=review_lines, vector_size=100, window=5, min_count=1, workers=4)
        word2vec_model.wv.save_word2vec_format(file_path + file_name, binary=False)
        word2vec_file = open(file_path + file_name, encoding="utf8")
    return word2vec_file


def fasttext_file_load(review_lines):
    file = get_embedding_file()
    file_path = file["file_path"]
    file_name = file["file_name"]
    if len(find_all(file_path, file_name)) >= 1:
        fasttext_file = open(file_path + file_name, encoding="utf8")
    else:
        fasttext_model = FastText(sentences=review_lines, vector_size=100, window=5, min_count=1, workers=4, sg=1)
        fasttext_model.wv.save_word2vec_format(file_path + file_name, binary=False)
        fasttext_file = open(file_path + file_name, encoding="utf8")
    return fasttext_file


def glove_file_load():
    file = get_embedding_file()
    file_path = file["file_path"]
    file_name = file["file_name"]
    glove_file = ""
    if len(find_all(file_path, file_name)) >= 1:
        glove_file = open(file_path + file_name, encoding="utf8")
    else:
        print("Glove file not found, load glove file (if lang=tr vectors.txt, else glove.6B.100d.txt)")
    return glove_file


def get_embedding_dictionary(review_lines):
    embeddings_dictionary = {}
    if embedding_model == "word2vec":
        embedding_file = word2vec_file_load(review_lines)
    elif embedding_model == "fasttext":
        embedding_file = fasttext_file_load(review_lines)
    elif embedding_model == "glove":
        embedding_file = fasttext_file_load(review_lines)
    else:
        pass

    for line in embedding_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    embedding_file.close()
    return embeddings_dictionary


def get_embedding_matrix(review_lines, num_words, word_index):
    embedding_matrix = np.zeros((num_words, 100))
    embedding_dictionary = get_embedding_dictionary(review_lines)
    for word, index in tqdm(word_index.items()):
        if index > num_words:
            continue
        embedding_vector = embedding_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix
