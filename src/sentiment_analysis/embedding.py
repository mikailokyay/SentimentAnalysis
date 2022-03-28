"""
This module is using for creating embedding matrix for word2vec, glove and fasttext
"""
import os
import numpy as np
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
from model_constants import get_embedding_file, EMBEDDING_MODEL, LANG


def find_all(path, name):
    """
    Check file in directory
    :param path: str
        path of file
    :param name:str
        name of file
    :return:list
        It is returns file in path as list
    """
    result = []
    for root, _, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


def word2vec_file_load(review_lines):
    """
    If it is existing in path loading word2vec file else creating and loading file
    :param review_lines:list
        For each review list of words
    :return:list
    It returns word2vec file which include word vectors as list
    """
    file = get_embedding_file()
    file_path = file["file_path"]
    file_name = file["file_name"]
    if len(find_all(file_path, file_name)) >= 1:
        with open(file_path + file_name, encoding="utf8") as opened_file:
            word2vec_file = opened_file.readlines()
    else:
        word2vec_model = Word2Vec(sentences=review_lines, vector_size=100,
                                  window=5, min_count=1, workers=4)
        word2vec_model.wv.save_word2vec_format(file_path + file_name, binary=False)
        with open(file_path + file_name, encoding="utf8") as opened_file:
            word2vec_file = opened_file.readlines()
    return word2vec_file


def fasttext_file_load(review_lines):
    """
    If it is existing in path loading fasttext file else creating and loading file
    :param review_lines:list
        For each review list of words
    :return:list
    It returns fasttext file which include word vectors as list
    """
    file = get_embedding_file()
    file_path = file["file_path"]
    file_name = file["file_name"]
    if len(find_all(file_path, file_name)) >= 1:
        with open(file_path + file_name, encoding="utf8") as opened_file:
            fasttext_file = opened_file.readlines()
    else:
        fasttext_model = FastText(sentences=review_lines, vector_size=100,
                                  window=5, min_count=1, workers=4, sg=1)
        fasttext_model.wv.save_word2vec_format(file_path + file_name, binary=False)
        with open(file_path + file_name, encoding="utf8") as opened_file:
            fasttext_file = opened_file.readlines()
    return fasttext_file


def glove_file_load():
    """
    If it is existing in path loading glove.
    :return:list
    It returns glove file which include word vectors as list
    """
    file = get_embedding_file()
    file_path = file["file_path"]
    file_name = file["file_name"]
    glove_file = ""
    if len(find_all(file_path, file_name)) >= 1:
        with open(file_path + file_name, encoding="utf8") as opened_file:
            glove_file = opened_file.readlines()
    else:
        print("Glove file not found, load glove file (if lang=tr vectors.txt, "
              "else glove.6B.100d.txt)")
    return glove_file


def get_embedding_dictionary(review_lines):
    """
    This function is creating embedding dictionary with word2vec_file,
    glove_file or fasttext file.
    :param review_lines:list
        For each review list of words
    :return:dict
    It returns embedding dictionary which include word and vector_dimensions
    """
    embeddings_dictionary = {}
    if EMBEDDING_MODEL == "word2vec":
        embedding_file = word2vec_file_load(review_lines)
    elif EMBEDDING_MODEL == "glove":
        embedding_file = glove_file_load()
    else:
        embedding_file = fasttext_file_load(review_lines)

    for line in tqdm(embedding_file):
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    return embeddings_dictionary


def get_embedding_matrix(review_lines, num_words, word_index):
    """
    This function is creating embedding matrix with word2vec,
    glove or fasttext embedding dictionary
    :param review_lines:list
        For each review list of words
    :param num_words:int
        Total number of words
    :param word_index:dict
        Word dictionary which include words and indexes
    :return:ndarray
        It returns embedding matrix of word2vec,
        glove or fasttext
    """
    if EMBEDDING_MODEL == "glove" and LANG == "tr":
        embedding_matrix = np.zeros((num_words, 300))
    else:
        embedding_matrix = np.zeros((num_words, 100))

    embedding_dictionary = get_embedding_dictionary(review_lines)
    for word, index in tqdm(word_index.items()):
        if index > num_words:
            continue
        embedding_vector = embedding_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix
