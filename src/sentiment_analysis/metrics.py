"""This module is using for getting evaluation metrics"""
import tensorflow.python.keras.backend as keras_b
import tensorflow as tf


def f1_score(y_true, y_predict):
    """
    This function is using for calculating f1 score with y_true and y_predicted
    :param y_true: ndarray
    The tensor y_true is the true data (or target, ground truth) you pass to the fit method.
    It's a conversion of the numpy array y_train into a tensor
    :param y_predict:ndarray
    The tensor y_predicted is the data predicted (calculated, output) by your model.
    :return:
    """
    y_predict = keras_b.round(y_predict)
    true_positive = keras_b.sum(keras_b.cast(y_true * y_predict, 'float'), axis=0)
    false_positive = keras_b.sum(keras_b.cast((1 - y_true) * y_predict, 'float'), axis=0)
    false_negative = keras_b.sum(keras_b.cast(y_true * (1 - y_predict), 'float'), axis=0)
    precision_score = true_positive / (true_positive + false_positive + keras_b.epsilon())
    recall_score = true_positive / (true_positive + false_negative + keras_b.epsilon())
    f1_form = 2 * precision_score * recall_score / (precision_score + recall_score + keras_b.epsilon())
    f1_result = tf.where(tf.math.is_nan(f1_form), tf.zeros_like(f1_form), f1_form)
    return keras_b.mean(f1_result)


def precision(y_true, y_predict):
    """
    This function is using for calculating precision with y_true and y_predicted
    :param y_true: ndarray
    The tensor y_true is the true data (or target, ground truth) you pass to the fit method.
    It's a conversion of the numpy array y_train into a tensor
    :param y_predict:ndarray
    The tensor y_predicted is the data predicted (calculated, output) by your model.
    :return:
    """
    y_predict = keras_b.round(y_predict)
    true_positive = keras_b.sum(keras_b.cast(y_true * y_predict, 'float'), axis=0)
    false_positive = keras_b.sum(keras_b.cast((1 - y_true) * y_predict, 'float'), axis=0)
    precision_score = true_positive / (true_positive + false_positive + keras_b.epsilon())
    return keras_b.mean(precision_score)


def recall(y_true, y_predict):
    """
    This function is using for calculating recall with y_true and y_predicted
    :param y_true: ndarray
    The tensor y_true is the true data (or target, ground truth) you pass to the fit method.
    It's a conversion of the numpy array y_train into a tensor
    :param y_predict:ndarray
    The tensor y_predicted is the data predicted (calculated, output) by your model.
    :return:
        """
    y_predict = keras_b.round(y_predict)
    true_positive = keras_b.sum(keras_b.cast(y_true * y_predict, 'float'), axis=0)
    false_negative = keras_b.sum(keras_b.cast(y_true * (1 - y_predict), 'float'), axis=0)
    recall_score = true_positive / (true_positive + false_negative + keras_b.epsilon())
    return keras_b.mean(recall_score)
