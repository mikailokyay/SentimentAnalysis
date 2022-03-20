import keras.backend as k
import tensorflow as tf


def f1(y_true, y_predict):
    y_predict = k.round(y_predict)
    tp = k.sum(k.cast(y_true * y_predict, 'float'), axis=0)
    fp = k.sum(k.cast((1 - y_true) * y_predict, 'float'), axis=0)
    fn = k.sum(k.cast(y_true * (1 - y_predict), 'float'), axis=0)
    p = tp / (tp + fp + k.epsilon())
    r = tp / (tp + fn + k.epsilon())
    f1_form = 2 * p * r / (p + r + k.epsilon())
    f1_score = tf.where(tf.math.is_nan(f1_form), tf.zeros_like(f1_form), f1_form)
    return k.mean(f1_score)


def precision(y_true, y_predict):
    y_predict = k.round(y_predict)
    tp = k.sum(k.cast(y_true * y_predict, 'float'), axis=0)
    fp = k.sum(k.cast((1 - y_true) * y_predict, 'float'), axis=0)
    precision_score = tp / (tp + fp + k.epsilon())
    return k.mean(precision_score)


def recall(y_true, y_predict):
    y_predict = k.round(y_predict)
    tp = k.sum(k.cast(y_true * y_predict, 'float'), axis=0)
    fn = k.sum(k.cast(y_true * (1 - y_predict), 'float'), axis=0)
    recall_score = tp / (tp + fn + k.epsilon())
    return k.mean(recall_score)
