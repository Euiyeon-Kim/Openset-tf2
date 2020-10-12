import os

import numpy as np
import tensorflow as tf
from keras.layers import ReLU, LeakyReLU, Activation


def get_ms():
    if os.name == 'nt':
        cross_device_ops = tf.distribute.ReductionToOneDevice()
    else:
        cross_device_ops = tf.distribute.NcclAllReduce()

    return tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)


def get_activation(activation):
    if activation is None or activation == 'linear':
        return lambda x: x
    elif activation == "relu":
        return ReLU()
    elif activation == "lrelu":
        return LeakyReLU(0.2)
    elif activation == "tanh":
        return Activation("tanh")
    elif activation == "sigmoid":
        return Activation("sigmoid")
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def normalize_img(img, label):
    return tf.cast(img, tf.float32) / 127.5 - 1, label


def denormalize_img(img):
    return ((img + 1.) * 127.5).astype(np.uint8)


def resize_img(img, label, size):
    return tf.image.resize(img, size), label


def openset_acc(pred, label):
    # TP(cor closed)  FP (incor closed)
    # TN(cor open)    FN (incor open)
    confusion_mat = np.zeros((2, 2))
    total_acc = np.sum(pred == label) / len(label)

    real_openset = tf.cast(label == -1, "int64")

    if np.sum(real_openset) == 0:
        pred_as_openset = np.array([0])
        open_acc = 0
    else:
        pred_as_openset = tf.cast(pred == -1, "int64")
        openset_preds = np.logical_and(real_openset, pred_as_openset)
        confusion_mat[1][1] = np.sum(real_openset != openset_preds)
        confusion_mat[1][0] = np.sum(real_openset) - confusion_mat[1][1]
        open_acc = np.sum(np.logical_and(pred_as_openset, real_openset)) / np.sum(real_openset)

    pred[pred == -1] = -2
    true_pos = np.sum(pred == label)
    close_acc = true_pos / (len(label) - np.sum(real_openset))
    confusion_mat[0][0] = true_pos
    confusion_mat[0][1] = len(label) - np.sum(real_openset) - true_pos

    # if confusion_mats[0][0] + confusion_mats[1][1] != 0:
    #     val_presision = confusion_mats[0][0] / (confusion_mats[0][0] + confusion_mats[0][1])
    #     val_recall = confusion_mats[0][0] / (confusion_mats[0][0] + confusion_mats[1][1])
    #     val_f1 = 2 * ((val_presision * val_recall) / (val_presision + val_recall + 1e-12))

    return total_acc, open_acc, close_acc, np.sum(real_openset), np.sum(pred_as_openset), confusion_mat


def closeset_acc(pred, label):
    return np.sum(pred == label) / len(label)