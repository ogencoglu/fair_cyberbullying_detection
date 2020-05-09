'''
metrics for binary classification performance and fairness
'''

__author__ = 'Oguzhan Gencoglu'

import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef)


def accuracy(labels, predictions):
    return accuracy_score(labels, predictions)


def error_rate(labels, predictions):
    return 1 - accuracy(labels, predictions)


def precision(labels, predictions):
    return precision_score(labels, predictions)


def recall(labels, predictions):
    return recall_score(labels, predictions)


def auc(labels, probabilities):
    return roc_auc_score(labels, probabilities, average='micro')


def f1(labels, predictions):
    return f1_score(labels, predictions)


def mcc(labels, predictions):
    return matthews_corrcoef(labels, predictions)


def false_negative_rate(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return fn / (fn + tp)


def false_positive_rate(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return fp / (fp + tn)


def group_mccs(labels, predictions, groups, round=4):
    mccs = []
    for i in range(groups.shape[1]):
        labels_ma = np.ma.array(labels, mask=groups[:, i])
        labels_filtered = labels_ma[labels_ma.mask].data

        predictions_ma = np.ma.array(predictions, mask=groups[:, i])
        predictions_filtered = predictions_ma[predictions_ma.mask].data
        mccs.append(mcc(labels_filtered, predictions_filtered))

    return np.round(mccs, round)


def group_false_negative_rates(labels, predictions, groups, round=4):
    fnrs = []
    for i in range(groups.shape[1]):
        labels_ma = np.ma.array(labels, mask=groups[:, i])
        labels_filtered = labels_ma[labels_ma.mask].data

        predictions_ma = np.ma.array(predictions, mask=groups[:, i])
        predictions_filtered = predictions_ma[predictions_ma.mask].data
        fnrs.append(false_negative_rate(labels_filtered, predictions_filtered))

    return np.round(fnrs, round)


def group_false_positive_rates(labels, predictions, groups, round=4):
    fprs = []
    for i in range(groups.shape[1]):
        labels_ma = np.ma.array(labels, mask=groups[:, i])
        labels_filtered = labels_ma[labels_ma.mask].data

        predictions_ma = np.ma.array(predictions, mask=groups[:, i])
        predictions_filtered = predictions_ma[predictions_ma.mask].data
        fprs.append(false_positive_rate(labels_filtered, predictions_filtered))

    return np.round(fprs, round)


def group_recalls(labels, predictions, groups, round=4):
    recalls = []
    for i in range(groups.shape[1]):
        labels_ma = np.ma.array(labels, mask=groups[:, i])
        labels_filtered = labels_ma[labels_ma.mask].data

        predictions_ma = np.ma.array(predictions, mask=groups[:, i])
        predictions_filtered = predictions_ma[predictions_ma.mask].data
        recalls.append(recall(labels_filtered, predictions_filtered))

    return np.round(recalls, round)


def group_precisions(labels, predictions, groups, round=4):
    precisions = []
    for i in range(groups.shape[1]):
        labels_ma = np.ma.array(labels, mask=groups[:, i])
        labels_filtered = labels_ma[labels_ma.mask].data

        predictions_ma = np.ma.array(predictions, mask=groups[:, i])
        predictions_filtered = predictions_ma[predictions_ma.mask].data
        precisions.append(precision(labels_filtered, predictions_filtered))

    return np.round(precisions, round)


def false_positive_equality_diff(labels, predictions, groups, round=4):
    group_fprs = group_false_positive_rates(labels, predictions, groups, 6)
    fpr = false_positive_rate(labels, predictions)
    fped = np.sum(np.abs(fpr - group_fprs))

    return np.round(fped, round)


def false_negative_equality_diff(labels, predictions, groups, round=4):
    group_fnrs = group_false_negative_rates(labels, predictions, groups, 6)
    fnr = false_negative_rate(labels, predictions)
    fned = np.sum(np.abs(fnr - group_fnrs))

    return np.round(fned, round)


def total_equality_diff(labels, predictions, groups, round=4):
    fped = false_positive_equality_diff(labels, predictions, groups, round=10)
    fned = false_negative_equality_diff(labels, predictions, groups, round=10)

    return np.round(fped + fned, round)
