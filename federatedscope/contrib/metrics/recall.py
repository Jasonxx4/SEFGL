from federatedscope.register import register_metric
from sklearn.metrics import balanced_accuracy_score
import numpy as np
METRIC_NAME = 'bacc'


def bacc(ctx,y_true,y_pred, **kwargs):
    y_true=y_true.flatten()
    y_pred = y_pred.flatten()
    bacc = balanced_accuracy_score(y_true,y_pred)
    return bacc


def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = bacc
        return METRIC_NAME, metric_builder, the_larger_the_better


register_metric(METRIC_NAME, call_my_metric)
