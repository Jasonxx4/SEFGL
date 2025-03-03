from federatedscope.register import register_metric
from sklearn.metrics import balanced_accuracy_score
import numpy as np
METRIC_NAME = 'bac'


def bac(ctx,y_true,y_pred, **kwargs):
    y_true=y_true.flatten()
    y_pred = y_pred.flatten()
    bac = balanced_accuracy_score(y_true,y_pred)
    return bac


def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = bac
        return METRIC_NAME, metric_builder, the_larger_the_better


register_metric(METRIC_NAME, call_my_metric)
