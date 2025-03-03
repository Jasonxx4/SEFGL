from federatedscope.register import register_metric
from sklearn.metrics import precision_score, f1_score
import numpy as np
METRIC_NAME = 'f1score'


def f1score(ctx,y_true,y_pred, **kwargs):
    y_true=y_true.flatten()
    y_pred = y_pred.flatten()
    f1 = f1_score(y_true,y_pred,average="weighted")
    return f1


def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = f1score
        return METRIC_NAME, metric_builder, the_larger_the_better


register_metric(METRIC_NAME, call_my_metric)
