from federatedscope.register import register_metric
from sklearn.metrics import precision_score
import numpy as np
METRIC_NAME = 'precision'


def precision(ctx,y_true,y_pred, **kwargs):
    y_true=y_true.flatten()
    y_pred = y_pred.flatten()
    pre = precision_score(y_true,y_pred,average="micro",zero_division=1)
    return pre


def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = precision
        return METRIC_NAME, metric_builder, the_larger_the_better


register_metric(METRIC_NAME, call_my_metric)
