from federatedscope.register import register_metric
import torch
import numpy as np
METRIC_NAME = 'global_model_acc'


def global_model_acc(ctx, y_true, **kwargs):
    labels = y_true
    y_prob = torch.Tensor(torch.cat(ctx.global_ys_prob)).to(ctx.device)
    if torch is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()
    if y_prob.ndim == 2:
        y_prob = np.expand_dims(y_prob, axis=-1)

    y_pred = np.argmax(y_prob, axis=1)

    acc_list = []


    for i in range(labels.shape[1]):
        is_labeled = labels[:, i] == labels[:, i]
        correct = labels[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)

    return acc

def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = global_model_acc
        return METRIC_NAME, metric_builder, the_larger_the_better

register_metric(METRIC_NAME, call_my_metric)
