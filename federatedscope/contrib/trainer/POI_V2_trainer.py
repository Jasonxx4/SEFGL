from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
from typing import Type
import torch
import torch.nn as nn
import copy
import logging
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, defaultdict
from federatedscope.model_heterogeneity.SFL_methods.POI.graph_generator import PGG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Build your trainer here.
"""
2023.8.27
核心思想：
用每个class的全局原型和每个样本的reps算相似度，把这个相似度作为logits和标签计算CE_loss---或者用这个logits和原本的logits计算KL散度
用原本的loss+这个新计算出来的CE_LOss作为本地的总损失

"""


class POIV2_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(POIV2_Trainer, self).__init__(model, data, device, config,
                                            only_for_eval, monitor)

        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto,
                                    "on_fit_end")

        self.register_hook_in_train(self._hook_on_epoch_start_for_proto,
                                    "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_proto,
                                   "on_epoch_start")

        # 在client每次本地训练之前调用；用来定义额外的用来优化selr.ctx.W的optimizer以及一些可视化用的变量
        self.register_hook_in_train(new_hook=self._hook_on_fit_start_clean,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_eval(new_hook=self._hook_on_fit_start_clean,
                                   trigger='on_fit_start',
                                   insert_pos=-1)

        self.task = config.MHFL.task
        self.num_classes = config.model.num_classes

        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        self.tau = config.MHFL.tau
    def _hook_on_fit_start_clean(self, ctx):
        ctx.global_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)  # 保存global model的输出结果用以验证

    def _hook_on_batch_forward(self, ctx):
        # only support node-level task
        batch = ctx.data_batch.to(ctx.device)
        split_mask = batch[f'{ctx.cur_split}_mask']  # train_mask,val_mask,test_mask

        pred_all, reps_all = ctx.model(batch)
        pred, reps = pred_all[split_mask], reps_all[split_mask]
        labels = batch.y[split_mask]

        loss1 = ctx.criterion(pred, labels)
        if len(ctx.global_protos) == 0:
            loss2 = kl_loss = 0 * loss1
            similarity = torch.zeros_like(pred,device=pred.device)
        else:
            global_protos = torch.stack(list(ctx.global_protos.values()))  # (num_class, feature_dim)
            reps= F.normalize(reps, dim=1)
            similarity = torch.matmul(reps, global_protos.T) / self.tau

            # loss2 = ctx.criterion(similarity, labels)
            # kl_loss = self.KL_Loss(self.LogSoftmax(pred), self.Softmax(similarity.detach()))
        loss = loss1 #+kl_loss
        logger.info(
            f'client#{self.ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} '
            f'\t loss1:{loss1} \t loos2:{0.0},\ttotal_loss:{loss}')

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(similarity, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(reps.detach().cpu())
        ####


    def update(self, global_proto, strict=False):
        self.ctx.global_protos = global_proto

    def _hook_on_epoch_start_for_proto(self, ctx):
        """定义一些fedproto需要用到的全局变量"""
        ctx.agg_protos_label = CtxVar(dict(), LIFECYCLE.ROUTINE)
        ctx.ys_feature = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.new_data = None

    def _hook_on_fit_end_agg_local_proto(self, ctx):
        reps_dict = defaultdict(list)
        agg_local_protos = dict()

        ctx.train_loader.reset()

        for batch_idx in range(ctx.num_train_batch):
            batch = next(ctx.train_loader)
            if self.task == "node":
                batch.to(ctx.device)
                split_mask = '{}_mask'.format(ctx.cur_split)
                labels = batch.y[batch[split_mask]]
                _, reps_all = ctx.model(batch)
                reps = reps_all[batch[split_mask]]
            else:
                images, labels = [_.to(ctx.device) for _ in batch]
                _, reps = ctx.model(images)

            owned_classes = labels.unique()
            for cls in owned_classes:
                filted_reps = reps[labels == cls].detach()
                reps_dict[cls.item()].append(filted_reps)

        for cls, protos in reps_dict.items():
            mean_proto = torch.cat(protos).mean(dim=0)
            agg_local_protos[cls] = mean_proto

        ctx.agg_local_protos = agg_local_protos

        # t-she可视化用
        if ctx.cfg.vis_embedding:
            ctx.node_emb_all = reps_all.clone().detach()
            ctx.node_labels = batch.y.clone().detach()

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_local_protos


def call_my_trainer(trainer_type):
    if trainer_type == 'poiv2_trainer':
        trainer_builder = POIV2_Trainer
        return trainer_builder


register_trainer('poiv2_trainer', call_my_trainer)
