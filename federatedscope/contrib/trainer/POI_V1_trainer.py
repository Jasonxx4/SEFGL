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
import numpy as np
from collections import OrderedDict, defaultdict
from federatedscope.model_heterogeneity.SFL_methods.POI.graph_generator import PGG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Build your trainer here.
"""
核心思想：每个client利用全局模型来生成（或替换）本地节点/图数据，达到数据增强的目的
"""


class POIV1_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(POIV1_Trainer, self).__init__(model, data, device, config,
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

        # TODO: 每次本地更新完之后是否要用 _hook_on_fit_end_free_cuda把W放到cpu上？
        # self.register_hook_in_train(new_hook=self._hook_on_fit_end_free_cuda,
        #                             trigger="on_fit_end",
        #                             insert_pos=-1)
        #
        # self.register_hook_in_eval(new_hook=self._hook_on_fit_end_free_cuda,
        #                            trigger="on_fit_end",
        #                            insert_pos=-1)

        self.task = config.MHFL.task
        self.num_classes = config.model.num_classes

        feature_dim = config.model.hidden
        input_dim = data['data'].x.shape[-1]
        self.ctx.W = nn.Linear(feature_dim, input_dim, bias=False).to(device)
        self.transform = PGG(data,alpha=1.0)

        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def _hook_on_fit_start_clean(self, ctx):
        # Set optimizer for ctx.W additionally.
        ctx.W.to(ctx.device)
        if ctx.cur_mode in [MODE.TRAIN]:
            ctx.W.train()
            ctx.W_optimizer = get_optimizer(ctx.W, **ctx.cfg.train.optimizer)
        elif ctx.cur_mode in [MODE.VAL, MODE.TEST]:
            ctx.W.eval()

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
            lam = 1.0
        else:
            global_protos = torch.stack(list(ctx.global_protos.values()))  # (num_class, feature_dim)
            rec_protos = ctx.W(global_protos)
            new_data, lam = self.transform(rec_protos, self.num_classes)
            pred_all_new, _ = ctx.model(new_data)
            pred_new = pred_all_new[split_mask]

            # 利用原型生成新的图数据
            loss2 = ctx.criterion(pred_new, labels)

            # kl_loss = self.KL_Loss(self.LogSoftmax(pred), self.Softmax(pred_new.detach()))

        loss = lam * loss1 + (1 - lam)*loss2  # + kl_loss
        # loss=loss1+kl_loss
        # logger.info(
        #     f'client#{self.ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} \t CE:{loss1}'
        #     f'\t ce2:{loss2},\t lam:{lam}\t kl:{kl_loss} \ttotal_loss:{loss}')

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(reps.detach().cpu())
        if len(ctx.global_protos) != 0:
            ctx.new_data = new_data.detach().cpu()
        ####

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.W_optimizer.zero_grad()

        ctx.loss_batch.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.W.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()
        # ctx.W_optimizer.step()

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
    if trainer_type == 'poiv1_trainer':
        trainer_builder = POIV1_Trainer
        return trainer_builder


register_trainer('poiv1_trainer', call_my_trainer)
