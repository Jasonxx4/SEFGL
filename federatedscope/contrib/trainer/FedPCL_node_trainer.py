from federatedscope.register import register_trainer
import torch
import torch.nn as nn
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar
from federatedscope.contrib.loss.MHFL_losses import ConLoss
from federatedscope.contrib.loss.fedpcl_prototype_loss import ProtoConloss
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
import logging
from collections import OrderedDict, defaultdict
import numpy as np
import copy
import torch.nn.functional as F
import datetime
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def agg_local_proto(protos, num_classes=62):
    """
    Average the protos for each local user
    """
    agg_protos = {}
    for [label, reps] in protos.items():
        agg_protos[label] = torch.mean(reps, dim=0).data

    # 对于未见类，生成全zero tensor上传server
    for label in range(num_classes):
        if label not in agg_protos:
            agg_protos[label] = torch.zeros(list(agg_protos.values())[0].shape[0],
                                            device=list(agg_protos.values())[0].device)

    return agg_protos


class FedPCL_Node_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedPCL_Node_Trainer, self).__init__(model, data, device, config,
                                                  only_for_eval, monitor)
        self.loss_mse = nn.MSELoss()
        self.loss_CL = ConLoss(temperature=0.07)
        self.ProtoConloss = ProtoConloss(temperature=0.07)
        self.nll_loss = nn.NLLLoss().to(device)
        self.num_users = config.federate.client_num
        self.device = device
        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto,
                                    "on_fit_end", insert_pos=0)
        self.register_hook_in_train(self._hook_on_fit_start_init_additionaly,
                                    "on_fit_start")
        self.register_hook_in_eval(self._hook_on_fit_start_init_additionaly,
                                   "on_epoch_start")
        self.debug = config.fedpcl.debug
        self.ctx.local_proto_sets=[]
        self.num_class = config.model.num_classes
    def _hook_on_batch_forward(self, ctx):
        data = ctx.data_batch.to(ctx.device)
        split_mask = data[f'{ctx.cur_split}_mask']

        labels = data.y[split_mask]

        pred_all, reps_all = ctx.model(data)
        reps_all = F.normalize(reps_all, dim=1)  # aling with the original code
        pred, reps = pred_all[split_mask], reps_all[split_mask]

        loss_ce =ctx.criterion(pred, labels)
        # compute regularized loss term
        if len(ctx.local_proto_sets) == self.num_users:
            features = reps
            L_g = self.ProtoConloss(features, labels, ctx.global_protos, self.num_class)
            L_p= torch.tensor(0.0,device=ctx.device)
            for i in range(1, self.num_users + 1):
                for label in ctx.global_protos.keys():
                    if label not in ctx.local_proto_sets[i].keys():
                        ctx.local_proto_sets[i][label] = ctx.global_protos[label]
                L_p += self.ProtoConloss(features, labels, ctx.local_proto_sets[i], self.num_class)

            loss = loss_ce+ L_g+ L_p/self.num_users
        else:
            loss= loss_ce

        if ctx.cfg.fedpcl.show_verbose:
            logger.info(
                f'client#{self.ctx.client_ID}  '
                f'{ctx.cur_split}  '
                f'round:{ctx.cur_state} '
                f'\t global prototype-based loss: {L_g if "L_g" in locals()  else 0}'
                f'\t local prototype-based loss: {L_p if "L_p" in locals() else 0}'
                f'\t total loss:{loss}')

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(reps.detach().cpu())
        ####

    @torch.no_grad()
    def _hook_on_fit_end_agg_local_proto(self, ctx):
        reps_dict = dict()

        data=ctx.data.train_data[0].to(ctx.device)
        split_mask = data['train_mask']
        labels = data.y[split_mask]

        _, reps_all = ctx.model(data)
        reps_all= F.normalize(reps_all, dim=1)
        reps = reps_all[split_mask]
        owned_classes = labels.unique()
        for cls in owned_classes:
            filted_reps = reps[labels == cls].detach()
            reps_dict[cls.item()]=filted_reps

        ctx.agg_protos = agg_local_proto(reps_dict, num_classes=ctx.cfg.model.num_classes)

    def _hook_on_fit_start_init_additionaly(self, ctx):
        ctx.agg_protos_label = CtxVar(dict(), LIFECYCLE.ROUTINE)  # 每次本地训练之前，初始化agg_protos_label为空字典
        ctx.ys_feature = CtxVar([], LIFECYCLE.ROUTINE)  # 保存每个样本的representation，基于local prototype 计算acc时会用到

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)
        training_begin_time = datetime.datetime.now()
        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)
        training_time = datetime.datetime.now() - training_begin_time
        self.ctx.monitor.track_training_time(training_time)  # 记录每次本地训练的训练时间
        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_protos


def call_my_torch_trainer(trainer_type):
    if trainer_type == 'fedpcl_node_trainer':
        trainer_builder = FedPCL_Node_Trainer
        return trainer_builder


register_trainer('fedpcl_node_trainer', call_my_torch_trainer)
