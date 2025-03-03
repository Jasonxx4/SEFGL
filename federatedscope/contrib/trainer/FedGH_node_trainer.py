from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, MODE
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.message import Message
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
import torch.nn as nn
import copy
import logging
import torch
import datetime
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from collections import OrderedDict, defaultdict


class FedGH_Node_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedGH_Node_Trainer, self).__init__(model, data, device, config,
                                                 only_for_eval, monitor)

        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto,
                                    "on_fit_end")

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        split_mask = '{}_mask'.format(ctx.cur_split)

        label = batch.y[batch[split_mask]]
        pred_all, _ = ctx.model(batch)
        pred = pred_all[batch[split_mask]]

        ctx.batch_size = torch.sum(batch[split_mask]).item()
        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)

    def _hook_on_fit_end_agg_local_proto(self, ctx):
        # reset dataloader
        ctx.train_loader.reset()

        # collect local NN parameters
        reps_dict = defaultdict(list)
        agg_local_protos = dict()

        for batch_idx in range(ctx.num_train_batch):
            batch = next(ctx.train_loader).to(ctx.device)
            split_mask = '{}_mask'.format(ctx.cur_split)

            label = batch.y[batch[split_mask]]
            _, reps_all = ctx.model(batch)
            reps = reps_all[batch[split_mask]]

            owned_classes = label.unique()
            for cls in owned_classes:
                filted_reps = reps[label == cls].detach()
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

        training_begin_time = datetime.datetime.now()
        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)
        training_end_time = datetime.datetime.now()
        training_time = training_end_time-training_begin_time
        self.ctx.monitor.track_training_time(training_time)  # 记录每次本地训练的训练时间
        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_local_protos


def call_my_trainer(trainer_type):
    if trainer_type == 'fedgh_node_trainer':
        trainer_builder = FedGH_Node_Trainer
        return trainer_builder


register_trainer('fedgh_node_trainer', call_my_trainer)
