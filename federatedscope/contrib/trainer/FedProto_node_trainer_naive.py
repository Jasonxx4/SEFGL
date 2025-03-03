from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, MODE
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.message import Message
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
import torch.nn as nn
import copy
import logging
import torch
import datetime
from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# prototype原代码
# Build your trainer here.
class FedProto_Node_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedProto_Node_Trainer, self).__init__(model, data, device, config,
                                                    only_for_eval, monitor)
        self.loss_mse = nn.MSELoss()
        self.proto_weight = self.ctx.cfg.fedproto.proto_weight
        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto,
                                    "on_fit_end")

        self.register_hook_in_train(self._hook_on_epoch_start_for_proto,
                                    "on_fit_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_proto,
                                   "on_fit_start")

        self.task = config.MHFL.task

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        split_mask = batch[f'{ctx.cur_split}_mask']

        labels = batch.y[split_mask]
        owned_classes = labels.unique()

        pred_all, reps_all = ctx.model(batch)
        pred = pred_all[split_mask]
        reps = reps_all[split_mask]

        loss1 = ctx.criterion(pred, labels)

        loss2 = 0 * loss1 #TODO: 测试用，记得删除
        if len(ctx.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            proto_new = copy.deepcopy(reps.data)  # TODO: .data会有问题吗，是不是应该用detach()？
            for cls in owned_classes:
                if cls.item() in ctx.global_protos.keys():
                    proto_new[labels == cls] = ctx.global_protos[cls.item()]
            loss2 = self.loss_mse(reps, proto_new)

        # if len(ctx.global_protos) != 0:
        #     global_protos = torch.stack(list(ctx.global_protos.values())).detach()
        #     similarity = torch.matmul(reps,  global_protos.T)
        # else:
        #     similarity=pred
        loss = loss1 + loss2 * self.proto_weight

        if ctx.cfg.fedproto.show_verbose:
            logger.info(
                f'client#{self.ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} \t CE_loss:{loss1}'
                f'\t proto_loss:{loss2},\t total_loss:{loss}')

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(reps.detach().cpu())
        ####

    def update(self, global_proto, strict=False):
        self.ctx.global_protos = global_proto

    def _hook_on_epoch_start_for_proto(self, ctx):
        """定义一些fedproto需要用到的全局变量"""
        ctx.ys_feature = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_fit_end_agg_local_proto(self, ctx):
        reps_dict = defaultdict(list)
        agg_local_protos = dict()
        ctx.train_loader.reset()

        for batch_idx in range(ctx.num_train_batch):
            batch = next(ctx.train_loader).to(ctx.device)
            split_mask = batch['train_mask']
            labels = batch.y[split_mask]
            _, reps_all = ctx.model(batch)
            reps = reps_all[split_mask]


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

        training_begin_time = datetime.datetime.now()

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        training_end_time = datetime.datetime.now()
        training_time = training_end_time-training_begin_time
        self.ctx.monitor.track_training_time(training_time)  # 记录每次本地训练的训练时间

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_local_protos


def call_my_trainer(trainer_type):
    if trainer_type == 'fedproto_node_trainer_naive':
        trainer_builder = FedProto_Node_Trainer
        return trainer_builder


register_trainer('fedproto_node_trainer_naive', call_my_trainer)
