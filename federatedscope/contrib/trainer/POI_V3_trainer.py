from federatedscope.register import register_trainer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
import torch
import torch.nn as nn
import copy
import logging
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, defaultdict
from torch_cluster import knn_graph
from federatedscope.model_heterogeneity.SFL_methods.POI.graph_generator import PGG
from federatedscope.contrib.model.label_prop import LabelPropagation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
2023.8.28
核心思想：
用接收到的类原型做原型传播，生成视图-->用生成的视图训练/纠正本地模型 or 纠正本地模型的预测结果

观察结果：

"""


# Build your trainer here.
class POIV3_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(POIV3_Trainer, self).__init__(model, data, device, config,
                                            only_for_eval, monitor)

        self.task = config.MHFL.task
        self.num_classes = config.model.num_classes

        self.ctx.global_model = LabelPropagation(num_layers=config.poi.LP_layer, alpha=config.poi.LP_alpha)

        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        self.tau = config.MHFL.tau

        self.register_our_hook()

        # self.weight_private = 0.5
        self.ctx.weight_private = torch.tensor([0.5], requires_grad=True, device=device)
        self.ctx.optimizer_learned_weight_for_inference = torch.optim.Adam([self.ctx.weight_private], lr=1e-2)

        self.use_knn = config.poi.use_knn
        if self.use_knn:
            x = data.train_data[0].x
            batch = torch.zeros((len(x)))
            edge_index = knn_graph(x, k=5, batch=batch, loop=False)
            self.ctx.knn_edge_index = edge_index

    def _hook_on_batch_forward(self, ctx):
        # only support node-level task

        batch = ctx.data_batch.to(ctx.device)
        split_mask = batch[f'{ctx.cur_split}_mask']  # train_mask,val_mask,test_mask

        pred_all, reps_all = ctx.model(batch)
        pred, reps = pred_all[split_mask], reps_all[split_mask]
        labels = batch.y[split_mask]

        loss1 = ctx.criterion(pred, labels)
        if len(ctx.global_protos) == 0:
            loss = loss1
            PL_reps = torch.zeros_like(reps, device=reps.device)
            PL_pred = pred_ensemble_adaptive = torch.zeros_like(pred, device=pred.device)
            # loss2 = kl_loss = 0 * loss1
            # pred_ensemble_adaptive=PL_pred = torch.zeros_like(pred, device=pred.device)
            # similarity = torch.zeros_like(pred, device=pred.device)
        else:
            global_protos = torch.stack(list(ctx.global_protos.values()))  # (num_class, feature_dim)
            # global_protos=F.normalize(global_protos, dim=1)

            proto_lable_init = initialize_prototype_label(batch['train_mask'], global_protos, reps_all.detach(),
                                                          batch.y)

            if self.use_knn:
                PL_reps = ctx.global_model(y=proto_lable_init, edge_index=ctx.knn_edge_index.to(ctx.device),
                                           train_mask=batch['train_mask'])
            else:
                PL_reps = ctx.global_model(y=proto_lable_init, edge_index=batch.edge_index,
                                           train_mask=batch['train_mask'])
            PL_pred = ctx.model.FC(PL_reps)[split_mask]
            pred_ensemble_adaptive = (ctx.weight_private * pred + (1 - ctx.weight_private) * PL_pred)
            loss2 = ctx.criterion(PL_pred, labels)
            # kl_loss = self.KL_Loss(self.LogSoftmax(pred), self.Softmax(PL_pred.detach()))
            loss = loss1 + loss2
            # loss3 = ctx.criterion(pred_ensemble_adaptive, labels)
            # reps = F.normalize(reps, dim=1)
            # similarity = torch.matmul(reps, global_protos.T) / self.tau
            # kl_loss = self.KL_Loss(self.LogSoftmax(pred), self.Softmax(PL_pred.detach()))
            # loss = loss1+ loss2
            # loss = loss1 + loss2 + loss3

        # logger.info(
        #     f'client#{self.ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} '
        #     f'\t loss1:{loss1} \t loos2:{loss2},\ttotal_loss:{loss}')

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(reps.detach().cpu())
        ctx.global_ys_prob.append(PL_pred.detach().cpu())
        ctx.ensemble_ys_prob.append(pred_ensemble_adaptive.detach().cpu())

        ctx.PL_node_emb_all = PL_reps.detach().clone()
        ####

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.optimizer_learned_weight_for_inference.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.weight_private,
                                           ctx.grad_clip)

        ctx.optimizer.step()
        ctx.optimizer_learned_weight_for_inference.step()
        # logger.info(f'当前weight:{ctx.weight_private}')
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def update(self, global_proto, strict=False):
        self.ctx.global_protos = global_proto

    def register_our_hook(self):
        # 训练结束聚合本地原型
        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto, "on_fit_end")

        # 定义/初始化要用到的中间变量
        self.register_hook_in_train(self._hook_on_epoch_start_for_variable_definition, "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_variable_definition, "on_epoch_start")

        # 在client每次本地训练之前调用，用来初始化ctx.global_ys_prob；这个变量用于保存global_model的输出结果
        self.register_hook_in_train(new_hook=self._hook_on_fit_start_clean, trigger='on_fit_start', insert_pos=-1)
        self.register_hook_in_eval(new_hook=self._hook_on_fit_start_clean, trigger='on_fit_start', insert_pos=-1)

        # # 用来在本地训练/推理结束时，额外地评估global model的ACC
        # self.register_hook_in_train(new_hook=self._hook_on_fit_end_eval_global_model, trigger="on_fit_end",
        #                             insert_pos=-1)
        # self.register_hook_in_eval(new_hook=self._hook_on_fit_end_eval_global_model, trigger="on_fit_end",
        #                            insert_pos=-1)

    def _hook_on_epoch_start_for_variable_definition(self, ctx):
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

    def _hook_on_fit_start_clean(self, ctx):
        ctx.global_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)  # 保存global model的输出结果用以验证
        ctx.ensemble_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)  # 保存global model的输出结果用以验证

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_local_protos


def initialize_prototype_label(train_mask, global_protos, reps_all, labels_all):
    # labels_init = torch.ones_like(reps_all) / len(reps_all)  # (N_all, feature_dim)
    labels_init = torch.zeros_like(reps_all)  # (N_all, feature_dim)
    labels_init[train_mask] = global_protos[labels_all[train_mask]]  # (N_train, feature_dim)
    # labels_init[idx_train] = labels_one_hot[idx_train]
    return labels_init


def eval_acc(y_true, y_pred, **kwargs):
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def call_my_trainer(trainer_type):
    if trainer_type == 'poiv3_trainer':
        trainer_builder = POIV3_Trainer
        return trainer_builder


register_trainer('poiv3_trainer', call_my_trainer)
