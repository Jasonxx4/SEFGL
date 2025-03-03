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
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your trainer here.
class FPL_Node_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FPL_Node_Trainer, self).__init__(model, data, device, config,
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

        ##########聚类proto在本地算法
        if len(ctx.global_protos) != 0:
            all_global_protos_keys = np.array(list(ctx.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = ctx.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(ctx.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f]  # 所有的proto
            mean_f = [item.detach() for item in mean_f]  # 每个类一个平均proto
        if len(ctx.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            i = 0
            loss2 = None
            for label in labels:
                if label.item() in ctx.global_protos.keys():
                    reps_now = reps[i].unsqueeze(0)
                    loss_instance = self.hierarchical_info_loss(reps_now, label, all_f, mean_f, all_global_protos_keys,ctx)
                    if loss2 is None:
                        loss2 = loss_instance
                    else:
                        loss2 += loss_instance
                i += 1
            loss2 = loss2 / i
        loss2 = loss2

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


    def hierarchical_info_loss(self,f_now, label, all_f, mean_f, all_global_protos_keys,ctx):
        for i, value in enumerate(all_global_protos_keys):
            if value == label.item():
                f_pos = all_f[i].to(ctx.device)
                mean_f_pos = mean_f[i].to(ctx.device)
        indices2 = [i for i, value in enumerate(all_global_protos_keys) if value != label.item()]
        f_neg = []
        for i in indices2:
            f_neg.append(all_f[i])
        f_neg = torch.cat(f_neg).to(ctx.device)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg, ctx.device)


        mean_f_pos = mean_f_pos.view(1, -1)
        # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
        # mean_f_neg = mean_f_neg.view(9, -1)

        loss_mse = nn.MSELoss()
        cu_info_loss = loss_mse(f_now, mean_f_pos)

        hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss

    def calculate_infonce(self,f_now, f_pos, f_neg,device):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self._cfg.fedproto.infoNCET
        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss


def call_my_trainer(trainer_type):
    if trainer_type == 'fpl_trainer':
        trainer_builder = FPL_Node_Trainer
        return trainer_builder


register_trainer('fpl_trainer', call_my_trainer)
