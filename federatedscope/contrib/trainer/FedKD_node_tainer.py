from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict
from typing import Type
import torch
import torch.nn as nn
import copy
import logging
import numpy as np
import datetime
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Build your trainer here.

class FedKD_node_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedKD_node_Trainer, self).__init__(model, data, device, config,
                                                 only_for_eval, monitor)
        '''
        self.model --> personalized local mentor model
        self.global_model --> shared mentee model
        '''
        global_hidden_dim = config.MHFL.global_model.hidden
        self.ctx.global_model = get_model(model_config=config.MHFL.global_model, local_data=data)
        # self.ctx.global_model.W = nn.Linear(global_hidden_dim, global_hidden_dim, bias=False).to(device)
        self.ctx.W = nn.Linear(global_hidden_dim, global_hidden_dim, bias=False).to(device)

        self.register_hook_in_train(new_hook=self._hook_on_fit_start_clean,
                                    trigger='on_fit_start',
                                    insert_pos=-1)

        self.register_hook_in_eval(new_hook=self._hook_on_fit_start_clean,
                                   trigger='on_fit_start',
                                   insert_pos=-1)

        # TODO: 注册_hook_on_batch_end_flop_count
        # TODO: 弄懂是否需要注册 _hook_on_fit_end_calibrate

        self.register_hook_in_train(new_hook=self._hook_on_fit_end_free_cuda,
                                    trigger="on_fit_end",
                                    insert_pos=-1)

        self.register_hook_in_eval(new_hook=self._hook_on_fit_end_free_cuda,
                                   trigger="on_fit_end",
                                   insert_pos=-1)

        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.mse_loss = nn.MSELoss()

    def get_model_para(self):
        """
        重写get_model_para(), 使其从基类的返回ctx.model的参数，变为返回ctx.global_model的参数
        trainer.train()会调用该函数，以获得更新好的本地模型
        """
        if self.cfg.federate.process_num > 1:
            return self._param_filter(self.ctx.global_model.state_dict())
        else:
            return self._param_filter(
                copy.deepcopy(self.ctx.global_model.state_dict()) if self.cfg.federate.
                share_local_model else copy.deepcopy(self.ctx.global_model.cpu().state_dict())
            )

    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
            修改被更新的模型为self.ctx.global_model。基类方法中更新的是self.ctx.model
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.ctx.global_model.state_dict().copy(),
                                        self._param_filter(model_parameters))
        self.ctx.global_model.load_state_dict(merged_param, strict=strict)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        split_mask = batch['{}_mask'.format(ctx.cur_split)]
        label = batch.y[split_mask]

        local_pred_all, local_rep_all = ctx.model(batch)
        global_pred_all, global_rep_all = ctx.global_model(batch)

        local_pred, local_rep = local_pred_all[split_mask], local_rep_all[split_mask]
        global_pred, global_rep = global_pred_all[split_mask], global_rep_all[split_mask]

        CE_local = ctx.criterion(local_pred, label)
        CE_global = ctx.criterion(global_pred, label)

        KL_local = self.KL_Loss(self.LogSoftmax(local_pred), self.Softmax(global_pred.detach())) / (
                CE_local + CE_global)
        KL_global = self.KL_Loss(self.LogSoftmax(global_pred), self.Softmax(local_pred.detach())) / (
                CE_local + CE_global)

        transform_hidden = self.ctx.W(global_rep)
        HL_local = self.mse_loss(local_rep, transform_hidden) / (CE_local + CE_global)  # hidden loss local
        HL_global = self.mse_loss(local_rep, transform_hidden) / (CE_local + CE_global)  # hidden loss mentee

        loss_local = CE_local + KL_local + HL_local
        loss_mentee = CE_global + KL_global + HL_global

        # 个性化模型的结果用来计算指标
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(local_pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_local, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

        # 记录全局模型的loss
        ctx.loss_batch_global = CtxVar(loss_mentee, LIFECYCLE.BATCH)

        # 记录全局模型的output
        ctx.global_ys_prob.append(global_pred.clone().detach().cpu().numpy())

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.global_optimizer.zero_grad()
        ctx.W_optimizer.zero_grad()

        ctx.loss_batch.backward(retain_graph=True)  # 等价于 loss_local.backward()
        ctx.loss_batch_global.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.global_model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.W.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()  # 更新local_model
        ctx.global_optimizer.step()  # 更新global_model
        ctx.W_optimizer.step()

    def _hook_on_fit_start_clean(self, ctx):
        # 保存global model的输出结果用以验证
        ctx.global_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

        # Set optimizer for global model additionally.
        ctx.global_model.to(ctx.device)
        if ctx.cur_mode in [MODE.TRAIN]:
            ctx.global_model.train()
            ctx.global_optimizer = get_optimizer(ctx.global_model,
                                                 **ctx.cfg.train.optimizer)
            ctx.W_optimizer = get_optimizer(ctx.W, **ctx.cfg.train.optimizer)
        elif ctx.cur_mode in [MODE.VAL, MODE.TEST]:
            ctx.global_model.eval()

    def _hook_on_fit_end_free_cuda(self, ctx):
        ctx.global_model.to(torch.device("cpu"))
        ctx.global_ys_prob = CtxVar(np.concatenate(ctx.global_ys_prob), LIFECYCLE.ROUTINE)
        y_true = ctx.ys_true
        y_prob = ctx.global_ys_prob
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=-1)
        if y_prob.ndim == 2:
            y_prob = np.expand_dims(y_prob, axis=-1)

        # if len(y_prob.shape) > len(y_true.shape):
        y_pred = np.argmax(y_prob, axis=1)

        acc = eval_acc(y_true, y_pred)
        logger.info(f'client#{self.ctx.client_ID} {ctx.cur_split} global_mentee_model acc :{acc}')
        # ctx.local_model.to(torch.device("cpu"))
    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        training_begin_time = datetime.datetime.now()
        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)
        training_end_time = datetime.datetime.now()
        training_time = training_end_time - training_begin_time
        self.ctx.monitor.track_training_time(training_time)  # 记录每次本地训练的训练时间

        return num_samples, self.get_model_para(), self.ctx.eval_metrics

def eval_acc(y_true, y_pred, **kwargs):
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def call_my_trainer(trainer_type):
    if trainer_type == 'fedkd_node_trainer':
        trainer_builder = FedKD_node_Trainer
        return trainer_builder


register_trainer('fedkd_node_trainer', call_my_trainer)
