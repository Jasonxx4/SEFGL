from federatedscope.register import register_trainer
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


# Build your trainer here.
class FedAPEN_Node_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedAPEN_Node_Trainer, self).__init__(model, data, device, config,
                                                   only_for_eval, monitor)
        '''
        self.model --> personalized local model
        self.global_model --> shared global model
        '''
        self.ctx.global_model = get_model(model_config=config.MHFL.global_model,
                                          local_data=data)  # TODO:初始时统一所有client的全局模型的权重
        self.learned_weight_for_inference = torch.tensor([0.5], requires_grad=True, device=device)
        self.optimizer_learned_weight_for_inference = torch.optim.SGD([self.learned_weight_for_inference], lr=1e-3)
        self.ctx.staged_learned_weight_inference = []
        self.epoch_for_learn_weight = config.fedapen.epoch_for_learn_weight
        self.register_our_hook()

    def get_model_para(self):
        """
        重写get_model_para(), 基类的方法是返回ctx.model的参数，现在是返回ctx.global_model的参数
        trainer.train()会调用该函数
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
        merged_param = merge_param_dict(copy.deepcopy(self.ctx.global_model.state_dict()),
                                        self._param_filter(model_parameters))
        self.ctx.global_model.load_state_dict(merged_param, strict=strict)

    def _hook_on_batch_forward(self, ctx):
        weight_private = ctx.weight_private
        batch = ctx.data_batch.to(ctx.device)
        split_mask = batch['{}_mask'.format(ctx.cur_split)]
        label = batch.y[split_mask]

        output_private = ctx.model(batch)[split_mask]
        output_shared = ctx.global_model(batch)[split_mask]
        ensemble_output_for_private = weight_private * output_private + (1 - weight_private) * output_shared.detach()
        ensemble_output_for_shared = weight_private * output_private.detach() + (1 - weight_private) * output_shared

        ce_private = CE_Loss(output_private, label)
        kl_private = KL_Loss(LogSoftmax(output_private), Softmax(output_shared.detach()))
        ce_shared = CE_Loss(output_shared, label)
        kl_shared = KL_Loss(LogSoftmax(output_shared), Softmax(output_private.detach()))

        loss_private = ce_private + kl_private + CE_Loss(ensemble_output_for_private,
                                                         label)  # the multiplication is to keep learning rate consistent with the vanilla mutual learning
        loss_shared = ce_shared + kl_shared + CE_Loss(ensemble_output_for_shared, label)

        # # adaptive
        pred_ensemble_adaptive = (weight_private * output_private + (1 - weight_private) * output_shared)

        print(f'training weight_private:{weight_private}')
        # 个性化模型的结果用来计算指标
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(output_private, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss_private, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

        # 记录全局模型的loss
        ctx.loss_batch_global = CtxVar(loss_shared, LIFECYCLE.BATCH)

        # 记录global model以及ensemble model的output
        ctx.global_ys_prob.append(output_shared.clone().detach().cpu())
        ctx.ensemble_ys_prob.append(pred_ensemble_adaptive.clone().detach().cpu())

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.global_optimizer.zero_grad()

        ctx.loss_batch.backward()  # 等价于 loss_local.backward()
        ctx.loss_batch_global.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.global_model.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()  # 更新local_model
        ctx.global_optimizer.step()  # 更新global_model

    def _hook_on_fit_start_clean(self, ctx):
        ctx.global_ys_prob = CtxVar([], LIFECYCLE.ROUTINE) # 保存global model的输出结果用以验证
        ctx.ensemble_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)  # 保存ensemble model的输出结果用以验证
        # Set optimizer for global model additionally.
        ctx.global_model.to(ctx.device)
        if ctx.cur_mode in [MODE.TRAIN]:
            ctx.global_model.train()
            ctx.global_optimizer = get_optimizer(ctx.global_model,
                                                 **ctx.cfg.train.optimizer)
        elif ctx.cur_mode in [MODE.VAL, MODE.TEST]:
            ctx.global_model.eval()

    def _hook_on_fit_end_eval_global_model(self, ctx):
        ctx.global_model.to(torch.device("cpu"))
        ctx.global_ys_prob = CtxVar(np.concatenate(ctx.global_ys_prob), LIFECYCLE.ROUTINE)
        y_true = ctx.ys_true
        y_prob = ctx.global_ys_prob
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=-1)
        if y_prob.ndim == 2:
            y_prob = np.expand_dims(y_prob, axis=-1)

        y_pred = np.argmax(y_prob, axis=1)

        acc = eval_acc(y_true, y_pred)
        logger.info(f'client#{self.ctx.client_ID} {ctx.cur_split} global_model acc :{acc}')

    def _hook_on_epoch_start_for_variable_definition(self, ctx):
        if len(ctx.staged_learned_weight_inference) == 0:
            weight_private = 0.5
        else:
            weight_private = ctx.staged_learned_weight_inference[-1]

        ctx.weight_private = CtxVar(weight_private, LIFECYCLE.EPOCH)

    def learn_weight_for_inference(self):
        """
        Learning for Adaptability
        @return:
        """
        ctx = self.ctx
        ctx.model.to(ctx.device)
        ctx.model.eval()
        ctx.global_model.to(ctx.device)
        ctx.global_model.eval()

        ctx.train_loader.reset()
        batch = next(ctx.train_loader).to(ctx.device)
        split_mask = batch['adaptability_mask']

        target = batch.y[split_mask]
        for _ in range(self.epoch_for_learn_weight):
            output_private = ctx.model(batch).detach()[split_mask]
            output_shared = ctx.global_model(batch).detach()[split_mask]

            ensemble_output = self.learned_weight_for_inference * output_private + (1 - self.learned_weight_for_inference) * output_shared
            loss = CE_Loss(ensemble_output, target)
            loss.backward()
            self.optimizer_learned_weight_for_inference.step()
            torch.clip_(self.learned_weight_for_inference.data, 0.0, 1.0)

        ctx.staged_learned_weight_inference.append(self.learned_weight_for_inference.cpu().data.item())
        ctx.model = ctx.model.cpu()
        ctx.global_model = ctx.global_model.cpu()

        print('client {0} learned weight for inference: {1}'.format(self.ctx.client_ID,
                                                                    self.learned_weight_for_inference.data.item()))

    def register_our_hook(self):
        # 定义/初始化要用到的中间变量: weight_private
        self.register_hook_in_train(self._hook_on_epoch_start_for_variable_definition, "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_variable_definition, "on_epoch_start")

        # 在client每次本地训练之前调用，用来初始化ctx.global_ys_prob(这个变量用于保存global_model的输出结果)以及global model的优化器
        self.register_hook_in_train(new_hook=self._hook_on_fit_start_clean, trigger='on_fit_start', insert_pos=-1)
        self.register_hook_in_eval(new_hook=self._hook_on_fit_start_clean, trigger='on_fit_start', insert_pos=-1)

        #
        # self.register_hook_in_train(new_hook=self._hook_on_learn_weight_for_inference, trigger='on_fit_end',
        #                             insert_pos=-1)

        # 用来在本地训练/推理结束时，额外地评估global model的ACC
        self.register_hook_in_train(new_hook=self._hook_on_fit_end_eval_global_model, trigger="on_fit_end",
                                    insert_pos=-1)
        self.register_hook_in_eval(new_hook=self._hook_on_fit_end_eval_global_model, trigger="on_fit_end",
                                   insert_pos=-1)


def eval_acc(y_true, y_pred, **kwargs):
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def call_my_trainer(trainer_type):
    if trainer_type == 'fedapen_node_trainer':
        trainer_builder = FedAPEN_Node_Trainer
        return trainer_builder


register_trainer('fedapen_node_trainer', call_my_trainer)
