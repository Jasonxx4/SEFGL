import copy
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
from federatedscope.contrib.common_utils import param2tensor
import logging
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict

import time
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from collections import defaultdict
from collections.abc import Sized
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class FedKD_Server(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(FedKD_Server, self).__init__(ID, state, config, data, model, client_num,
                                           total_round_num, device, strategy, **kwargs)
        self.device = device
        self.task = config.MHFL.task
        self.T_start = config.fedkd.tmin
        self.T_end = config.fedkd.tmax
        self.total_round = config.federate.total_round_num


        self.model = get_model(model_config=config.MHFL.global_model, local_data=data)
        # global_hidden_dim = config.MHFL.global_model.hidden
        # self.model.W = nn.Linear(global_hidden_dim, global_hidden_dim, bias=False).to(device)
        self.models[0] = self.model

        self.svd = config.fedkd.use_SVD

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        # TODO: 需要完善当采样率不等于0时的实现
        min_received_num = len(self.comm_manager.get_neighbors().keys())

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process

                #################################################################
                train_msg_dict = self.msg_buffer['train'][self.state]
                if self.svd:
                    train_msg_dict = self.reconstruction(train_msg_dict)
                aggregated_num = self._perform_federated_aggregation(train_msg_dict)
                #################################################################

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True
        else:
            move_on_flag = False

        return move_on_flag

    def reconstruction(self, train_msg_dict):
        new_train_msg_dict = dict()
        for client_id, (sample_size, model_para) in train_msg_dict.items():
            new_model_para = dict()
            for name, values in model_para.items():
                if isinstance(values, np.ndarray) and values.size == 1:
                    new_model_para[name] = param2tensor(
                        values)  # 对于size为1的numpy.ndarray对象，不能直接用len()函数，会报错: len() of unsized object
                    continue
                if len(values) == 3:
                    values = np.matmul(values[0] * values[1][..., None, :], values[2])
                new_model_para[name] = param2tensor(values)
            new_train_msg_dict[client_id] = (sample_size, new_model_para)
        return new_train_msg_dict

    def _perform_federated_aggregation(self, new_train_msg_buffer):
        """
        Perform federated aggregation and update the global model
        """
        model = self.model
        aggregator = self.aggregators[0]
        msg_list = list()
        staleness = list()

        for client_id, values in new_train_msg_buffer.items():
            msg_list.append(values)  # values: (train_sample_size, model_para)

        # Aggregate
        aggregated_num = len(msg_list)
        agg_info = {
            'client_feedback': msg_list,
            'recover_fun': self.recover_fun,
            'staleness': staleness,
        }
        result = aggregator.aggregate(agg_info)
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(model.state_dict().copy(), result)
        model.load_state_dict(merged_param, strict=True)

        return aggregated_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.models[0].state_dict()

        #####################################################################################
        if msg_type == 'model_para' and self.svd:
            energy = self.T_start + ((1 + self.state) / self.total_round) * (self.T_end - self.T_start)
            model_para = decomposition(model_para, energy)

        ########################################################################################
        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class FedKD_Client(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FedKD_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                           strategy, is_unseen_client, *args, **kwargs)
        self.T_start = config.fedkd.tmin
        self.T_end = config.fedkd.tmax
        self.total_round = config.federate.total_round_num
        self.svd = config.fedkd.use_SVD
    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        model_para = message.content

        if self.svd:
            model_para=reconstruction(model_para)

        self.trainer.update(model_para, strict=True)  # TODO: 替换content为global model 的para
        self.state = round

        if self.is_unseen_client:
            sample_size, model_para_all, results = 0, self.trainer.get_model_para(), {}
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()

            sample_size, model_para_all, results = self.trainer.train()
            model_para_all = copy.deepcopy(model_para_all)

            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                     save_file_name="")

        if self.svd:
            energy = self.T_start + ((1 + round) / self.total_round) * (self.T_end - self.T_start)
            model_para_all = decomposition(model_para_all, energy)

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, model_para_all)))

def reconstruction(model_para):
    new_model_para = dict()
    for name, values in model_para.items():
        if isinstance(values, np.ndarray) and values.size == 1:
            # 对于size为1的numpy.ndarray对象，不能直接用len()函数，会报错: len() of unsized object
            new_model_para[name] = param2tensor(values)
            continue
        if len(values) == 3:
            values = np.matmul(values[0] * values[1][..., None, :], values[2])
        new_model_para[name] = param2tensor(values)
    return new_model_para


def decomposition(model_para, energy):
    """
    refer to: https://github.com/TsingZ0/HFL/blob/main/system/flcore/clients/clientkd.py#L113-L150
    """
    compressed_param = {}
    for name, param in model_para.items():
        try:
            param_cpu = param.detach().cpu().numpy()
        except:
            param_cpu = param
        # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
        if len(param_cpu.shape) > 1 and 'embeddings' not in name:
            u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
            # support high-dimensional CNN param
            if len(u.shape) == 4:
                u = np.transpose(u, (2, 3, 0, 1))
                sigma = np.transpose(sigma, (2, 0, 1))
                v = np.transpose(v, (2, 3, 0, 1))
            threshold = 0
            if np.sum(np.square(sigma)) == 0:
                compressed_param_cpu = param_cpu
            else:
                for singular_value_num in range(len(sigma)):
                    if np.sum(np.square(sigma[:singular_value_num])) > energy * np.sum(np.square(sigma)):
                        threshold = singular_value_num
                        break
                u = u[:, :threshold]
                sigma = sigma[:threshold]
                v = v[:threshold, :]
                # support high-dimensional CNN param
                if len(u.shape) == 4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (1, 2, 0))
                    v = np.transpose(v, (2, 3, 0, 1))
                compressed_param_cpu = [u, sigma, v]
        elif 'embeddings' not in name:
            compressed_param_cpu = param_cpu

        compressed_param[name] = compressed_param_cpu

    return compressed_param

def call_my_worker(method):
    if method == 'fedkd':
        worker_builder = {'client': FedKD_Client, 'server': FedKD_Server}
        return worker_builder


register_worker('fedkd', call_my_worker)
