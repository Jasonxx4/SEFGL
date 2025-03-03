from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import torch
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
import torch
import torch.nn as nn
from collections import Counter
import copy
import datetime
logger = logging.getLogger(__name__)


# Build your worker here.
class POIV5_Server(Server):
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
        super(POIV5_Server, self).__init__(ID, state, config, data, model, client_num,
                                           total_round_num, device, strategy, **kwargs)
        self.softmax = nn.Softmax(dim=-1)
        proto_agg_type = config.poi.proto_agg_type

        if proto_agg_type == 'train_loss':
            self.aggfun = self._proto_aggregation_based_on_loss
        elif proto_agg_type == 'label_distribution':
            self.aggfun = self._proto_aggregation_based_on_label_distribution
        else:
            self.aggfun = self._proto_aggregation

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
                # update global protos
                #################################################################
                local_protos_dict = dict()
                local_loss_dict = dict()
                label_distribution = dict()
                msg_list = self.msg_buffer['train'][self.state]
                for key, values in msg_list.items():
                    local_protos_dict[key] = values[1]
                    local_loss_dict[key] = values[2]
                    label_distribution[key] = values[3]
                # global_protos = self._proto_aggregation(local_protos_dict,local_loss_dict)
                global_protos = self.aggfun(local_protos_dict, local_loss_dict, label_distribution)
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
                    self._start_new_training_round(global_protos)
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

    def _proto_aggregation(self, local_protos_dict, local_loss_dict=None, label_distribution=None):
        agg_protos_label = dict()
        for idx in local_protos_dict:
            local_protos = local_protos_dict[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = proto / len(proto_list)
            else:
                agg_protos_label[label] = proto_list[0].data

        return agg_protos_label

    def _proto_aggregation_based_on_loss(self, local_protos_dict, local_loss_dict=None, label_distribution=None):
        agg_protos_label = dict()
        loss_weights = self.softmax(torch.tensor(list(local_loss_dict.values())))

        for idx, client_id in enumerate(local_protos_dict):
            weight_coefficient = loss_weights[idx]
            local_protos = local_protos_dict[client_id]
            for label in local_protos.keys():
                protos = local_protos[label] * weight_coefficient
                if label in agg_protos_label:
                    agg_protos_label[label].append(protos)
                else:
                    agg_protos_label[label] = [protos]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                # agg_protos_label[label] = proto / len(proto_list)
                agg_protos_label[label] = proto
            else:
                agg_protos_label[label] = proto_list[0].data

        return agg_protos_label

    def _proto_aggregation_based_on_label_distribution(self, local_protos_dict, local_loss_dict=None,
                                                       label_distribution=None):
        agg_protos_label = dict()
        label_num_all  =merge_dicts(label_distribution)

        for idx, client_id in enumerate(local_protos_dict):
            local_protos = local_protos_dict[client_id]
            for label in local_protos.keys():
                local_label_num =label_distribution[client_id][label]
                weight_coefficient = local_label_num/label_num_all[label]
                protos = local_protos[label] * weight_coefficient
                if label in agg_protos_label:
                    agg_protos_label[label].append(protos)
                else:
                    agg_protos_label[label] = [protos]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = proto / len(proto_list)
                # agg_protos_label[label] = proto
            else:
                agg_protos_label[label] = proto_list[0].data

        return agg_protos_label

    def _start_new_training_round(self, global_protos):
        self._broadcast_custom_message(msg_type='global_proto', content=global_protos)

    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate', content=None, filter_unseen_clients=False)

    def _broadcast_custom_message(self, msg_type, content,
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

        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=content))

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class POIV5_Client(Client):
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
        super(POIV5_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                           strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = []
        self.trainer.ctx.client_ID = self.ID
        self.register_handlers('global_proto',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])

        # For visualization of node embedding
        self.client_agg_proto = dict()
        self.client_node_emb_all = dict()
        self.client_node_labels = dict()
        self.glob_proto_on_client = dict()
        self.client_PL_node_emb_all = dict()

        # 收集训练集的标签分布
        class_num = config.model.num_classes
        self.train_label_distribution = {i: 0 for i in range(class_num)}

        train_mask = data['data'].train_mask
        train_label = data['data'].y[train_mask]
        train_label_distribution_new = [j.item() if isinstance(j, torch.Tensor) else j[1] for j in train_label]
        train_label_distribution_new = dict(Counter(train_label_distribution_new))
        self.train_label_distribution.update(train_label_distribution_new)

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        content = message.content

        if message.msg_type == 'global_proto':
            self.trainer.update(content)
        self.state = round
        self.trainer.ctx.cur_state = self.state
        sample_size, model_para, results, agg_protos = self.trainer.train()

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
            self._monitor.save_formatted_results(train_log_res,
                                                 save_file_name="")

        self.history_results = merge_dict_of_results(
            self.history_results, train_log_res['Results_raw'])

        if self._cfg.vis_embedding:
            self.glob_proto_on_client[round] = self.trainer.ctx.global_protos
            self.client_node_emb_all[round] = self.trainer.ctx.node_emb_all
            self.client_node_labels[round] = self.trainer.ctx.node_labels
            self.client_agg_proto[round] = agg_protos
            self.client_PL_node_emb_all[round] = self.trainer.ctx.PL_node_emb_all

        # POIV5: 上传本地train_avg_loss
        local_train_loss = train_log_res['Results_raw']['train_avg_loss']

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, agg_protos, local_train_loss, self.train_label_distribution)
                    )
        )

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None and self._cfg.federate.method not in ['fedmd']:  # TODO:检查fedmd是否会更新模型
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
            "local", "global"
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                inference_begin_time = datetime.datetime.now()
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)
                training_end_time = datetime.datetime.now()
                if split =='test':
                    test_inference_time = training_end_time-inference_begin_time
                    self._monitor.track_inference_time(test_inference_time)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            self._monitor.update_best_result(self.best_results,
                                             formatted_eval_res['Results_raw'],
                                             results_type=f"client #{self.ID}")
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            self.early_stopper.track_and_check(self.history_results[
                                                   self._cfg.eval.best_res_update_round_wise_key])

            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(formatted_eval_res['Results_raw'],
                                                     save_file_name="")

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_finish(self, message: Message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        if message.content is not None:
            self.trainer.update(message.content, strict=True)
        if self._cfg.vis_embedding:
            folderPath = self._cfg.MHFL.emb_file_path
            torch.save(self.glob_proto_on_client, f'{folderPath}/global_protos_on_client_{self.ID}.pth')  # 全局原型
            torch.save(self.client_agg_proto, f'{folderPath}/agg_protos_on_client_{self.ID}.pth')  # 本地原型
            torch.save(self.client_node_emb_all,
                       f'{folderPath}/local_node_embdeddings_on_client_{self.ID}.pth')  # 每个节点的embedding
            torch.save(self.client_node_labels, f'{folderPath}/node_labels_on_client_{self.ID}.pth')  # 标签
            torch.save(self.data, f'{folderPath}/raw_data_on_client_{self.ID}.pth')  # 划分给这个client的pyg data
            torch.save(self.client_PL_node_emb_all, f'{folderPath}/PP_node_embeddings_on_client_{self.ID}.pth')
        self._monitor.finish_fl()

def merge_dicts(dicts):
    #多个dict 相同的key对应的value累加
    result = {}
    for d in list(dicts.values()):
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
    return result


def call_my_worker(method):
    if method == 'poiv5':
        worker_builder = {'client': POIV5_Client, 'server': POIV5_Server}
        return worker_builder


register_worker('poiv5', call_my_worker)
