import copy
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.contrib.model.FedGH_FC import FedGH_FC

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class POI_Server(Server):
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
        super(POI_Server, self).__init__(ID, state, config, data, model, client_num,
                                         total_round_num, device, strategy, **kwargs)

        self.model = get_model(model_config=config.MHFL.global_model, local_data=data)
        self.models[0] = self.model


class POI_client(Client):
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
        super(POI_client, self).__init__(ID, server_id, state, config, data, model, device,
                                         strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.client_ID = self.ID

    #     # For visualization of node embedding
    #     self.client_agg_proto = dict()
    #     self.client_node_emb_all = dict()
    #     self.client_node_labels = dict()
    #     self.glob_proto_on_client = dict()
    #
    # def callback_funcs_for_model_para(self, message: Message):
    #     round = message.state
    #     sender = message.sender
    #     timestamp = message.timestamp
    #     content = message.content
    #
    #     self.trainer.update(content, strict=True)
    #     self.state = round
    #     self.trainer.ctx.cur_state = round
    #     sample_size, model_para, results, agg_protos = self.trainer.train()
    #
    #     train_log_res = self._monitor.format_eval_res(
    #         results,
    #         rnd=self.state,
    #         role='Client #{}'.format(self.ID),
    #         return_raw=True)
    #     logger.info(train_log_res)
    #
    #     if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
    #         self._monitor.save_formatted_results(train_log_res,
    #                                              save_file_name="")
    #
    #     self.comm_manager.send(
    #         Message(msg_type='model_para',
    #                 sender=self.ID,
    #                 receiver=[sender],
    #                 state=self.state,
    #                 content=(sample_size, agg_protos)))
    #
    #     if self._cfg.vis_embedding:
    #         self.client_node_emb_all[self.state] = self.trainer.ctx.node_emb_all
    #         self.client_node_labels[self.state] = self.trainer.ctx.node_labels
    #         self.client_agg_proto[self.state] = agg_protos
    #
    # def callback_funcs_for_finish(self, message: Message):
    #     logger.info(
    #         f"================= client {self.ID} received finish message "
    #         f"=================")
    #
    #     if message.content is not None:
    #         self.trainer.update(message.content, strict=True)
    #     if self._cfg.vis_embedding:
    #         folderPath = self._cfg.MHFL.emb_file_path
    #         torch.save(self.glob_proto_on_client, f'{folderPath}/global_protos_on_client_{self.ID}.pth')  # 全局原型
    #         torch.save(self.client_agg_proto, f'{folderPath}/agg_protos_on_client_{self.ID}.pth')  # 本地原型
    #         torch.save(self.client_node_emb_all,
    #                    f'{folderPath}/local_node_embdeddings_on_client_{self.ID}.pth')  # 每个节点的embedding
    #         torch.save(self.client_node_labels, f'{folderPath}/node_labels_on_client_{self.ID}.pth')  # 标签
    #         torch.save(self.data, f'{folderPath}/raw_data_on_client_{self.ID}.pth')  # 划分给这个client的pyg data
    #     self._monitor.finish_fl()


def call_my_worker(method):
    if method == 'poi':
        worker_builder = {'client': POI_client, 'server': POI_Server}
        return worker_builder


register_worker('poi', call_my_worker)
