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
import datetime
logger = logging.getLogger(__name__)


class FML_Server(Server):
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
        super(FML_Server, self).__init__(ID, state, config, data, model, client_num,
                                         total_round_num, device, strategy, **kwargs)

        self.model = get_model(model_config=config.fml.meme_model, local_data=data)
        self.models[0] = self.model


class FML_client(Client):
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
        super(FML_client, self).__init__(ID, server_id, state, config, data, model, device,
                                         strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.client_ID = self.ID

def call_my_worker(method):
    if method == 'fml':
        worker_builder = {'client': FML_client, 'server': FML_Server}
        return worker_builder


register_worker('fml', call_my_worker)
