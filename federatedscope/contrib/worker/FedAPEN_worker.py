from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import torch
import copy
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict

logger = logging.getLogger(__name__)


# Build your worker here.
class FedAPEN_Server(Server):
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
        super(FedAPEN_Server, self).__init__(ID, state, config, data, model, client_num,
                                         total_round_num, device, strategy, **kwargs)
        self.model = get_model(model_config=config.MHFL.global_model, local_data=data)
        self.models[0] = self.model


class FedAPEN_Client(Client):
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
        super(FedAPEN_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                             strategy, is_unseen_client, *args, **kwargs)
        self.set_adaptability_dataset()

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        content = message.content
        self.state = round
        self.trainer.ctx.cur_state=round

        self.trainer.update(content)

        if self.state!=0:
            # update the weight for inference according to the adaptability set
            self.trainer.learn_weight_for_inference()

        sample_size, model_para_all, results = self.trainer.train() #local training
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

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para_all)))

    def callback_funcs_for_finish(self, message: Message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")
        if message.content is not None:
            self.trainer.update(message.content, strict=True)
        self._monitor.finish_fl()

    def set_adaptability_dataset(self):
        adaptability_ratio = self._cfg.fedapen.adaptability_ratio
        train_data = self.trainer.ctx.data.train_data[0]
        train_mask = train_data.train_mask

        num_train = train_mask.sum().item()
        num_adaptability = int(num_train * adaptability_ratio)

        new_num_train = num_train - num_adaptability
        train_node_indices = torch.nonzero(train_mask).squeeze()
        new_train_indices = train_node_indices[:new_num_train]
        adaptability_indices = train_node_indices[new_num_train:]

        train_mask.fill_(False)  #
        adaptability_mask = torch.zeros_like(train_mask)

        train_mask[new_train_indices] = True
        adaptability_mask[adaptability_indices] = True

        # train_data.train_mask = train_mask
        train_data.adaptability_mask = adaptability_mask



def call_my_worker(method):
    if method == 'fedapen':
        worker_builder = {'client': FedAPEN_Client, 'server': FedAPEN_Server}
        return worker_builder


register_worker('fedapen', call_my_worker)
