from scipy.spatial.distance import cdist

from federatedscope.contrib.utils.finch import FINCH
from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import torch
import numpy as np
import time
import datetime
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict

logger = logging.getLogger(__name__)


# Build your worker here.
class SEFGLServer(Server):

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        #TODO: 需要完善当采样率不等于0时的实现
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
                hd_se = {}
                global_protos = {}
                local_protos_list = dict()
                msg_list = self.msg_buffer['train'][self.state]
                aggregated_num = len(msg_list)
                for key, values in msg_list.items():
                    local_protos_list[key] = values[1]
                    hd_se[key] = msg_list[key][-1]
                #print("***********************local_protos_list*************:", local_protos_list.keys())
                for key in local_protos_list.keys():
                    global_protos[key] = self._proto_aggregation(local_protos_list, key, hd_se)
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

    def _proto_aggregation(self, local_protos_list,client_id, hd_se):
        agg_protos_label = dict()
        closest_keys = self.find_closest_keys(hd_se, client_id)
        new_local_protos_list = self.new_local_protos_list(local_protos_list, closest_keys)
        for idx in new_local_protos_list:
            local_protos = new_local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)
                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)
                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = torch.tensor([proto_list[0].data]).to('cuda:4')
                # agg_protos_label[label] = torch.tensor(proto_list[0].data, device='cuda:4')

        return agg_protos_label

    def _start_new_training_round(self, global_protos):
        # print("*********************client_id:",self.ID)
        self._broadcast_custom_message(msg_type='global_proto',content=global_protos)


    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate',content=None, filter_unseen_clients=False)

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

    def find_closest_keys(self, hd_se, client_id, k=7):
        keys = list(hd_se.keys())
        values = np.array(list(hd_se.values())).reshape(-1, 1)

        distances = cdist(values, values, metric='euclidean')
        client_index = keys.index(client_id)
        closest_indices = np.argsort(distances[client_index])[:k]  # 包括自己，并选出前 k 个最近的
        closest_keys = {client_id: [keys[idx] for idx in closest_indices]}

        return closest_keys

    def new_local_protos_list(self, local_protos_list, closest_keys):
        new_local_protos = {}
        for key, close_keys in closest_keys.items():
            new_local_protos = {k: local_protos_list[k] for k in close_keys}
        return new_local_protos


class SEFGLClient(Client):
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
        super(SEFGLClient, self).__init__(ID, server_id, state, config, data, model, device,
                                             strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = {}
        self.trainer.ctx.client_ID = self.ID
        self.trainer.ctx.hd_se = []
        self.register_handlers('global_proto',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para', 'model_hd_se'])


        # For visualization of node embedding
        self.client_agg_proto = dict()
        self.client_node_emb_all = dict()
        self.client_node_labels = dict()
        self.client_node_aug_emb_all = dict()
        self.client_node_aug_labels = dict()
        self.glob_proto_on_client = dict()


    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        """
        self.trainer.data_transfer(self.trainer.ctx)
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))
    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender

        content = message.content

        if message.msg_type == 'global_proto':
            self.trainer.update(content[self.ID])
        self.state = round
        self.trainer.ctx.cur_state = self.state
        train_start = time.time()
        sample_size, model_para, results, agg_protos = self.trainer.train()
        train_end = time.time()
        # print("一轮本地训练时间", train_end-train_start)
        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
            self._monitor.save_formatted_results(train_log_res,
                                                 save_file_name="")

        if self._cfg.vis_embedding:
            self.glob_proto_on_client[round] = self.trainer.ctx.global_protos
            self.client_node_emb_all[round] = self.trainer.ctx.node_emb_all
            self.client_node_labels[round] = self.trainer.ctx.node_labels
            self.client_node_aug_emb_all[round] = self.trainer.ctx.node_aug_all
            self.client_node_aug_labels[round] = self.trainer.ctx.node_aug_labels
            self.client_agg_proto[round] = agg_protos

            folderPath = self._cfg.MHFL.emb_file_path
            print("****************************floderpath:",folderPath)
            torch.save(self.glob_proto_on_client, f'{folderPath}/global_protos_on_client_{self.ID}.pth')
            torch.save(self.client_agg_proto, f'{folderPath}/agg_protos_on_client_{self.ID}.pth')
            torch.save(self.client_node_emb_all,
                       f'{folderPath}/local_node_embdeddings_on_client_{self.ID}.pth')
            torch.save(self.client_node_labels, f'{folderPath}/node_labels_on_client_{self.ID}.pth')
            torch.save(self.client_node_aug_emb_all,
                       f'{folderPath}/local_node_aug_embdeddings_on_client_{self.ID}.pth')
            torch.save(self.client_node_aug_labels, f'{folderPath}/node_aug_labels_on_client_{self.ID}.pth')
            torch.save(self.data, f'{folderPath}/raw_data_on_client_{self.ID}.pth')



        hd_se_data = self.trainer.ctx.hd_se
        content_to_send = (sample_size, agg_protos, hd_se_data)

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=content_to_send))

    def callback_funcs_for_finish(self, message: Message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        if message.content is not None:
            self.trainer.update(message.content, strict=True)
        if self._cfg.vis_embedding:
            folderPath = self._cfg.MHFL.emb_file_path
            # print("****************************floderpath:",folderPath)
            torch.save(self.glob_proto_on_client, f'{folderPath}/global_protos_on_client_{self.ID}.pth')
            torch.save(self.client_agg_proto, f'{folderPath}/agg_protos_on_client_{self.ID}.pth')
            torch.save(self.client_node_emb_all,
                       f'{folderPath}/local_node_embdeddings_on_client_{self.ID}.pth')
            torch.save(self.client_node_labels, f'{folderPath}/node_labels_on_client_{self.ID}.pth')
            torch.save(self.client_node_aug_emb_all,
                       f'{folderPath}/local_node_aug_embdeddings_on_client_{self.ID}.pth')
            torch.save(self.client_node_aug_labels, f'{folderPath}/node_aug_labels_on_client_{self.ID}.pth')
            torch.save(self.data, f'{folderPath}/raw_data_on_client_{self.ID}.pth')
        self._monitor.finish_fl()



def call_my_worker(method):
    if method == 'sefgl_worker':
        worker_builder = {'client': SEFGLClient, 'server': SEFGLServer}
        return worker_builder


register_worker('sefgl_worker', call_my_worker)
