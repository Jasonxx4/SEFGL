from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.contrib.model.FedTGP_server_model import Trainable_prototypes
from collections import defaultdict
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters


class FedTGPServer(Server):
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
                 unseen_clients_id=None,
                 **kwargs):
        super(FedTGPServer, self).__init__(ID, state, config, data, model, client_num, total_round_num,
                                           device, strategy, unseen_clients_id, **kwargs)

        # feature_dim = config.FedTGP.feature_dim
        feature_dim = config.model.feature_dim
        server_hidden_dim = feature_dim
        num_classes = config.model.num_classes
        self.server_learning_rate = config.train.optimizer.lr # Consistent with local learning rate
        self.batch_size =config.FedTGP.server_batch_size
        self.PROTO = Trainable_prototypes(
            num_classes,
            server_hidden_dim,
            feature_dim,
            device
        ).to(device)

        self.server_epochs = config.FedTGP.server_epochs
        self.margin_threthold = config.FedTGP.margin_threthold

        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

        self.num_classes = num_classes

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        min_received_num = len(self.comm_manager.get_neighbors().keys())
        if check_eval_result and self._cfg.federate.mode.lower() == "standalone":
            min_received_num = len(self.comm_manager.get_neighbors().keys())
        move_on_flag = True

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                # update global protos
                #################################################################
                local_protos_dict = dict()
                msg_list = self.msg_buffer['train'][self.state]
                for client_id, values in msg_list.items():
                    local_protos = values[1]
                    local_protos_dict[client_id] = local_protos
                self.receive_protos(local_protos_dict)
                global_protos=self.update_Gen()
                # global_protos = self._proto_aggregation(local_protos_dict)
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


    def _proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
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

    def _start_new_training_round(self, global_protos):
        self._broadcast_custom_message(msg_type='global_proto', content=global_protos)

    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate', content=None, filter_unseen_clients=False)

    def update_Gen(self):
        Gen_opt = torch.optim.SGD(self.PROTO.parameters(), lr=self.server_learning_rate)
        self.PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size,
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = y.type(torch.int64).to(self.device)

                proto_gen = self.PROTO(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                gap2 = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * gap2
                loss = self.CEloss(-dist, y)

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        logger.info(f'Server loss: {loss.item()}')
        self.uploaded_protos = []
        # save_item(PROTO, self.role, 'PROTO', self.save_folder_name)

        self.PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = self.PROTO(torch.tensor(class_id, device=self.device)).detach()
        return global_protos
        # save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

    def receive_protos(self,local_protos_dict):
        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client_id in local_protos_dict:
            self.uploaded_ids.append(client_id)
            protos = local_protos_dict[client_id]
            # protos = load_item(client.role, 'protos', client.save_folder_name)
            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            uploaded_protos_per_client.append(protos)

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)


class FedTGPClient(Client):
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
        super(FedTGPClient, self).__init__(ID, server_id, state, config, data, model, device,
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

        if self._cfg.vis_embedding:
            self.glob_proto_on_client[round] = self.trainer.ctx.global_protos
            self.client_node_emb_all[round] = self.trainer.ctx.node_emb_all
            self.client_node_labels[round] = self.trainer.ctx.node_labels
            self.client_agg_proto[round] = agg_protos

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, agg_protos)))

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
        self._monitor.finish_fl()


def call_my_worker(method):
    if method == 'fedtgp':
        worker_builder = {'client': FedTGPClient, 'server': FedTGPServer}
        return worker_builder


register_worker('fedtgp', call_my_worker)
