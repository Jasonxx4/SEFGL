import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils import args
from models.utilss.federated_model import FederatedModel
import torch
from utils.finch import FINCH
import numpy as np


def get_parser() -> args.ArgumentParser:
    parser = args.ArgumentParser(description='Federated learning via FedHierarchy.')
    args.add_management_args(parser)
    args.add_experiment_args(parser)
    return parser


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


class FPL(FederatedModel):
    NAME = 'fpl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FPL, self).__init__(nets_list, args, transform)
        self.global_protos = []
        self.local_protos = {}
        self.infoNCET = args.infoNCET

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                # # c一个NxP矩阵，其中N是数据点的数量，P是分区。这个矩阵中的每个元素代表每个数据点所属的聚类（簇）标签。，num_clust是指有多少个聚类，req_c是指所需聚类的标签数组（Nx1）。仅当req_clust参数不为None时才设置。它包含了指定的聚类数量对应的标签。
                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])# 存储的是每个数据点所属的聚类标签

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)#获得该聚类的平均proto

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto# 该类对应的聚类proto的dict
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        f_pos = np.array(all_f)[all_global_protos_keys == label.item()][0].to(self.device)
        f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(self.device)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

        mean_f_pos = np.array(mean_f)[all_global_protos_keys == label.item()][0].to(self.device)
        mean_f_pos = mean_f_pos.view(1, -1)
        # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
        # mean_f_neg = mean_f_neg.view(9, -1)

        loss_mse = nn.MSELoss()
        cu_info_loss = loss_mse(f_now, mean_f_pos)

        hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss

    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.global_protos = self.proto_aggregation(self.local_protos)# 返回类的聚类proto的dict{1:[[],[],[]],2:[[],[]]}
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f] #所有的proto
            mean_f = [item.detach() for item in mean_f] #每个类一个平均proto

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.classifier(f)

                lossCE = criterion(outputs, labels)

                if len(self.global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i = 0
                    loss_InfoNCE = None

                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            f_now = f[i].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys)

                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i
                loss_InfoNCE = loss_InfoNCE

                loss = lossCE + loss_InfoNCE
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, loss_InfoNCE)
                optimizer.step()

                if iter == self.local_epoch - 1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i, :])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i, :]]

        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos


