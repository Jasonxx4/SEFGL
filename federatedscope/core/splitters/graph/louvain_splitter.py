import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx

import networkx as nx
import community as community_louvain

from federatedscope.core.splitters import BaseSplitter


class LouvainSplitter(BaseTransform, BaseSplitter):
    """
    Split Data into small data via louvain algorithm.

    Args:
        client_num (int): Split data into ``client_num`` of pieces.
        delta (int): The gap between the number of nodes on each client.
    """
    def __init__(self, client_num, delta=40):
        self.delta = delta
        BaseSplitter.__init__(self, client_num)


    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        partition = community_louvain.best_partition(G)

        cluster2node = {}
        for node in partition:
            cluster = partition[node]
            if cluster not in cluster2node:
                cluster2node[cluster] = [node]
            else:
                cluster2node[cluster].append(node)

        max_len = len(G) // self.client_num - self.delta
        max_len_client = len(G) // self.client_num

        tmp_cluster2node = {}
        for cluster in cluster2node:
            while len(cluster2node[cluster]) > max_len:
                tmp_cluster = cluster2node[cluster][:max_len]
                tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                                 1] = tmp_cluster
                cluster2node[cluster] = cluster2node[cluster][max_len:]
        cluster2node.update(tmp_cluster2node)

        orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

        client_node_idx = {idx: [] for idx in range(self.client_num)}
        idx = 0
        for (cluster, node_list) in orderedc2n:
            while len(node_list) + len(
                    client_node_idx[idx]) > max_len_client + self.delta:
                idx = (idx + 1) % self.client_num
            client_node_idx[idx] += node_list
            idx = (idx + 1) % self.client_num

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            graphs.append(from_networkx(nx.subgraph(G, nodes)))


        return graphs
# class LouvainSplitter(BaseTransform, BaseSplitter):
#     def __init__(self, client_num, delta=20):
#         self.delta = delta
#         super().__init__(client_num)
#
#     def __call__(self, data, **kwargs):
#         data.index_orig = torch.arange(data.num_nodes)
#         G = to_networkx(data, node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'], to_undirected=True)
#         nx.set_node_attributes(G, {nid: nid for nid in range(nx.number_of_nodes(G))}, name="index_orig")
#         partition = community_louvain.best_partition(G)
#
#         cluster2node = {}
#         for node, cluster in partition.items():
#             cluster2node.setdefault(cluster, []).append(node)
#
#         max_len = len(G) // self.client_num - self.delta
#         max_len_client = len(G) // self.client_num
#
#         tmp_cluster2node = {}
#         for cluster, nodes in cluster2node.items():
#             while len(nodes) > max_len:
#                 tmp_cluster = nodes[:max_len]
#                 new_cluster_id = len(cluster2node) + len(tmp_cluster2node) + 1
#                 tmp_cluster2node[new_cluster_id] = tmp_cluster
#                 nodes = nodes[max_len:]
#             cluster2node[cluster] = nodes
#         cluster2node.update(tmp_cluster2node)
#
#         orderedc2n = sorted(cluster2node.items(), key=lambda x: len(x[1]), reverse=True)
#
#         client_node_idx = {idx: [] for idx in range(self.client_num)}
#         idx = 0
#         for cluster, node_list in orderedc2n:
#             while len(node_list) + len(client_node_idx[idx]) > max_len_client + self.delta:
#                 idx = (idx + 1) % self.client_num
#             client_node_idx[idx].extend(node_list)
#             idx = (idx + 1) % self.client_num
#
#         graphs = []
#         unique_labels = set(data.y.tolist())
#         for owner, nodes in client_node_idx.items():
#             subgraph = nx.subgraph(G, nodes)
#             subgraph = from_networkx(subgraph)
#             subgraph_indices = set(subgraph.index_orig.tolist())
#             existing_labels = {data.y[i].item() for i in subgraph_indices}
#             missing_labels = unique_labels - existing_labels
#
#             for label in missing_labels:
#                 label_indices = (data.y == label).nonzero(as_tuple=True)[0]
#                 chosen_index = label_indices[0].item()  # Choose the first index for simplicity
#                 new_node_feature = data.x[chosen_index]
#                 new_node_label = data.y[chosen_index]
#
#                 subgraph.x = torch.cat((subgraph.x, new_node_feature.unsqueeze(0)), dim=0)
#                 subgraph.y = torch.cat((subgraph.y, new_node_label.unsqueeze(0)), dim=0)
#                 subgraph.train_mask = torch.cat((subgraph.train_mask, torch.tensor([True])))
#                 subgraph.val_mask = torch.cat((subgraph.val_mask, torch.tensor([False])))
#                 subgraph.test_mask = torch.cat((subgraph.test_mask, torch.tensor([False])))
#                 subgraph.index_orig = torch.cat((subgraph.index_orig, torch.tensor([data.num_nodes])))
#
#             graphs.append(subgraph)
#
#         return graphs
