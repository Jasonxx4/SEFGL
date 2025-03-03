import copy
import torch
from torch_geometric.utils import subgraph


class PGG:
    r"""
    核心思想：每个client利用全局模型来生成（或替换）本地节点/图数据，达到数据增强的目的

    Args:
        alpha (float): The interpolation factor for the slow nodes.
        p (float): The probability of half-hopping an edge.
        inplace (bool): If set to :obj:`False`, will not modify the input graph
            and will instead return a new graph.
    """

    def __init__(self, data, alpha=0.5, p=1.0, inplace=True):
        assert 0. <= p <= 1., f"p must be in [0, 1], got {p}"
        assert 0. <= alpha <= 1., f"alpha must be in [0, 1], got {alpha}"

        self.p = p
        self.alpha = alpha
        self.inplace = inplace

        self.data = data['data']

    def __call__(self, global_protos, num_classes):
        device = global_protos.device
        data = copy.deepcopy(self.data).to(device)


        x, edge_index = data.x, data.edge_index


        if self.p == 1.:
            node_mask = torch.ones(data.num_nodes, device=device, dtype=torch.bool)
        else:
            node_mask = torch.rand(data.num_nodes, device=device) < self.p

        train_mask = data.train_mask.to(device)
        # node_mask = node_mask & train_mask

        for label in range(num_classes):
            label_mask = data.y ==label
            overall_mask = train_mask & node_mask & label_mask.to(device)
            x[overall_mask] = self.alpha* x[overall_mask] + (1-self.alpha)*global_protos[label]
            # x[overall_mask,:].mul_(self.alpha).add_(global_protos[label], alpha=1. - self.alpha)
        data.x = x
        # x[node_mask,:]=
        # # add new slow nodes, and use linear interpolation to initialize their features
        # slow_node_ids = torch.arange(edge_index_to_halfhop.size(1), device=device) + data.num_nodes
        # x_slow_node = x[edge_index_to_halfhop[0]]
        # x_slow_node.mul_(self.alpha).add_(x[edge_index_to_halfhop[1]], alpha=1. - self.alpha)
        # new_x = torch.cat([x, x_slow_node], dim=0)
        #
        # # add new edges between slow nodes and the original nodes that replace the original edges
        # edge_index_slow = [
        #     torch.stack([edge_index_to_halfhop[0], slow_node_ids]),
        #     torch.stack([slow_node_ids, edge_index_to_halfhop[1]]),
        #     torch.stack([edge_index_to_halfhop[1], slow_node_ids])
        # ]
        # new_edge_index = torch.cat([edge_index_to_keep, edge_index_self_loop, *edge_index_slow], dim=1)
        #
        # # prepare a mask that distinguishes between original nodes and slow nodes
        # slow_node_mask = torch.cat([
        #     torch.zeros(x.size(0), device=device),
        #     torch.ones(slow_node_ids.size(0), device=device)
        # ], dim=0).bool()
        #
        # data.x, data.edge_index, data.slow_node_mask = new_x, new_edge_index, slow_node_mask

        return data

    def __repr__(self):
        return '{}(alpha={}, p={})'.format(self.__class__.__name__, self.alpha, self.p)
