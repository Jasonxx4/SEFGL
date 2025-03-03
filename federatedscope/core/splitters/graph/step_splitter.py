import random
import torch
from torch_geometric.transforms import BaseTransform
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.graph.data_utils import get_step_imbalanced_split_data


class StepSplitter(BaseTransform, BaseSplitter):
    """
    Split Data into small data via step imbalance method.

    Args:
        client_num (int): Split data into ``client_num`` of pieces.
        delta (int): The gap between the number of nodes on each client.
        imbratio (int): Imbalance ratio for the entire dataset.
        head_class_size (int): Size of the head classes.
        test_ratio (float): Ratio of nodes to use for testing.
        random_seed (int): Seed for random number generator.
        verbose (bool): Whether to print verbose information.
    """

    def __init__(self, client_num, delta=20, imbratio=10, head_class_size=20, test_ratio=0.5, random_seed=43,
                 verbose=False):
        self.delta = delta
        self.imbratio = imbratio
        self.head_class_size = head_class_size
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.verbose = verbose
        BaseSplitter.__init__(self, client_num)

    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)

        # Perform step imbalance split
        split_data = get_step_imbalanced_split_data(
            data,
            imbratio=self.imbratio,
            head_class_size=self.head_class_size,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
            verbose=self.verbose
        )

        # Assign nodes to clients
        client_node_idx = {idx: [] for idx in range(self.client_num)}
        train_mask = split_data.train_mask
        indices = torch.arange(len(train_mask))[train_mask].tolist()
        random.shuffle(indices)

        max_len = len(indices) // self.client_num - self.delta
        max_len_client = len(indices) // self.client_num
        idx = 0
        for node in indices:
            while len(client_node_idx[idx]) >= max_len_client + self.delta:
                idx = (idx + 1) % self.client_num
            client_node_idx[idx].append(node)
            idx = (idx + 1) % self.client_num

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            node_mask = torch.zeros_like(train_mask, dtype=torch.bool)
            node_mask[nodes] = True
            subgraph = split_data.clone()
            subgraph.train_mask = node_mask
            graphs.append(subgraph)

        return graphs
