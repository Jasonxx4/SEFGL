import torch
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
splits = [0.6, 0.2, 0.2] #TODO: 不要硬编码


def load_Amazon_Dataset(path, name):
    dataset = Amazon(path, name, T.NormalizeFeatures(), pre_transform=get_split_mask)
    data = dataset[0]
    return data

def get_split_mask(data):
    train_rate = splits[0]
    val_rate = splits[1]
    test_rate = splits[2]
    num_classes = data.y.max().item() + 1
    num_train_per_class = int(round(train_rate * len(data.y) / num_classes))

    num_val = int(round(val_rate * len(data.y)))
    num_test = int(round(test_rate * len(data.y)))

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True
    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data.val_mask.fill_(False)
    data.val_mask[remaining[:num_val]] = True

    data.test_mask.fill_(False)
    data.test_mask[remaining[num_val:num_val + num_test]] = True

    return data

# class Amazon_with_mask(Amazon):
#     def __init__(self, root: str, name: str,
#                  transform: Optional[Callable] = None,
#                  pre_transform: Optional[Callable] = None,
#                  splits=[0.6, 0.2, 0.2]):
#         super().__init__(root, name, transform, pre_transform)
#         self.splits = splits
#
#     def process(self):
#         data = read_npz(self.raw_paths[0])
#         data = data if self.pre_transform is None else self.pre_transform(data,self.splits)
#         data, slices = self.collate([data])
#         torch.save((data, slices), self.processed_paths[0])
