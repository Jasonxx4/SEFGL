import torch
import dgl
from torch.cuda.amp import autocast


def add_knn(k, node_embed, edge_index, device):
    if device == torch.device("cpu"):
        knn_g = dgl.knn_graph(node_embed,
                              k,
                              algorithm='bruteforce',
                              dist='cosine')
    else:
        knn_g = dgl.knn_graph(node_embed,
                              k,
                              algorithm='bruteforce-sharemem',
                              dist='cosine')
    knn_g = dgl.add_reverse_edges(knn_g)
    knn_edge_index = knn_g.edges()
    knn_edge_index = torch.cat(
        (knn_edge_index[0].reshape(1, -1), knn_edge_index[1].reshape(1, -1)),
        dim=0)
    knn_edge_index = knn_edge_index.t()
    edge_index_2 = torch.concat((edge_index, knn_edge_index), dim=0)
    edge_index_2 = torch.unique(edge_index_2, dim=0)
    return edge_index_2


def calc_e1(adj: torch.Tensor):
    adj = adj - torch.diag_embed(torch.diag(adj))
    degree = adj.sum(dim=1)
    vol = adj.sum()
    idx = degree.nonzero().reshape(-1)
    g = degree[idx]
    return -((g / vol) * torch.log2(g / vol)).sum()


def get_adj_matrix(node_num, edge_index, weight) -> torch.Tensor:
    adj_matrix = torch.zeros((node_num, node_num), device= weight.device)
    adj_matrix[edge_index.t()[0], edge_index.t()[1]] = weight
    adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))  #去除对角线
    return adj_matrix
# def get_adj_matrix(node_num, edge_index, weight) -> torch.Tensor:
#     adj_matrix = torch.zeros((node_num, node_num), device=edge_index.device)
#     adj_matrix[edge_index.t()[0], edge_index.t()[1]] = weight
#     adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))  # 去除对角线
#     return adj_matrix



# def get_weight(node_embedding, edge_index):
#     # Ensure all tensors are on the same device
#     node_embedding = node_embedding.to(edge_index.device)
#
#     # Get the embeddings for the edges
#     links = node_embedding[edge_index]
#
#     # Compute mean for each link pair
#     means = links.mean(dim=1, keepdim=True)
#
#     # Compute the deviations from the mean
#     deviations = links - means
#
#     # Compute the covariance
#     covariances = (deviations[:, 0] * deviations[:, 1]).mean(dim=1)
#
#     # Compute the standard deviations
#     std_devs = torch.sqrt((deviations ** 2).mean(dim=1))
#     std_devs = std_devs[:, 0] * std_devs[:, 1]
#
#     # Compute correlation coefficients
#     corr_coefs = covariances / std_devs
#     corr_coefs[torch.isnan(corr_coefs)] = 0
#
#     weight = corr_coefs + 1
#     node_num = node_embedding.shape[0]
#     M = weight.mean() / (2 * node_num)
#     weight += M
#
#     return weight

# def get_weight(node_embedding, edge_index):
#     # Get all node pairs using advanced indexing
#     node_pairs = node_embedding[edge_index].view(-1, 2, node_embedding.size(1))
#
#     # Standardize features for each node in pairs
#     mean = node_pairs.mean(dim=2, keepdim=True)
#     std = node_pairs.std(dim=2, keepdim=True)
#     node_pairs = (node_pairs - mean) / std
#
#     # Calculate correlation coefficients using batch matrix multiplication
#     corr = torch.bmm(node_pairs[:, 0, :, None], node_pairs[:, 1, :, None].transpose(1, 2)).squeeze()
#
#     # Add 1 and replace NaNs
#     weight = torch.nan_to_num(corr + 1, nan=0.0)
#
#     # Calculate and adjust by mean
#     M = weight.mean() / (2 * node_embedding.shape[0])
#     weight += M
#
#     return weight

# baoxiancun
# def get_weight(node_embedding, edge_index):
#     # Get all node pairs using advanced indexing
#     node_pairs = node_embedding[edge_index].view(-1, 2, node_embedding.size(1))
#
#     # Standardize features for each node in pairs
#     mean = node_pairs.mean(dim=2, keepdim=True)
#     std = node_pairs.std(dim=2, keepdim=True)
#     node_pairs_normalized = (node_pairs - mean) / (std + 1e-6)  # Adding a small epsilon to avoid division by zero
#
#     # Flatten the embeddings to perform element-wise multiplication
#     flattened_pairs = node_pairs_normalized.view(-1, node_embedding.size(1) * 2)
#
#     # Calculate dot product via multiplication
#     prod = torch.sum(flattened_pairs[:, :node_embedding.size(1)] * flattened_pairs[:, node_embedding.size(1):], dim=1)
#     norm = torch.sqrt(torch.sum(flattened_pairs[:, :node_embedding.size(1)] ** 2, dim=1) * torch.sum(
#         flattened_pairs[:, node_embedding.size(1):] ** 2, dim=1))
#
#     # Calculate correlation coefficient
#     corr = prod / norm
#
#     # Handle any NaNs and add 1 to the correlation coefficient
#     weight = torch.nan_to_num(corr + 1, nan=0.0)
#
#     # Adjust by the mean value
#     M = weight.mean() / (2 * node_embedding.shape[0])
#     weight += M
#
#     return weight

# def get_weight(node_embedding, edge_index):
#     with autocast():
#         node_pairs = node_embedding[edge_index].view(-1, 2, node_embedding.size(1))
#         mean = node_pairs.mean(dim=2, keepdim=True)
#         std = node_pairs.std(dim=2, keepdim=True)
#         node_pairs_normalized = (node_pairs - mean) / (std + 1e-6)
#
#         prod = torch.sum((node_pairs_normalized[:, 0] * node_pairs_normalized[:, 1]), dim=1)
#         norm = torch.sqrt(torch.sum(node_pairs_normalized[:, 0] ** 2, dim=1) *
#                           torch.sum(node_pairs_normalized[:, 1] ** 2, dim=1))
#         corr = torch.nan_to_num(prod / norm + 1, nan=0.0)
#         M = corr.mean() / (2 * node_embedding.shape[0])
#         weight = corr + M
#
#     return weight

def get_weight(node_embedding, edge_index):
    node_pairs = node_embedding[edge_index].view(-1, 2, node_embedding.size(1))
    mean = node_pairs.mean(dim=2, keepdim=True)
    std = node_pairs.std(dim=2, keepdim=True)
    node_pairs_normalized = (node_pairs - mean) / (std + 1e-6)

    # Combine flattening and multiplication
    # Calculate dot product via multiplication
    prod = torch.sum((node_pairs_normalized[:, 0] * node_pairs_normalized[:, 1]), dim=1)
    norm = torch.sqrt(torch.sum(node_pairs_normalized[:, 0] ** 2, dim=1) *
                      torch.sum(node_pairs_normalized[:, 1] ** 2, dim=1))

    # Calculate correlation coefficient and handle NaNs
    corr = torch.nan_to_num(prod / norm + 1, nan=0.0)

    # Adjust by the mean value
    M = corr.mean() / (2 * node_embedding.shape[0])
    weight = corr + M

    return weight

def knn_maxE1(edge_index: torch.Tensor, node_embedding: torch.Tensor, device):
    old_e1 = 0
    node_num = node_embedding.shape[0]
    k = 1
    while k < 50:
        edge_index_k = add_knn(k, node_embedding, edge_index, device)
        weight = get_weight(node_embedding, edge_index_k)
        # e1 = calc_e1(edge_index_k, weight)
        adj = get_adj_matrix(node_num, edge_index_k, weight)
        e1 = calc_e1(adj)
        if e1 - old_e1 > 0.1:
            k += 5
        elif e1 - old_e1 > 0.01:
            k += 3
        elif e1 - old_e1 > 0.001:
            k += 1
        else:
            break
        old_e1 = e1
    #print(f'max1SE k: {k}')
    return k
