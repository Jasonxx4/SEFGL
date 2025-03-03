import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import expm

@torch.no_grad()
def get_ins_neighbor_dist(num_nodes, edge_index, train_mask, device):
    """
    Compute adjacent node distribution.
    """
    ## Utilize GPU ##
    train_mask = train_mask.clone().to(device)
    edge_index = edge_index.clone().to(device)
    row, col = edge_index[0], edge_index[1]

    # Compute neighbor distribution
    neighbor_dist_list = []
    for j in range(num_nodes):
        neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32).to(device)

        idx = row[(col==j)]
        neighbor_dist[idx] = neighbor_dist[idx] + 1
        neighbor_dist_list.append(neighbor_dist)

    neighbor_dist_list = torch.stack(neighbor_dist_list,dim=0)
    neighbor_dist_list = F.normalize(neighbor_dist_list,dim=1,p=1)

    return neighbor_dist_list

def get_adj_matrix(x, edge_index) -> np.ndarray:
    num_nodes = x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.05) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)#邻接矩阵+单位矩阵
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))#度矩阵的增强版本 D_tilde，其中每个节点的度数的倒数被放置在对角线上，这是一种标准化度数的方式。
    H = D_tilde @ A_tilde @ D_tilde#最后，根据 PPR 的计算公式，使用逆矩阵运算计算了 PPR 矩阵。这个计算包括了 alpha 参数，用于控制随机游走的程度。
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H) # numpy, [N, N]矩阵的元素表示了节点之间的 PPR 值

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_PPR_adj(x, edge_index, alpha=0.05, k=None, eps=None):
    assert ((k==None and eps!=None) or (k!=None and eps==None))

    adj_matrix = get_adj_matrix(x, edge_index)#根据输入的节点特征 x 和边索引 edge_index 来生成原始的邻接矩阵
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)#是基于给定的邻接矩阵 adj_matrix 计算 Personalized PageRank (PPR) 矩阵。PPR 是一种用于测量节点之间关系强度的方法，它考虑了从某个节点出发，经过随机游走到其他节点的概率。

    if k!=None:
        ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)#这个函数可能用于获取 PPR 矩阵中前 k 个最重要的元素（也就是与每个节点最相关的 k 个节点）。这样可以控制生成的邻接矩阵的密度，只保留与每个节点最相关的一些邻居节点。
    elif eps!=None:
        ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)#这个函数可能用于对 PPR 矩阵中的元素进行剪切，将小于某个阈值 eps 的元素设置为零。这样可以控制生成的邻接矩阵的稀疏性，剪切掉与每个节点关系不太密切的邻居节点。
    else:
        raise ValueError

    return torch.tensor(ppr_matrix).float().to(x.device)

def get_heat_adj(x, edge_index, t=5.0, k=None, eps=None):
    assert ((k==None and eps!=None) or (k!=None and eps==None))
    adj_matrix = get_adj_matrix(x, edge_index)
    heat_matrix = get_heat_matrix(adj_matrix, t=t)

    if k!=None:
        heat_matrix = get_top_k_matrix(heat_matrix, k=k)
    elif eps!=None:
        heat_matrix = get_clipped_matrix(heat_matrix, eps=eps)
    else:
        raise ValueError
    
    return torch.tensor(heat_matrix).float().to(x.device)

