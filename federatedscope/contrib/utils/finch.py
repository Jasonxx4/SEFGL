import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:#如果数据集大小小于或等于 ANN_THRESHOLD，则通过计算节点间的距离矩阵（使用给定的距离度量）来确定首个邻居的排名
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance) # 获得该key对应的不同客户端的proto的距离
        np.fill_diagonal(orig_dist, 1e12) # 把对角线设为无限大
        initial_rank = np.argmin(orig_dist, axis=1)# 获得各自距离最小的
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')# 与单位矩阵相加
    A = A @ A.T

    A = A.tolil()# 这行代码将矩阵A从CSR（压缩稀疏行）格式转换为LIL（列表列表）格式。LIL格式允许有效地按行编辑稀疏矩阵。
    A.setdiag(0)
    return A, orig_dist #邻接矩阵和距离矩阵


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)# 聚类的数量，和每个节点所属类的标签
    return u, num_clust


def cool_mean_old(M, u):
    _, nf = np.unique(u, return_counts=True)
    idx = np.argsort(u)
    M = M[idx, :]
    M = np.vstack((np.zeros((1, M.shape[1])), M))

    np.cumsum(M, axis=0, out=M)
    cnf = np.cumsum(nf)
    nf1 = np.insert(cnf, 0, 0)
    nf1 = nf1[:-1]

    M = M[cnf, :] - M[nf1, :]
    M = M / nf[:, None]
    return M


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)# 获取唯一的聚类标签 un，以及每个聚类中数据点的数量 nf。
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]#计算每个聚类的均值特征


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data = data.astype(np.float32)

    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)# 聚类，group 是一个标签数组，它指示了每个节点属于哪个组（或簇）。具体来说，group[i] 表示第 i 个节点所属的组的标签。这个数组使得您可以知道哪些节点在同一个组中，从而进行聚类操作。num_clust是组的个数
    c, mat = get_merge([], group, data)# c每个reps对应的聚类，mat是每个聚类的均值

    if verbose:
        # print('Partition 0: {} clusters'.format(num_clust))
        pass

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, initial_rank, distance)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c