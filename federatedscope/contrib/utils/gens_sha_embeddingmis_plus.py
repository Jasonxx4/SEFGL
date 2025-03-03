import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
###有节点混合
@torch.no_grad()
def sampling_idx_individual_dst(class_num_list, idx_info, device):#每个类的节点数list7，每个类对应的训练索引构成的7个tensor
    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)#计算采样多少节点才能达到最大节点数
    new_class_num_list = torch.Tensor(class_num_list).to(device)

    # Compute # of source nodes ///class_num_list多的,cls_idx就少
    sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
                        for cls_idx, samp_num in zip(idx_info, sampling_list)]#对于每个类别，使用 torch.randint 函数生成一个随机整数索引，范围在 [0, 该类别对应训练索引长度) 之间，且生成数量等于该类别对应sampling_list 的随机整数。这样，就相当于从每个类别的训练集中随机选择了指定数量的节点索引。
    sampling_src_idx = torch.cat(sampling_src_idx)

    #源节点是已知的节点，而目标节点是根据概率分布随机选择的节点索引
    # Generate corresponding destination nodes
    prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()#计算了目标节点的采样概率分布，根据节点数的对数来计算概率分布。这是为了确保不同类别的节点被采样的机会与其节点数成比例。
    prob = prob.repeat_interleave(new_class_num_list.long())#将 prob 中的概率值按照每个节点类别的数量进行复制，以构建一个新的概率分布，其中每个节点类别的概率值被复制了相应数量的次数，以与 new_class_num_list 中的节点数量匹配。
    temp_idx_info = torch.cat(idx_info)
    #返回下标，使用 torch.multinomial 函数根据概率分布 prob 进行多项式采样，采样的数量由 sampling_src_idx.shape[0] 决定。参数 True 表示有放回采样，即可以多次采样相同的节点。
    #这意味着每个源节点都要根据概率分布独立进行采样，以确定连接的目标节点。
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
    sampling_dst_idx = temp_idx_info[dst_idx]



    return sampling_src_idx, sampling_dst_idx, sampling_list

def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam, global_embedding, sampling_list):
    """

    Args:
        x:输入的节点特征矩阵，每行代表一个节点的特征
        sampling_src_idx:采样的源节点索引，用于生成混合节点。
        sampling_dst_idx:采样的目标节点索引，用于生成混合节点。
        lam:Mixup 系数，控制混合程度的参数。

    Returns:

    """
    mixed_node = torch.tensor([]).to(x.device)
    sampling_src_idx = sampling_src_idx[:]
    sampling_dst_idx = sampling_dst_idx[:]
    lam = lam.to(x.device)
    for i in range(7):
        new_src = x[sampling_src_idx[:int(sampling_list[i])].to(x.device), :].clone()#获得x中sampling_src_idx索引节点的特征
        new_dst = x[sampling_dst_idx[:int(sampling_list[i])].to(x.device), :].clone()
        sampling_src_idx = sampling_src_idx[int(sampling_list[i]):]
        sampling_dst_idx = sampling_dst_idx[int(sampling_list[i]):]

        if (len(global_embedding)>0):
            one_class_mixed_node = lam[:int(sampling_list[i])] * new_src + (0.5-lam[:int(sampling_list[i])]) * new_dst +0.5*global_embedding[i]
        else:
            one_class_mixed_node = lam[:int(sampling_list[i])] * new_src + (1 - lam[:int(sampling_list[i])]) * new_dst
        lam = lam[int(sampling_list[i]):]
        mixed_node = torch.cat([mixed_node,one_class_mixed_node], dim =0)
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x

@torch.no_grad()
def  duplicate_neighbor(total_node, edge_index, sampling_src_idx):#2708
    device = edge_index.device
    #sampling_src_idx为新节点的源节点索引
    #edge_index（表示原始的边列表，包括行索引和列索引）
    # edge_index = torch.tensor([
    #     [0, 1, 2, 3],  # 源节点的标识符
    #     [1, 2, 3, 0]  # 目标节点的标识符
    # ])
    # Assign node index for augmented nodes
    row, col = edge_index[0], edge_index[1] 
    row, sort_idx = torch.sort(row)
    col = col[sort_idx] 
    degree = scatter_add(torch.ones_like(row), row)#这行代码计算了图中每个节点的度数，它将度数信息存储在 degree 张量中。
    #根据每个源节点的度数，重复相应数量的节点索引，以确保新节点连接的度数与源节点的度数相匹配。新节点数和源节点数也相同
    new_row =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])#为新节点分配唯一的索引，确保它们不与原始节点的索引重叠，根据每个源节点的度数来重复相应数量的节点索引。这意味着如果某个源节点的度数为 k，那么对应的索引将被重复 k 次。这是为了为每个源节点创建多个副本，以匹配其度数。
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)#使用 scatter_add 函数再次计算每个源节点的度数，并将结果存储在 temp 张量中

    # Duplicate the edges of source nodes
    # 创建一个节点掩码 node_mask，用于标记哪些节点是源节点。通过找到唯一的源节点索引，并将对应位置设置为 True。以此找到源节点的边
    node_mask = torch.zeros(total_node, dtype=torch.bool)
    unique_src = torch.unique(sampling_src_idx)
    node_mask[unique_src] = True 
    row_mask = node_mask[row] 
    edge_mask = col[row_mask] 
    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])#b_idx 是一个与边数相同长度的张量，它确定了每个边所属的批次（源节点）。这样，当后续代码使用 to_dense_batch 函数将边组织成批次时，每个批次都包含来自同一个源节点的边。
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)#使用 to_dense_batch 函数将 edge_mask 根据批次索引 b_idx 转换成一个稠密的边张量 edge_dense。其中每一行表示一个批次，每一列表示一个节点，元素的值是节点索引，表示节点之间的连接关系。小于最大度数的填充为-1。
    if len(temp[temp!=0]) != edge_dense.shape[0]:
        cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
        cut_temp = temp[temp!=0][:-cut_num]
    else:
        cut_temp = temp[temp!=0]
    edge_dense = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense!= -1]#根据 edge_dense 去除填充值为 -1 的部分变一维，得到新的列索引 new_col
    inv_edge_index = torch.stack([new_col, new_row], dim=0)#创建一个新的边索引张量 inv_edge_index，将新的行索引 new_row 和列索引 new_col 堆叠在一起，以表示新的边。
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)#最后，将新的边索引 inv_edge_index 与原始的边索引 edge_index 水平连接，创建一个包含新边的边列表 new_edge_index。

    return new_edge_index

@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    
    # Find the nearest nodes and mix target pool  sampling_src_idx的节点的和其他的节点的距离
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)#会进行自动排序

    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)

    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    #degree_dist 中的每个元素表示具有相同度数的节点数量。degree_dist 的目的是为了从中采样新节点的度数，以便在生成新节点的边时使用。
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    #shape(len(sampling_src_idx), num_nodes)degree_dist.unsqueeze(dim=0)：这一步在degree_dist张量的维度0上添加一个维度，将其从一维张量变成了二维矩阵。现在，degree_dist的形状变为(1, num_nodes)，其中第一个维度是 1。
    #.repeat(len(sampling_src_idx), 1)：接下来，这个二维矩阵会被沿着维度 0重复len(sampling_src_idx)次，即生成一个形状为(len(sampling_src_idx), num_nodes) 的新矩阵。这个新矩阵的每一行都是degree_dist张量的拷贝，一共有len(sampling_src_idx)行，对应于要生成的新节点的数量。
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    #aug_degree 张量中存储了每个新节点的度数信息,每一行都作为一个分布，所以输出的是len(sampling_src_idx)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])
    # edge_index11 = torch.tensor([
    #     [0, 1, 1, 2, 3, 3, 4],  # 源节点的标识符
    #     [1, 2, 3, 0, 2, 4, 0]  # 目标节点的标识符
    # ])
    # col11 = edge_index11[1]
    # print(scatter_add(torch.ones_like(col11), col11))-->tensor([2, 1, 2, 1, 1])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]#这行代码利用tgt_index和aug_degree，筛选出符合条件的新目标节点索引new_col。它的目的是确保每个源节点的新目标节点数量不超过其预先采样的度数。
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)#将new_row中的每个源节点索引重复多次，重复的次数由对应的aug_degree值决定
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

@torch.no_grad()
def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False, no_mask=False):
    max_num, n_cls = max(class_num_list), len(class_num_list) 
    if not max_flag: # mean
        max_num = sum(class_num_list) / n_cls
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)

    prev_out_local = F.softmax(prev_out_local/tau, dim=1)
    prev_out_local = prev_out_local.cpu() 

    src_idx_all = []
    dst_idx_all = []
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0: 
            continue

        # first sampling --vanc

        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze()
        if prob.dim() == 0:
            prob = prob.unsqueeze(0)

        src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True)#以从概率分布 prob 中选择 num 个样本，返回的是prob的下标
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]] 

        # second sampling --caux
        conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]] #conf_src 是一个概率分布，它表示第一次采样中选中的源节点对应的每个节点在当前类别 cls_idx 下的置信度。
        if not no_mask:
            conf_src[:,cls_idx] = 0
        neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze().tolist() 

        # third sampling --vaux
        neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls] 
        dst_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0] 
            dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)

        src_idx_all.append(src_idx)
        dst_idx_all.append(dst_idx)
    
    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)
    
    return src_idx_all, dst_idx_all, sampling_list #vanc,vaux

