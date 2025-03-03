from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, MODE
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.message import Message
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
import torch.nn as nn
import copy
import logging
import torch
import datetime
from collections import OrderedDict, defaultdict
import numpy as np
from torch_scatter import scatter_add
from federatedscope.contrib.utils.neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist
from federatedscope.contrib.utils.gens import sampling_node_source, neighbor_sampling, duplicate_neighbor, saliency_mixup, sampling_idx_individual_dst


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 将prototype直接用原始特征合成，然后基于graphsha，加入globalprototype生成,0.813050386734156
# Build your trainer here.
class FedProto_Node_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedProto_Node_Trainer, self).__init__(model, data, device, config,
                                                    only_for_eval, monitor)
        self.loss_mse = nn.MSELoss()
        self.proto_weight = self.ctx.cfg.fedproto.proto_weight
        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto,
                                    "on_fit_end")
        self.register_hook_in_train(self._hook_on_before_epochs_for_proto,
                                    "on_fit_start")
        self.register_hook_in_train(self._hook_on_epoch_start_for_proto,
                                    "on_fit_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_proto,
                                   "on_fit_start")

        self.task = config.MHFL.task

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.batch
        global_protos = ctx.global_protos
        ####graphsha
        if ctx.cur_epoch_i>self._cfg.graphsha.warmup:#ctx.cur_epoch_i为现在所处的epoch
            # identifying source samples
            prev_out_local = ctx.prev_out[ctx.train_idx]
            # 返回的是Vanc和Vaux的集合
            sampling_src_idx, sampling_dst_idx, sampling_list = sampling_node_source(ctx.class_num_list, prev_out_local, ctx.idx_info_local,
                                                                      ctx.train_idx, self._cfg.graphsha.tau, max_flag=False, no_mask=False)
            beta = torch.distributions.beta.Beta(1, 100)  # 创建一个Beta分布，1 和 100 的选择意味着 Beta 分布会有一个倾向性，偏向于生成接近于 0 的随机数
            lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)  # 从 Beta 分布中生成一组随机样本，数量等于 len(sampling_src_idx)
            new_x = saliency_mixup(batch.x, sampling_src_idx, sampling_dst_idx, lam, global_protos, sampling_list)
            # Sorting src idx with corresponding dst idx
            sampling_src_idx, sorted_idx = torch.sort(
                sampling_src_idx)  # 第一个返回值是排序后的张量，包含了按升序排列的元素。第二个返回值是一个索引张量，它包含了排序后的张量元素在原始张量中的索引位置。
            sampling_dst_idx = sampling_dst_idx[sorted_idx]
            # semimxup
            new_edge_index = neighbor_sampling(batch.x.size(0), batch.edge_index[:, ctx.train_edge_mask], sampling_src_idx,
                                               ctx.neighbor_dist_list)  # neighbor_dist_list(N,N),邻居距离

        else:
            # 返回源节点和目标节点的索引，目标节点是指源节点的相邻节点
            sampling_src_idx, sampling_dst_idx, sampling_list = sampling_idx_individual_dst(ctx.class_num_list, ctx.idx_info, torch.device('cuda:2'))  # class_num_list每个类的节点数list7，idx_info每个类对应的索引构成的7个tensor
            beta = torch.distributions.beta.Beta(2, 2)  # Beta 分布是一种概率分布，通常用于表示随机变量在 [0, 1] 区间内的概率分布，它具有两个参数：α和 β。
            lam = beta.sample((len(sampling_src_idx),)).unsqueeze(
                1)  # 使用 beta.sample() 方法从 Beta 分布中生成随机样本。参数 (len(sampling_src_idx),) 指定要生成的样本数量，通常用于批处理操作。这将返回一个包含指定数量随机样本的张量，每个样本都是根据 Beta 分布生成的值。
            new_x = saliency_mixup(batch.x, sampling_src_idx, sampling_dst_idx, lam, global_protos, sampling_list)

            # Sorting src idx with corresponding dst idx
            sampling_src_idx, sorted_idx = torch.sort(
                sampling_src_idx)  # 第一个返回值是排序后的张量，包含了按升序排列的元素。第二个返回值是一个索引张量，它包含了排序后的张量元素在原始张量中的索引位置。
            sampling_dst_idx = sampling_dst_idx[sorted_idx]
            # 在保留原始边的情况下，为新节点创建额外的边，以确保新节点连接的度数与源节点的度数相匹配。，新节点的边是通过复制源节点的边来创建的，这意味着新节点与源节点连接到相同的目标节点。
            new_edge_index = duplicate_neighbor(batch.x.size(0), batch.edge_index[:, ctx.train_edge_mask], sampling_src_idx)

        x,edge_index = new_x, new_edge_index
        new_data = (x,edge_index)
        output,_ = ctx.model(new_data)
        prev_out = output[:batch.x.size(0)]
        ctx.prev_out = prev_out

        add_num = len(output) - ctx.data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device=batch.x.device)
        new_train_mask = torch.cat((ctx.data_train_mask, new_train_mask), dim=0)
        _new_y = batch.y[sampling_src_idx].clone()
        new_y = torch.cat((batch.y[ctx.data_train_mask], _new_y), dim=0)
        loss1 = ctx.criterion(output[new_train_mask], new_y)

        ###########################此时本地的原型为数据增强后的原型
        owned_classes = new_y.unique()
        reps = new_x[new_train_mask]
        loss2 = 0 * loss1 #TODO: 测试用，记得删除
        if len(ctx.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            proto_new = copy.deepcopy(reps.data)  # TODO: .data会有问题吗，是不是应该用detach()？
            for cls in owned_classes:
                if cls.item() in ctx.global_protos.keys():
                    proto_new[new_y == cls] = ctx.global_protos[cls.item()]
            loss2 = self.loss_mse(reps, proto_new)

        # if len(ctx.global_protos) != 0:
        #     global_protos = torch.stack(list(ctx.global_protos.values())).detach()
        #     similarity = torch.matmul(reps,  global_protos.T)
        # else:
        #     similarity=pred
        loss = loss1 + loss2 * self.proto_weight

        if ctx.cfg.fedproto.show_verbose:
            logger.info(
                f'client#{self.ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} \t CE_loss:{loss1}'
                f'\t proto_loss:{loss2},\t total_loss:{loss}')


        split_mask = batch[f'{ctx.cur_split}_mask']

        labels = batch.y[split_mask]
        if len(split_mask) < len(output):
            num_to_add = len(output) - len(split_mask)
            padding = [False] * num_to_add
            split_mask = torch.cat((split_mask, torch.tensor(padding).to('cuda:2')))
        pred = output[split_mask]
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)


        ####
        ctx.ys_feature.append(reps.detach().cpu())#用来可视化的
        ####

    def update(self, global_proto, strict=False):
        self.ctx.global_protos = global_proto

    def _hook_on_epoch_start_for_proto(self, ctx):
        """定义一些fedproto需要用到的全局变量"""
        ctx.ys_feature = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_before_epochs_for_proto(self, ctx):

        batch = ctx.data.train_data[0].to('cuda:2')

        ###########################
        n_cls = batch.y.max().item() + 1  # 其中 data.y 是节点标签的张量。首先，data.y.max() 返回标签中的最大值，然后通过 .item() 方法将其转换为标量，并最后加上 1，以获得类数。这是为了确保类别从 0 开始计数。
        stats = batch.y[batch.train_mask]
        n_data = []
        for i in range(n_cls):
            data_num = (
                        stats == i).sum()  # 它使用 (stats == i) 来创建一个布尔类型的掩码，其中为 True 的位置表示节点的标签与类别 i 匹配。然后，.sum() 函数用于计算掩码中 True 的数量，即属于类别 i 的数据点数量。
            n_data.append(int(data_num.item()))  # 每个类别数量
        idx_info = get_idx_info(batch.y, n_cls, batch.train_mask)
        class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
            make_longtailed_data_remove(batch.edge_index, batch.y, n_data, n_cls, self._cfg.graphsha.imb_ratio,
                                        batch.train_mask)  # 去掉长尾数据后的数据
        train_idx = data_train_mask.nonzero().squeeze()  # 训练集的数据索引
        labels_local = batch.y.view([-1])[train_idx]  # 打平变一维后的训练集label
        train_idx_list = train_idx.cpu().tolist()  # 训练集的数据索引
        local2global = {i: train_idx_list[i] for i in
                        range(len(train_idx_list))}  # {0: 0, 1: 4, 2: 5, 3: 7, 4: 8, 5: 13
        global2local = dict([val, key] for key, val in local2global.items())  # {0: 0, 4: 1, 5: 2, 7: 3, 8: 4, 13: 5
        idx_info_list = [item.cpu().tolist() for item in idx_info]  # 每个类对应的索引构成的7个list
        idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                          idx_info_list]  # 根据每个7类对应的list索引，获得7个tensor
        labels_local = batch.y.view([-1])[train_idx]
        if self._cfg.graphsha.gdc == 'ppr':
            neighbor_dist_list = get_PPR_adj(batch.x, batch.edge_index[:, train_edge_mask], alpha=0.05, k=128, eps=None)
            ctx.neighbor_dist_list = neighbor_dist_list
        saliency, prev_out = None, None
        ctx.prev_out = prev_out
        ctx.idx_info = idx_info
        ctx.class_num_list = class_num_list
        ctx.data_train_mask = data_train_mask
        ctx.data_val_mask = batch.val_mask.clone()
        ctx.data_test_mask = batch.test_mask.clone()
        ctx.saliency = saliency
        ctx.train_idx = train_idx
        ctx.train_edge_mask = train_edge_mask
        ctx.idx_info_local = idx_info_local
        ctx.batch = batch

    #计算被本地原型
    def _hook_on_fit_end_agg_local_proto(self, ctx):
        reps_dict = defaultdict(list)
        agg_local_protos = dict()
        ctx.train_loader.reset()

        for batch_idx in range(ctx.num_train_batch):
            batch = next(ctx.train_loader).to(ctx.device)
            split_mask = batch['train_mask']
            labels = batch.y[split_mask]
            reps_all = batch.x
            reps = reps_all[split_mask]


            owned_classes = labels.unique()
            for cls in owned_classes:
                filted_reps = reps[labels == cls].detach()
                reps_dict[cls.item()].append(filted_reps)

        for cls, protos in reps_dict.items():
            mean_proto = torch.cat(protos).mean(dim=0)
            agg_local_protos[cls] = mean_proto

        ctx.agg_local_protos = agg_local_protos

        # t-she可视化用
        if ctx.cfg.vis_embedding:
            ctx.node_emb_all = reps_all.clone().detach()
            ctx.node_labels = batch.y.clone().detach()


    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        training_begin_time = datetime.datetime.now()

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,#调用_hook_on_batch_forward
                                        target_data_split_name)

        training_end_time = datetime.datetime.now()
        training_time = training_end_time-training_begin_time
        self.ctx.monitor.track_training_time(training_time)  # 记录每次本地训练的训练时间

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_local_protos
    ########################################
def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(
        len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1 / ratio, 1 / (n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1:  # We does not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1, n_round[i] + 1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list, [])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(
                row.device)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            _, remove_idx = torch.topk(degree, (r * remove_class_num_list[i]) // n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list, [])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask.to('cuda:2'), idx_info, node_mask, edge_mask
def call_my_trainer(trainer_type):
    if trainer_type == 'fedproto_node_trainer':
        trainer_builder = FedProto_Node_Trainer
        return trainer_builder


register_trainer('fedproto_node_trainer', call_my_trainer)
