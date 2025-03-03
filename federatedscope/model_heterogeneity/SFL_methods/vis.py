import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from federatedscope.model_heterogeneity.SFL_methods.simple_tshe import *


def t_she_vis(node_emb, emb_labels,node_aug_emb_all,node_aug_labels_all, glob_protos, local_agg_protos_dict, state=0, client_id=None,
              show_local_proto=False, show_split=[], multiple_client=True,
              trainable_global_protos=False, K=1, num_class=7, show_global_proto=True):
    '''
    用t-she可视化某一个client所有节点的embedding以及学到的全局原型(global prototype)
    Args:
        client_emb_all: (N,d) 二维张量
        emb_labels:  client_emb_all对应的标签
        glob_protos: (class_num,d) C是样本数
        state: 代表可视化第几轮的结果
    '''
    client_node_num = []
    node2client = []
    client_num = len(node_emb) if multiple_client else 1
    if isinstance(node_emb, torch.Tensor) and isinstance(emb_labels, torch.Tensor) :
        # node_emb: (node_num, feature_dim)
        # emb_labels: (node_num,)
        embedding_tensor = node_emb
        node_emb_num = embedding_tensor.shape[0]
    elif isinstance(node_emb, list) and isinstance(emb_labels, list):
        # 在一张图上可视化多个client的节点embedding以及全局原型
        # 此时node_emb 和emb_labes 是dict，他们的key都是client_id
        # node_emb[client_id]存放这个client的所有node embedding; emb_labes[client_id]则存放对应节点的labels

        temp_emb_list = []
        temp_label_list = []
        for myid, emb in enumerate(node_emb):
            temp_emb_list.append(emb)
            client_node_num.append(len(emb))
            node2client.extend([myid] * len(emb))
        for label_dict in emb_labels:
            temp_label_list.append(label_dict)
        # for myid, emb in enumerate(node_aug_emb_all):
        #     temp_emb_list.append(emb)
        #     client_node_num.append(len(emb))
        #     node2client.extend([myid] * len(emb))
        # for label_dict in node_aug_labels_all:
        #     temp_label_list.append(label_dict)
        embedding_tensor = torch.cat(temp_emb_list, dim=0)
        node_emb_num = len(embedding_tensor)
        emb_labels = torch.cat(temp_label_list, dim=0)
    glob_proto_tensor=[]
    key_list=[]
    if show_global_proto:
        if trainable_global_protos:
            glob_proto_tensor = torch.cat(list(glob_protos))
            group_index = [i // K for i in range(len(glob_proto_tensor))]
            label_temp = torch.Tensor(group_index).long().to('cuda:1')
        elif isinstance(glob_protos, dict):
            for key, tensor_list in glob_protos.items():
                for tensor in tensor_list:
                    glob_proto_tensor.append(tensor)
                    key_list.append(key)
            glob_proto_tensor =torch.cat(glob_proto_tensor, dim=0)
            label_temp = torch.Tensor(key_list).long().to('cuda:1')

        embedding_tensor = torch.cat(
            [embedding_tensor.to('cuda:1'), glob_proto_tensor.to('cuda:1')])  # global prototype 和 embedding_tensor 做拼接
        emb_labels = torch.cat([emb_labels.to('cuda:1'), label_temp])  # 将global prototype的标签添加至emb_labels的末尾

        # 将全局原型的客户端标签设定为客户端总数
        node2client.extend([client_num] * len(label_temp))

        global_proto_num = len(glob_proto_tensor)
    else:
        global_proto_num = 0

    if show_local_proto:
        # 增加本地原型embedding和对应的标签
        if multiple_client:
            for client_id, proto_dict in local_agg_protos_dict.items():
                for proto_label, proto in proto_dict.items():
                    embedding_tensor = torch.cat([embedding_tensor, proto.unsqueeze(0)])
                    emb_labels = torch.cat([emb_labels, torch.tensor(proto_label).unsqueeze(0).to('cuda:4')])
                    node2client.append(client_num)
        else:
            for local_proto_label, proto in local_agg_protos_dict.items():
                if isinstance(proto, list):
                    # 每个client的每个class有多个本地原型（质心）
                    temp = torch.cat(proto, dim=0).to('cuda:4')
                    temp_label = torch.full((len(proto),), local_proto_label).long().to('cuda:4')
                    embedding_tensor = torch.cat([embedding_tensor, temp])
                    emb_labels = torch.cat([emb_labels, temp_label])
                elif isinstance(proto, torch.Tensor) and proto.dim() == 1:
                    # 每个client的每个class只有一个本地原型（质心）
                    embedding_tensor = torch.cat([embedding_tensor, proto.unsqueeze(0)])
                    tmp_label = torch.tensor(local_proto_label).to('cuda:4')
                    emb_labels = torch.cat([emb_labels, tmp_label.unsqueeze(0)])

            node2client.extend([0] * embedding_tensor.shape[0])  # 如果一次只可视化一个client，那么指定cient编号为0

    node2client = np.array(node2client)  # TODO: 将本地原型的client标签加进node2client

    # 嵌入向量数据和对应的标签
    embeddings = embedding_tensor.cpu().numpy()
    labels = emb_labels.cpu().numpy()

    # 使用t-SNE算法降维为2维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # 根据标签分组
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = plt.get_cmap('tab10')

    # 画图
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
    if client_id is None:
        client_id = 'None'
    for i in range(num_labels):
        # 每个节点颜色是对应的标签
        mask1 = labels == unique_labels[i]
        mask2 = np.zeros(len(labels), dtype=bool)
        mask3 = mask2.copy()

        mask2[:node_emb_num] = True
        mask3[node_emb_num:node_emb_num + global_proto_num] = True

        mask_emb = mask1 & mask2
        mask_glob_proto = mask1 & mask3
        axs.scatter(embeddings_tsne[mask_emb, 0], embeddings_tsne[mask_emb, 1], color=colors(i),
                       label=str(unique_labels[i]),
                       s=10)
        axs.scatter(embeddings_tsne[mask_glob_proto, 0], embeddings_tsne[mask_glob_proto, 1], color=colors(i),
                       # label='glob_' + str(unique_labels[i]),
                       s=70, marker='^', edgecolors='black', zorder=3)
        if show_local_proto:
            mask4 = np.zeros(len(labels), dtype=bool)
            mask4[node_emb_num + global_proto_num:] = True
            mask_local_proto = mask1 & mask4
            axs.scatter(embeddings_tsne[mask_local_proto, 0], embeddings_tsne[mask_local_proto, 1],
                           color=colors(i),
                           label='local_proto_' + str(unique_labels[i]),
                           s=50, marker='*', edgecolors='w')
        for spine in axs.spines.values():
            spine.set_linewidth(0.5)  # Set the desired linewidth (e.g., 0.5 points)

        # axs[0].legend()
    # for i in np.unique(node2client):
    #     mask = node2client == i
    #     if i != client_num:  # 当node2client中的val为 client+1 时，代表这个节点为prototype
    #         axs[1].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], color=colors(i),
    #                        label='client '+ str(i+1),
    #                        s=10)
    #     else:
    #         pass
            # if multiple_client:
            #     axs[1].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], color=colors(i),
            #                    label='prototype',
            #                    s=50, marker='^', edgecolors='w')
        # axs[1].set_title(f'Rounds: {state}')
        # axs[1].legend()
    # 图例
    # 自定义图例
    # from matplotlib.lines import Line2D
    # from matplotlib.path import Path
    #
    # triangle = Line2D([0], [0], color='white', marker='^', markerfacecolor='none', markeredgecolor='black',
    #                   markersize=10)
    # circle = Line2D([0], [0], color='white', marker='o', markerfacecolor='none', markeredgecolor='black', markersize=10)
    # axs[0].legend([triangle], ['Global Prototypes'])
    # axs[1].legend()
    #隐藏坐标轴
    plt.xticks([])
    plt.yticks([])
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)

    # Save or show the plot
    fig.savefig(f"/data/dzh2023/SEFGL/federatedscope/model_heterogeneity/figure_fedproto/scatter_plot_round_{state}.pdf", bbox_inches='tight')



def load_emb_files(file_path, client_id):
    node_emb_dict = torch.load(f'{file_path}/local_node_embdeddings_on_client_{client_id}.pth')
    node_labels = torch.load(f'{file_path}/node_labels_on_client_{client_id}.pth')
    node_aug_emb_dict = torch.load(f'{file_path}/local_node_aug_embdeddings_on_client_{client_id}.pth')
    node_aug_labels = torch.load(f'{file_path}/node_aug_labels_on_client_{client_id}.pth')
    glob_protos_dict = torch.load(f'{file_path}/global_protos_on_client_{client_id}.pth')
    local_agg_protos_dict = torch.load(f'{file_path}/agg_protos_on_client_{client_id}.pth')
    client_raw_data = torch.load(f'{file_path}/raw_data_on_client_{client_id}.pth')
    return node_emb_dict, node_labels,node_aug_emb_dict,node_aug_labels, glob_protos_dict, local_agg_protos_dict, client_raw_data


def load_graph_files(file_path, client_id):
    client_raw_data = torch.load(f'{file_path}/raw_data_on_client_{client_id}.pth')
    client_generated_data = torch.load(f'{file_path}/generated_data_on_client_{client_id}.pth')
    node_labels = torch.load(f'{file_path}/node_labels_on_client_{client_id}.pth')
    return client_raw_data, client_generated_data, node_labels


def vis_emb_single_client(file_path, client_IDs, rounds=[1, 10, 50], trainable_global_protos=False,
                          number_proto_per_class=1, num_class=7, show_local_proto=False, show_global_proto=True):
    print(file_path)
    for idx in client_IDs:
        node_emb_dict, node_labels, glob_protos_dict, \
        local_agg_protos_dict, client_raw_data = load_emb_files(file_path,
                                                                idx)

        for rnd in rounds:
            glob_protos = glob_protos_dict[rnd] if show_global_proto else dict()

            t_she_vis(node_emb=node_emb_dict[rnd],
                      emb_labels=node_labels[rnd],
                      glob_protos=glob_protos,
                      local_agg_protos_dict=local_agg_protos_dict[rnd],
                      state=rnd,
                      client_id=idx,
                      multiple_client=False,
                      trainable_global_protos=trainable_global_protos,
                      K=number_proto_per_class,
                      num_class=num_class,
                      show_local_proto=show_local_proto,
                      show_global_proto=show_global_proto)


def vis_emb_multiple_clients(file_path, client_IDs, rounds=[1, 10, 50], trainable_global_protos=False,
                             number_proto_per_class=1, num_class=7, show_local_proto=False, show_global_proto=True,
                             different_local_rounds=[]):
    for rnd in rounds:
        node_emb_all,node_aug_emb_all, node_labels_all,node_aug_labels_all = [], [], [], []
        local_protos_all = dict()
        client_ids_str = ""

        for idx in client_IDs:
            node_emb_dict, node_labels,node_aug_emb_dict,node_aug_labels, glob_protos_dict, local_agg_protos_dict, _ = load_emb_files(file_path, idx)
            node_emb_all.append(node_emb_dict[rnd])
            node_labels_all.append(node_labels[rnd])
            node_aug_emb_all.append(node_aug_emb_dict[rnd])
            node_aug_labels_all.append(node_aug_labels[rnd])
            local_protos_all[idx] = local_agg_protos_dict[rnd]
            client_ids_str += str(idx) + ' '

        glob_protos = glob_protos_dict[rnd] if show_global_proto else dict()

        t_she_vis(
            node_emb=node_emb_all,
            emb_labels=node_labels_all,
            node_aug_emb_all=node_aug_emb_all,
            node_aug_labels_all=node_aug_labels_all,
            glob_protos=glob_protos,
            local_agg_protos_dict=local_protos_all,
            state=rnd,
            client_id=client_ids_str,
            multiple_client=True,
            trainable_global_protos=trainable_global_protos,
            K=number_proto_per_class,
            num_class=num_class,
            show_local_proto=show_local_proto,
            show_global_proto=show_global_proto
        )

    if len(different_local_rounds) == len(client_IDs):
        node_emb_all, node_labels_all = [], []
        local_protos_all = dict()
        client_ids_str = ""
        for idx in client_IDs:
            rnd = different_local_rounds[idx - 1]

            node_emb_dict, node_labels, glob_protos_dict, local_agg_protos_dict, _ = load_emb_files(file_path, idx)
            node_emb_all.append(node_emb_dict[rnd])
            node_labels_all.append(node_labels[rnd])
            local_protos_all[idx] = local_agg_protos_dict[rnd]
            client_ids_str += str(idx) + ' '

        show_global_proto = False
        glob_protos = dict()

        t_she_vis(
            node_emb=node_emb_all,
            emb_labels=node_labels_all,
            glob_protos=glob_protos,
            local_agg_protos_dict=local_protos_all,
            state=different_local_rounds,
            client_id=client_ids_str,
            multiple_client=True,
            trainable_global_protos=trainable_global_protos,
            K=number_proto_per_class,
            num_class=num_class,
            show_local_proto=True,
            show_global_proto=show_global_proto
        )


def vis_generated_graph_multiple_clients(file_path, client_IDs, rounds=[1, 10, 50], num_classes=7, selected_labels=[],
                                         **kwargs):
    for rnd in rounds:

        for idx in client_IDs:
            client_raw_data, client_generated_data, node_labels = load_graph_files(file_path, idx)
            train_mask = client_raw_data['data'].train_mask

            raw_x = client_raw_data['data'].x[train_mask]
            new_x = client_generated_data[rnd].x[train_mask]

            embedding = torch.cat((raw_x, new_x))
            labels = torch.cat(
                (client_raw_data['data'].y[train_mask], client_raw_data['data'].y[train_mask] + num_classes))

            client_raw_data['data'].y[train_mask].repeat(2, 1)
            TSHE_generated_graph(embeddings=embedding, labels=labels, client_id=idx, round=rnd, num_classes=num_classes,
                                 selected_labels=selected_labels,
                                 only_generated=kwargs['only_generated'])


def vis_PP_node_embedding_multiple_clients(file_path, client_IDs, rounds=[1, 10, 50], selected_labels=[],
                                           **kwargs):
    """
    用 T-SNE可视化基于原型传播(PP)的节点嵌入以及主GNN输出的节点嵌入
    """

    for rnd in rounds:

        for idx in client_IDs:
            node_emb = torch.load(f'{file_path}/local_node_embdeddings_on_client_{idx}.pth')
            PP_node_emb = torch.load(f'{file_path}/PP_node_embeddings_on_client_{idx}.pth')
            node_labels = torch.load(f'{file_path}/node_labels_on_client_{idx}.pth')
            glob_protos_dict = torch.load(f'{file_path}/global_protos_on_client_{idx}.pth')
            local_agg_protos_dict = torch.load(f'{file_path}/agg_protos_on_client_{idx}.pth')

            node_emb_gnn = node_emb[rnd]
            node_emb_pp = PP_node_emb[rnd]
            embedding = torch.cat((node_emb_gnn, node_emb_pp))
            labels = node_labels[rnd].repeat(2)

            TSNE_vis_gnn_and_pp_node_embeddings(embeddings=embedding, labels=labels, client_id=idx, round=rnd,
                                                selected_labels=selected_labels, show_global_proto=True,
                                                glob_protos=glob_protos_dict[rnd])


from IPython import display
from matplotlib_inline import backend_inline


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # d2l.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
        axes.grid()


# mixup 相关函数
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


print(os.getcwd())
filepath = '/data/dzh2023/SEFGL/federatedscope/model_heterogeneity/embedding'
# vis_PP_node_embedding_multiple_clients(
#     filepath,
#     client_IDs=[1],
#     rounds=[i for i in range(1, 77)],
#     selected_labels=[0, 1, 2, 3, 4, 5, 6],
# )

# vis_generated_graph_multiple_clients(
#     file_path=filepath,
#     client_IDs=[1],
#     rounds=[i for i in range(1,51)],
#     num_classes=7,
#     selected_labels=[0,1,2,3,4,5,6],
#     only_generated=True
# )

vis_emb_multiple_clients(
    file_path=filepath,
    client_IDs=[1,2,3,4,5,6,7,8,9,10],
    rounds=[18],
    trainable_global_protos=False,
    number_proto_per_class=1,
    num_class=7,
    show_local_proto=False,
    show_global_proto=False,
    # different_local_rounds=[18]
)

# vis_emb_single_client(
#     file_path=filepath,
#     client_IDs=[1, 2, 3],
#     rounds=[1, 10, 50],
#     trainable_global_protos=False,
#     number_proto_per_class=1,
#     num_class=7,
#     show_local_proto=True,
#     show_global_proto=False)
