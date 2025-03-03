import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch


def simple_TSHE(embeddings, labels,aug_embeddings, aug_labels,global_protos, client_id=0, round=0):
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    aug_embeddings = aug_embeddings.cpu().numpy()
    aug_labels = aug_labels.cpu().numpy()

    global_proto_labels = np.array(list(global_protos.keys()))
    global_proto_embeddings = [tensor.cpu().numpy() for tensor in global_protos.values()]
    global_proto_embeddings = np.array(global_proto_embeddings)

    combined_embeddings = np.concatenate((embeddings, aug_embeddings, global_proto_embeddings), axis=0)

    # 将 labels 和 aug_labels 沿着行方向拼接
    combined_labels = np.concatenate((labels, global_proto_labels), axis=0)



    # 使用t-SNE算法降维为2维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(combined_embeddings)

    # 根据标签分组
    unique_labels = np.unique(combined_labels)
    num_labels = len(unique_labels)
    colors = plt.get_cmap('tab10')

    # 画图
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)

    for i in range(num_labels):
        label_mask = labels == unique_labels[i]
        aug_label_mask = aug_labels == unique_labels[i]
        global_label_mask = global_proto_labels == unique_labels[i]

        axs.scatter(embeddings_tsne[:len(embeddings)][label_mask, 0], embeddings_tsne[:len(embeddings)][label_mask, 1], color=colors(i),
                    label=str(unique_labels[i]), marker='o', s=10)  # 使用圆形标记
        # axs.scatter(embeddings_tsne[len(embeddings):len(embeddings)+len(aug_embeddings)][aug_label_mask, 0], embeddings_tsne[len(embeddings):len(embeddings)+len(aug_embeddings)][aug_label_mask, 1], color=colors(i),
        #             marker='s', s=10, edgecolors='black', linewidths=1)  # 使用正方形标记
        axs.scatter(embeddings_tsne[len(embeddings)+len(aug_embeddings):][global_label_mask, 0], embeddings_tsne[len(embeddings)+len(aug_embeddings):][global_label_mask, 1], color=colors(i),
                    marker='^', s=50, edgecolors='black', linewidths=1.5, label=f'Global Proto {i}')
    for spine in axs.spines.values():
        spine.set_linewidth(0.5)
    # axs.set_title(f'Client #{client_id} rounds {round} class')
    plt.xticks([])
    plt.yticks([])
    # if round % 10 == 0:
    print(f'/data/dzh2023/SEFGL/federatedscope/model_heterogeneity/result/figure/Client_#{client_id}_rounds_{round}.pdf')
    plt.savefig(f'/data/dzh2023/SEFGL/federatedscope/model_heterogeneity/result/figure/Client_#{client_id}_rounds_{round}.pdf',bbox_inches='tight')


def TSHE_generated_graph(embeddings, labels, client_id=0, round=0, selected_labels=[0], num_classes=7,
                         only_generated=False):
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    # 使用t-SNE算法降维为2维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # 根据标签分组
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('tab10')

    # 画图
    fig, ax = plt.subplots()

    for i in selected_labels:
        # 每个节点颜色是对应的标签
        label_mask = labels == i
        label_mask_2 = labels == i + num_classes
        if not only_generated:
            ax.scatter(
                embeddings_tsne[label_mask, 0], embeddings_tsne[label_mask, 1], color=colors(i),
                label='raw_' + str(unique_labels[i]), zorder=3, s=30, marker='^', edgecolors='black')
        ax.scatter(
            embeddings_tsne[label_mask_2, 0], embeddings_tsne[label_mask_2, 1], color=colors(i),
            label='g_' + str(unique_labels[i]), marker='+', s=30)
        ax.set_title(f'Client #{client_id} rounds {round} class')
        ax.legend()
    # plt.legend()
    plt.show()


def TSNE_vis_gnn_and_pp_node_embeddings(embeddings, labels, client_id=0, round=0, selected_labels=[0],
                                        show_global_proto=False, glob_protos=None):
    from matplotlib.lines import Line2D

    num_nodes = len(labels) // 2

    if show_global_proto:
        glob_proto_tensor = torch.stack(list(glob_protos.values())).to('cuda:0')
        label_proto = torch.Tensor(list(glob_protos.keys())).long().to('cuda:0')

        embeddings = torch.cat([embeddings, glob_proto_tensor.to('cuda:0')])
        labels = torch.cat([labels, label_proto])
        global_proto_num = len(glob_proto_tensor)


    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    # 使用t-SNE算法降维为2维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # 根据标签分组
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('tab10')

    # 画图
    fig, ax = plt.subplots()

    # 自定义图例
    legend_handles = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor='green', markersize=10, label='PP node embeddings'),
        Line2D([0], [0], color='w', marker='^', markerfacecolor='red', markersize=10, label='GNN node embeddings')
    ]

    zeros_mask = np.zeros(len(labels), dtype=bool)

    for i in selected_labels:
        label_mask = labels == i

        pp_mask = zeros_mask.copy()
        pp_mask[num_nodes:] = True

        node_mask = zeros_mask.copy()
        node_mask[:num_nodes*2]=True


        gnn_node_mask = ~pp_mask & label_mask & node_mask
        pp_node_mask = pp_mask & label_mask & node_mask
        global_proto_mask = ~node_mask  & label_mask

        # 可视化GNN生成的node embeddings
        ax.scatter(embeddings_tsne[gnn_node_mask, 0], embeddings_tsne[gnn_node_mask, 1], color=colors(i),
                   label='gnn_emb_' + str(unique_labels[i]), zorder=2, s=30, marker='^')

        # 可视化基于原型传播生成的 node embeddings
        ax.scatter(embeddings_tsne[pp_node_mask, 0], embeddings_tsne[pp_node_mask, 1], color=colors(i),
                   label='pp_emb_' + str(unique_labels[i]), marker='+', s=30)

        # 可视化当前class 的全局原型
        ax.scatter(embeddings_tsne[global_proto_mask, 0], embeddings_tsne[global_proto_mask, 1], color=colors(i),
                   label='global_proto_' + str(unique_labels[i]), marker='*', s=120, edgecolors='black',zorder=3)

        ax.set_title(f'Client #{client_id} rounds {round} class')
        # ax.legend()
    # plt.legend()
    ax.legend(handles=legend_handles, frameon=False, numpoints=1)
    plt.show()
