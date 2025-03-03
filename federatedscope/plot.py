import matplotlib.pyplot as plt
import numpy as np

# 数据输入
# clients = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
# clients = ['1', '2', '3', '4', '5', '6', '7']
clients = ['1', '2', '3', '4', '5']

categories = ['1', '2', '3', '4', '5', '6', '7']

data = np.array([[38, 13, 202, 7, 3, 23, 72],
                 [92, 15, 0, 61, 26, 61, 17],
                 [6, 161, 3, 42, 58, 80, 43],
                 [33, 8, 6, 96, 66, 9, 5],
                 [63, 20, 21, 26, 79, 59, 43]])

# data = np.array([[6, 11, 195, 2, 3, 8, 16],
#                  [9, 8, 0, 35, 63, 77, 8],
#                  [8, 161, 3, 53, 3, 2, 28],
#                  [42, 20, 9, 72, 19, 29, 13],
#                  [43, 7, 14, 27, 31, 88, 46],
#                  [42, 4, 10, 20, 63, 22, 66],
#                  [82, 6, 1, 23, 50, 6, 3]])

# data = np.array([[10, 6, 130, 7, 1, 2, 0],
#                  [14, 11, 6, 41, 25, 4, 4],
#                  [6, 165, 4, 15, 5, 5, 4],
#                  [41, 16, 3, 43, 7, 28, 10],
#                  [36, 2, 4, 11, 24, 89, 47],
#                  [35, 1, 6, 12, 49, 18, 91],
#                  [80, 2, 0, 14, 50, 1, 1],
#                  [3, 6, 11, 4, 51, 81, 19],
#                  [1, 8, 66, 35, 5, 2, 2],
#                  [6, 0, 2, 50, 15, 2, 2]])

# 创建气泡图
fig, ax = plt.subplots(figsize=(10, 8))

# 为每个客户端和类别绘制气泡
for i, client in enumerate(clients):
    for j, category in enumerate(categories):
        ax.scatter(i, j, s=data[i, j] * 10, alpha=0.5, color='orange')

# 设置 x 轴和 y 轴的标签和刻度
ax.set_xticks(np.arange(len(clients)))
ax.set_xticklabels(clients, fontdict={'fontname': 'Times New Roman', 'fontsize': 20})
ax.set_yticks(np.arange(len(categories)))
ax.set_yticklabels(categories, fontdict={'fontname': 'Times New Roman', 'fontsize': 20})

font = {'family': 'Times New Roman', 'size':40}
ax.set_xlabel('Clients', fontdict=font)
ax.set_ylabel('Categories', fontdict=font)

plt.show()