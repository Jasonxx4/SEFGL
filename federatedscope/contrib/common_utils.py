import copy
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import logging
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image
import pickle
import base64
import torch_geometric
import shutil
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)


# TODO: 过多的函数/方法被放到了这个类里，代码耦合度高。需要整理、归类

def plot_num_of_samples_per_classes(data, modified_cfg, scaling=5.0):
    if isinstance(data[0]['train'], torch_geometric.loader.dataloader.DataLoader):
        label_statistics_for_graph_dataset(data, modified_cfg, scaling)
    else:
        label_statistics_for_cv_dataset(data, modified_cfg, scaling)


def label_statistics_for_cv_dataset(data, modified_cfg, scaling):
    client_num = modified_cfg.federate.client_num
    class_num = modified_cfg.model.out_channels
    client_list = [i for i in range(1, client_num + 1)]
    train_label_distribution = {i: {j: 0 for j in range(class_num)} for i in range(1, client_num + 1)}
    test_label_distribution = copy.deepcopy(train_label_distribution)
    fig, axs = fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # 输出训练集标签分布
    for idx in range(1, client_num + 1):
        train_dataset = data[idx].train_data
        train_label_distribution_new = [j[1].item() if isinstance(j[1], torch.Tensor) else j[1] for j in train_dataset]
        train_label_distribution_new = dict(Counter(train_label_distribution_new))
        train_label_distribution[idx].update(train_label_distribution_new)

        size = [count / scaling for count in list(train_label_distribution[idx].values())]

        axs[0].scatter([idx] * class_num, list(train_label_distribution[idx].keys()),
                       s=size, color='red')
    axs[0].set_title(f'Train Dataset label distribution')

    # 输出验证集标签分布
    if modified_cfg.data.local_eval_whole_test_dataset:
        dataset = data[1].test_data
        test_label_distribution_new = [j[1] for j in dataset]
        print(Counter(test_label_distribution_new))
    else:
        for idx in range(1, client_num + 1):
            test_dataset = data[idx].test_data
            test_label_distribution_new = [j[1].item() if isinstance(j[1], torch.Tensor) else j[1] for j in
                                           test_dataset]
            test_label_distribution_new = dict(Counter(test_label_distribution_new))
            test_label_distribution[idx].update(test_label_distribution_new)
            axs[1].scatter([idx] * class_num, list(test_label_distribution[idx].keys()),
                           s=list(test_label_distribution[idx].values()), color='blue')

            print(f" Client:{idx}, {test_label_distribution_new}")

    axs[1].set_title(f'Test Dataset label distribution')
    plt.show()


def  label_statistics_for_graph_dataset(data, modified_cfg, scaling):
    client_num = modified_cfg.federate.client_num
    class_num = modified_cfg.model.num_classes

    client_list = [i for i in range(1, client_num + 1)]
    train_label_distribution = {i: {j: 0 for j in range(class_num)} for i in range(1, client_num + 1)}
    test_label_distribution = copy.deepcopy(train_label_distribution)
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(1, 1, figsize=(5, 4), sharex=True, sharey=True)
    # 输出训练集标签分布
    print(f"The training set label distribution")
    for idx in range(1, client_num + 1):
        graph_data = data[idx]['data']
        train_mask = graph_data.train_mask
        train_label = graph_data.y[train_mask]
        train_label_distribution_new = [j.item() if isinstance(j, torch.Tensor) else j[1] for j in train_label]
        train_label_distribution_new = dict(Counter(train_label_distribution_new))
        train_label_distribution[idx].update(train_label_distribution_new)

        size = [count / scaling for count in list(train_label_distribution[idx].values())]

        axs.scatter([idx] * class_num, list(train_label_distribution[idx].keys()),
                       s=size, color='red')
        sorted_label_dict = dict(sorted(train_label_distribution_new.items()))
        print(f"Client {idx}: {sorted_label_dict}")
    # axs.set_title(f'Train Dataset label distribution of Cora')
    plt.xticks(range(1, client_num + 1))
    plt.yticks(range(0, 7))
    plt.savefig('/data/dzh2023/FGPL1/federatedscope/model_heterogeneity/result/dataDistribution/Train_Dataset_label_distribution_Cora.pdf')
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(5, 4), sharex=True, sharey=True)
    # 输出测试集标签分布
    print(f"The test set label distribution")
    for idx in range(1, client_num + 1):
        graph_data = data[idx].test_data[0]
        test_mask = graph_data.test_mask
        test_label = graph_data.y[test_mask]

        test_label_distribution_new = [j.item() if isinstance(j, torch.Tensor) else j[1] for j in
                                       test_label]
        test_label_distribution_new = dict(Counter(test_label_distribution_new))
        test_label_distribution[idx].update(test_label_distribution_new)
        axs.scatter([idx] * class_num, list(test_label_distribution[idx].keys()),
                       s=list(test_label_distribution[idx].values()), color='blue')
        sorted_label_dict = dict(sorted(test_label_distribution_new.items()))
        print(f" Client:{idx}, {sorted_label_dict}")
    plt.xticks(range(1, client_num + 1))
    plt.yticks(range(0, 7))
    # axs.set_title(f'Test Dataset label distribution of Cora')
    plt.savefig('/data/dzh2023/FGPL1/federatedscope/model_heterogeneity/result/dataDistribution/Test_Dataset_label_distribution_Cora.pdf')
    plt.close()


def divide_dataset_epoch(dataset, epochs, num_samples_per_epoch=5000):
    """
    这个用来指定每个client在不同的epoch中使用公共数据集的哪些样本，从而确保在每一个通信轮次中，不同client上传的logits是基于同样的样本算出来的
    适用方法：FedMD，FSFL （在他们的源码中，没有模拟client-server的通信过程）

    key是epoch编号: 0，2，..., collaborative_epoch-1
    values是一个list: 保存着对应的epoch要用到的公共数据集的样本编号。
    Note:让每个client确定每一轮用哪些公共数据集的样本直觉上看起来并非是一个高效的做法，但是为了加快实现速度，我暂时没有对这一部分做改进
    """
    dict_epoch, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(epochs):
        dict_epoch[i] = set(np.random.choice(all_idxs, num_samples_per_epoch, replace=False))
        # all_idxs = list(set(all_idxs) - dict_epoch[i])
    return dict_epoch


def get_public_dataset(dataset, labels_offset=0):
    # TODO:核对每个数据集Normalize的值是否正确
    data_dir = './data'
    if dataset == 'mnist':
        apply_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        data_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=apply_transform)
        data_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=apply_transform)
    elif dataset == 'cifar100':
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        data_train = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    if labels_offset != 0:
        # 增加训练集标签编号的偏移量
        for i in range(len(data_train)):
            image, label = data_train[i]
            data_train.targets[i] = label + labels_offset

        # 增加测试集标签编号的偏移量
        for i in range(len(data_test)):
            image, label = data_test[i]
            data_test.targets[i] = label + labels_offset

    return data_train, data_test


def get_classes_num(dataset):
    dataset_mapping = {
        'CIFAR10@torchvision': 10,
        'cifar100': 100,
        'SVHN@torchvision': 10,
        'office_caltech': 10,
        'mnist': 10,
    }
    if dataset not in dataset_mapping:
        logger.warning(f"未找到对应 数据集{dataset}，返回默认classes_num:10")
    return dataset_mapping.get(dataset, 10)


def train_CV(model, optimizer, criterion, train_loader, device, client_id, epoch):
    """
        本函数封装本地模型在public dataset或是private dataset上的训练过程(针对计算机视觉数据集)
        被用于FSFL
    """
    model.train()
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # TODO:FSFL的模型输出维度和标签对不上
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    logger.info(f'Train Epoch: {epoch} \t Train Loss: {train_loss}')
    return train_loss


@torch.no_grad()
def eval_CV(model, criterion, test_loader, device, client_id, epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            gt_labels.append(labels.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    val_loss = val_loss / len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)  # TODO:查看federateddscope里的计算精度的代码
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    logger.info(f'Eval Epoch: {epoch} \tTest Loss: {val_loss}, Test Acc: {acc}')
    return val_loss, acc


def test(model, test_loader, device):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    # print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
    #       .format(test_loss, acc))
    return acc, test_loss


class EarlyStopMonitor(object):
    def __init__(self, max_round=10, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        update_best_this_round = True
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
            update_best_this_round = False

        self.epoch_count += 1

        return self.num_round >= self.max_round, update_best_this_round

    def reset(self):
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None


class Ensemble(torch.nn.Module):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/synthesizers.py
    """

    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def pack_images(images, col=None, channel_last=False, padding=1):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """

    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


class ImagePool(object):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform)


class DeepInversionHook():
    '''
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):  # hook_fn(module, input, output) -> None
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()


def average_weights(w):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class KLDiv(nn.Module):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """

    def __init__(self, T=1.0, reduction='batchmean'):
        """

        :rtype: object
        """
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/helpers/utils.py
    """
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """
    source: https://github.com/NaiboWang/Data-Free-Ensemble-Selection-For-One-Shot-Federated-Learning/blob/master/DENSE/heter_fl.py
    """
    if is_best:
        torch.save(state, filename)


class TwoCropTransform:
    """
    Create two crops of the same image
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def param2tensor(param):
    if isinstance(param, list):
        param = torch.FloatTensor(param)
    elif isinstance(param, int):
        param = torch.tensor(param, dtype=torch.long)
    elif isinstance(param, float):
        param = torch.tensor(param, dtype=torch.float)
    elif isinstance(param, str):
        param = pickle.loads((base64.b64decode(param)))
    elif isinstance(param, np.ndarray):
        param = torch.from_numpy(param)
    return param


def sort_dict_by_key(dictionary):
    # TODO: 是否有python自带的函数？
    sorted_dict = dict()
    sorted_keys = sorted(dictionary.keys())
    for key in sorted_keys:
        sorted_dict[key] = dictionary[key]
    return sorted_dict


def delete_embeeding_files(path):
    """
    refer to: https://github.com/TsingZ0/HeFL/blob/main/system/clean_temp_files.py
    """
    # Forcefully delete the directory and its contents
    try:
        shutil.rmtree(path)
        print('Deleted.')
    except:
        print('Already deleted.')


def result_to_csv(result, init_cfg, best_round, runner, client_cfg_file):
    # 获取当前时间
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M")

    if init_cfg.federate.make_global_eval:
        test_acc = result['server_global_eval']['test_acc']
    else:
        test_acc = result['client_summarized_avg']['test_acc']

    if init_cfg.model.task == 'node':
        out_dict = {
            'exp_time': [time_string],
            'seed': [init_cfg.seed],
            'method': [init_cfg.federate.method],
            'optimizer': [init_cfg.train.optimizer.type],
            'lr': [init_cfg.train.optimizer.lr],
            'datasets': [init_cfg.data.type],
            'splitter': [init_cfg.data.splitter],
            'client_num': [init_cfg.federate.client_num],
            'local_updates': [init_cfg.train.local_update_steps],
            'dropout': [init_cfg.model.dropout],
            'client_cfg_file': [client_cfg_file],
            'best_round': [best_round],
            'test_acc': [test_acc],
            'whole_test_dataset': [init_cfg.data.local_eval_whole_test_dataset]
        }
    else:
        out_dict = {
            'exp_time': [time_string],
            'seed': [init_cfg.seed],
            'method': [init_cfg.federate.method],
            'batch_size': [init_cfg.dataloader.batch_size],
            'optimizer': [init_cfg.train.optimizer.type],
            'lr': [init_cfg.train.optimizer.lr],
            'datasets': [init_cfg.data.type],
            'splitter': [init_cfg.data.splitter],
            'client_num': [init_cfg.federate.client_num],
            'local_updates': [init_cfg.train.local_update_steps],
            'test_acc': [test_acc],
            'best_round': [best_round],
            'fml_alpha': [init_cfg.fml.alpha],
            'fml_beta': [init_cfg.fml.beta],
            'fedmd_public_subset_size': [init_cfg.fedmd.public_subset_size],
            'fedmd_digest_epochs': [init_cfg.fedmd.digest_epochs],
            'fedhenn_eta': [init_cfg.fedhenn.eta],
            'dropout': [init_cfg.model.dropout],
            'client_cfg_file': [client_cfg_file],
            'whole_test_dataset': [init_cfg.data.local_eval_whole_test_dataset]
        }

    if len(init_cfg.data.splitter_args) != 0 and 'alpha' in init_cfg.data.splitter_args[0]:
        out_dict['alpha'] = init_cfg.data.splitter_args[0]['alpha']

    method = out_dict['method'][0]

    if method == 'fedproto':
        out_dict['proto_weight'] = init_cfg.fedproto.proto_weight

        if 'test_acc_based_on_global_proto' in result['client_summarized_avg'].keys():
            out_dict['test_acc_based_on_global_proto'] = \
                result['client_summarized_avg']['test_acc_based_on_global_prototype']
    if 'poi' in method:
        out_dict['LP_layer']=init_cfg.poi.LP_layer
        out_dict['LP_alpha']=init_cfg.poi.LP_alpha
    if 'fgpl' in method:
        out_dict['delta'] = init_cfg.fgpl.delta
        out_dict['mu'] = init_cfg.fgpl.mu
        out_dict['lamda'] = init_cfg.fgpl.lamda
        out_dict['imb_ratio'] = init_cfg.fgpl.imb_ratio
        out_dict['beta'] = init_cfg.fgpl.beta
    if 'sefgl' in method:
        out_dict['delta'] = init_cfg.sefgl.delta
        out_dict['mu'] = init_cfg.sefgl.mu
        out_dict['lamda'] = init_cfg.sefgl.lamda
        out_dict['imb_ratio'] = init_cfg.sefgl.imb_ratio
        out_dict['beta'] = init_cfg.sefgl.beta
    if method == 'fml':
        out_dict['FML_alpha'] = init_cfg.fml.alpha
        out_dict['FML_beta'] = init_cfg.fml.beta
        out_dict['FML_meme_model'] = init_cfg.fml.meme_model.type

    if out_dict['method'][0] == 'fccl':
        out_dict['off_diag_weight'] = init_cfg.fccl.off_diag_weight
        out_dict['loss_dual_weight'] = init_cfg.fccl.loss_dual_weight
        out_dict['public_dataset_name'] = init_cfg.MHFL.public_dataset

    if out_dict['method'][0] == 'fedpcl':
        out_dict['test_acc_based_on_local_proto'] = result['client_summarized_avg']['test_acc_based_on_local_prototype']
        print(f"client summarized avg test_acc (based on local proto):{out_dict['test_acc_based_on_local_proto']}")
    out_dict['local_eval_whole_test_dataset'] = [init_cfg.data.local_eval_whole_test_dataset]

    if init_cfg.show_client_best_individual:
        individual_best_avg_result = show_per_client_best_individual(runner)
        out_dict['individual_best_bac_avg'] = individual_best_avg_result['test_bac']
        out_dict['individual_best_f1score_avg'] = individual_best_avg_result['test_f1score']
        out_dict['individual_best_test_acc_avg'] = individual_best_avg_result['test_acc']
        if out_dict['method'][0] == 'fedproto':
            gp_test_acc = individual_best_avg_result['test_acc_based_on_global_prototype']
            lp_test_acc = individual_best_avg_result['test_acc_based_on_local_prototype']
            out_dict['individual_avg_test_acc_with_global_prototype'] = gp_test_acc
            out_dict['individual_avg_test_acc_with_local_prototype'] = lp_test_acc
        if out_dict['method'][0] == 'fedpcl':
            out_dict['individual_avg_test_acc_with_global_prototype'] = individual_best_avg_result['test_acc_based_on_global_prototype']
            out_dict['individual_avg_test_acc_with_local_prototype'] = individual_best_avg_result['test_acc_based_on_local_prototype']
            out_dict['individual_avg_test_f1_with_global_prototype'] = individual_best_avg_result[
                'test_f1_based_on_global_prototype']
            out_dict['individual_avg_test_pre_with_local_prototype'] = individual_best_avg_result[
                'test_bac_based_on_global_prototype']

        if 'test_ensemble_model_acc' in individual_best_avg_result.keys():
            out_dict['test_ensemble_model_acc'] = individual_best_avg_result['test_ensemble_model_acc']

        if 'test_global_model_acc' in individual_best_avg_result.keys():
            out_dict['test_global_model_acc']=individual_best_avg_result['test_global_model_acc']

    df = pd.DataFrame(out_dict, columns=out_dict.keys())
    folder_path = init_cfg.result_floder
    csv_path = f'{folder_path}/{init_cfg.exp_name}.csv'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 如果已存在csv，则在csv末尾添加本次的实验记录
    if not os.path.exists(csv_path) or not os.path.getsize(csv_path):
        df.to_csv(csv_path, mode='a', index=False, header=True)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)
    logger.info(f'The detailed results of the experiment have been saved to: {csv_path} file')
    # print(df)

    return df


def plot_acc_curve(runner):
    cfg = runner.cfg
    client_num = cfg.federate.client_num

    fig, axs = plt.subplots(1, client_num, figsize=(12, 4), sharex=True, sharey=True)
    for client_id in range(1, client_num + 1):
        history_result = runner.client[client_id].history_results
        model_type = runner.client_cfgs[f'client_{client_id}'].model.type

        for key in history_result.keys():
            if ('test' in key or 'train' in key) and 'acc' in key:
                idx = client_id - 1
                x = np.arange(1, len(history_result[key]) + 1)
                y = np.array(history_result[key])
                axs[idx].plot(x, y, label=key)
                axs[idx].xlabel = 'epoch'
                axs[idx].ylabel = 'accuracy'
                axs[idx].set_title(f'Client {client_id}:{model_type}')
                axs[idx].legend()
    plt.show()

def show_per_client_best_individual(runner):
    client_num = runner.cfg.federate.client_num
    total = defaultdict(float)
    avg_results = defaultdict(float)
    for client_id in range(1, client_num + 1):
        best_results = list(runner.client[client_id].best_results.values())[0]
        history_result = runner.client[client_id].history_results
        print(f"\nClient {client_id}:")

        for key in history_result.keys():
            value = best_results.get(key, 0.0)
            total[key] += value
            if value != 0.0 and ('acc'in key or 'bac' in key or 'f1score' in key ):
                print(f"best_{key}: {value} ")

    if runner.cfg.plot_acc_curve:
        plot_acc_curve(runner)

    print('\nAverage:')
    for key, value in total.items():
        avg_results[key] = value / client_num
        if 'acc' in key and 'test' in key:
            print(f'best_avg_individual_{key}:\t {avg_results[key]}')

    return avg_results

    # return avg_results['test_acc'], avg_results['val_acc'], avg_results['test_acc_based_on_local_prototype'], \
    #        avg_results['test_acc_based_on_global_prototype']

        # for key in history_result.keys():
        #     value = best_results.get(key, 0.0)
        #     total[key] += value
        #
        #     try:
        #         corresponding_local_round = [index for index, val in enumerate(history_result[key]) if val == value]
        #     except:
        #         corresponding_local_round=None
        #
        #     if corresponding_local_round is not None and 'acc' in key and corresponding_local_round!=0.0:
        #         print(f"best_{key}: {value} \t corresponding_local_round:{corresponding_local_round}")

        # history_result_arr = np.array(history_result[key])
        # indices = np.where(history_result_arr==np.max(history_result_arr))
        # best_history_round = history_result[key].index(max(history_result[key]))
        # best_results = [index for index, value in enumerate(history_result) if value == max(history_result[key])]
        # if key=='test_acc' and corresponding_local_round not in indices:
        #     print(f"输出的test_acc的round:{corresponding_local_round},但是test_acc的最大zhilocal round:{indices}")
    # plt.legend()
    # plt.show()