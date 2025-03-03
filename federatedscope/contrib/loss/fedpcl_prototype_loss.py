import torch
import torch.nn as nn
import torch.nn.functional as F
"""
FedPCL源码中的los实现似乎无法和论文公式对应；效果也不好
故自己尝试按照FedPCL的论文公式重写一个loss
"""
class ProtoConloss(nn.Module):
    """
    souce: https://github.com/yuetan031/FedPCL/blob/main/lib/losses.py
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ProtoConloss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, global_protos_dict, num_classes):
        """Compute contrastive loss between feature and global prototype
        """
        device = features.device
        global_protos = torch.stack(list(global_protos_dict.values())) # dict转tensor -> (class_num, hidden_size)
        onehot_mask = F.one_hot(labels, num_classes=num_classes).bool().to(device) # (batch_size, class_num)，用来标注每一个样本属于哪个类别
        '''
            if labels = [0, 1, 1], class_num =3,
            此时的mask：
               [1., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.]
        '''
        anchor_feature = features
        contrast_feature = global_protos

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # (N, class_num) ：anchor_dot_contrast(i,j)代表第i个节点的embedding和第j类的原型的相似度

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) #* (~onehot_mask) #L_g和L_p中分式的分母，~onehot_mask中每一个样本所属类对应的元素为0，其他为1
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (onehot_mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
