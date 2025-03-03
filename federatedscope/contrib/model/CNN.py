import torch.nn.functional as F
from federatedscope.register import register_model
from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
import torch
import torch.nn as nn

class CNN_2layers(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 filters=[128, 256],
                 use_bn=True,
                 dropout=.0,
                 return_proto=False):
        super(CNN_2layers, self).__init__()

        n1 = filters[0]
        n2 = filters[1]

        self.conv1 = Conv2d(in_channels, n1, 5, padding=2)
        self.conv2 = Conv2d(n1, n2, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(n1)
            self.bn2 = BatchNorm2d(n2)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * n2, hidden)
        self.FC = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout
        self.return_proto = return_proto

    def forward(self, x, GAN=False):
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        if GAN:
            return x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.relu(self.fc1(x))
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.FC(x)

        if self.return_proto:
            return x, x1
        else:
            return x


class CNN_3layers(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 filters=[64, 128, 256],
                 use_bn=True,
                 dropout=.0,
                 return_proto=False):
        super(CNN_3layers, self).__init__()

        n1 = filters[0]
        n2 = filters[1]
        n3 = filters[2]

        self.conv1 = Conv2d(in_channels, n1, 5, padding=2)
        self.conv2 = Conv2d(n1, n2, 5, padding=2)
        self.conv3 = Conv2d(n2, n3, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(n1)
            self.bn2 = BatchNorm2d(n2)
            self.bn3 = BatchNorm2d(n3)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * n3, hidden)
        self.FC = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout
        self.return_proto = return_proto

    def forward(self, x, GAN=False):
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = self.bn3(self.conv3(x)) if self.use_bn else self.conv3(x)
        x = self.relu(x)

        x = Flatten()(x)
        if GAN:
            return x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.relu(self.fc1(x))
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.FC(x)

        if self.return_proto:
            return x, x1
        else:
            return x


class CNN_4layers(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 filters=[64, 64, 64, 64],
                 use_bn=True,
                 dropout=.0,
                 return_proto=False):
        super(CNN_4layers, self).__init__()

        n1 = filters[0]
        n2 = filters[1]
        n3 = filters[2]
        n4 = filters[3]

        self.conv1 = Conv2d(in_channels, n1, 5, padding=2)
        self.conv2 = Conv2d(n1, n2, 5, padding=2)
        self.conv3 = Conv2d(n2, n3, 5, padding=2)
        self.conv4 = Conv2d(n3, n4, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(n1)
            self.bn2 = BatchNorm2d(n2)
            self.bn3 = BatchNorm2d(n3)
            self.bn4 = BatchNorm2d(n4)

        self.fc1 = Linear((h // 2 // 2 // 2) * (w // 2 // 2 // 2) * n4, hidden)
        self.FC = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout

        self.return_proto = return_proto

    def forward(self, x, GAN=False):
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = self.bn3(self.conv3(x)) if self.use_bn else self.conv3(x)
        x = self.maxpool(self.relu(x))
        x = self.bn4(self.conv4(x)) if self.use_bn else self.conv4(x)
        x = self.relu(x)

        x = Flatten()(x)
        if GAN:
            return x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.relu(self.fc1(x))
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.FC(x)

        if self.return_proto:
            return x, x1
        else:
            return x


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, hidden=1024, return_features=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, hidden),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(hidden, num_classes)

        self.return_features = return_features

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        x1 = self.fc1(out)
        out = self.fc(x1)
        if self.return_features:
            return out, x1
        else:
            return out


def call_our_cnn(model_config, input_shape):
    if 'CNN_2layers' in model_config.type:
        model = CNN_2layers(in_channels=input_shape[-3],
                            w=input_shape[-2],
                            h=input_shape[-1],
                            hidden=model_config.hidden,
                            class_num=model_config.out_channels,
                            filters=model_config.filter_channels,
                            use_bn=model_config.use_bn,
                            dropout=model_config.dropout,
                            return_proto=model_config.return_proto)
        return model
    elif 'CNN_3layers' in model_config.type:
        model = CNN_3layers(in_channels=input_shape[-3],
                            w=input_shape[-2],
                            h=input_shape[-1],
                            hidden=model_config.hidden,
                            class_num=model_config.out_channels,
                            filters=model_config.filter_channels,
                            use_bn=model_config.use_bn,
                            dropout=model_config.dropout,
                            return_proto=model_config.return_proto)
        return model

    elif 'CNN_4layers' in model_config.type:
        model = CNN_4layers(in_channels=input_shape[-3],
                            w=input_shape[-2],
                            h=input_shape[-1],
                            hidden=model_config.hidden,
                            class_num=model_config.out_channels,
                            filters=model_config.filter_channels,
                            use_bn=model_config.use_bn,
                            dropout=model_config.dropout,
                            return_proto=model_config.return_proto)
        return model
    elif 'FedAvgCNN' in model_config.type:
        model = FedAvgCNN(in_features=input_shape[-3], num_classes=model_config.out_channels,
                          hidden=model_config.hidden, return_features=model_config.return_proto)
        return model


register_model('call_our_cnn', call_our_cnn)
