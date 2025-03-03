import torch
import torch.nn as nn

class LocalModelWithFC(nn.Module):
    def __init__(self, local_model, return_features, feature_dim, out_dim):
        super(LocalModelWithFC, self).__init__()
        self.local_model = local_model
        self.FC = nn.Linear(feature_dim, out_dim)
        self.return_features = return_features

    def forward(self, x):
        if self.return_features:
            features = self.local_model(x)
            x = self.FC(features)
            return x, features
        else:
            x = self.local_model(x)
            x = self.FC(x)
            return x
