import torch
import torch.nn as nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, emb_dim,  hidden_size=230):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(emb_dim, hidden_size, 3, padding=1)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X):
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X, _ = torch.max(X, 1)
        X = F.relu(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        hidden_size = X.shape[-1]
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, hidden_size * 3)
