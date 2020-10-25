import torch
import torch.nn as nn
from torch.nn import functional as F


class PCNN(nn.Module):
    def __init__(self, emb_dim,  hidden_size=230):
        super(PCNN, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.cnn = nn.Conv1d(emb_dim, hidden_size, 3, padding=1)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X, X_mask):
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X = self.pool(X, X_mask)
        X = F.relu(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        hidden_size = X.shape[-1]
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, hidden_size * 3)
