import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class PCNN(nn.Module):
    def __init__(self, pre_word_vec, pos_len, pos_dim=5, hidden_size=230):
        super(PCNN, self).__init__()
        word_embedding = torch.from_numpy(np.load(pre_word_vec))
        word_dim = word_embedding.shape[-1]
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=-1)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.pos1_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.cnn = nn.Conv1d(word_dim + 2 * pos_dim, hidden_size, 3, padding=1)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X, X_Pos1, X_Pos2, X_mask):
        X = self.word_pos_embedding(X, X_Pos1, X_Pos2)
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X = self.pool(X, X_mask)
        X = F.relu(X)
        return X

    def word_pos_embedding(self, X, X_Pos1, X_Pos2):
        X = self.word_embedding(X)
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        X = torch.cat([X, X_Pos1, X_Pos2], -1)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        hidden_size = X.shape[-1]
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, hidden_size * 3)
