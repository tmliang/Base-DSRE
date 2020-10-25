import torch
import torch.nn as nn
from torch.nn import functional as F
from .PCNN import PCNN

class Model(nn.Module):
    def __init__(self, pre_word_vec, rel_num, opt, hidden_size=230):
        super(Model, self).__init__()
        self.PCNN = PCNN(pre_word_vec, opt['max_pos_length'])
        self.rel = nn.Linear(hidden_size * 3, rel_num)
        self.drop = nn.Dropout(opt['dropout'])
        self.init_weight()

    def forward(self, X, X_Pos1, X_Pos2, X_Mask, X_Scope, X_Rel=None):
        X = self.PCNN(X, X_Pos1, X_Pos2, X_Mask)
        X = self.drop(X)
        X = self.sentence_attention(X, X_Scope, X_Rel)
        return X

    def init_weight(self):
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.zeros_(self.rel.bias)

    def sentence_attention(self, X, X_Scope, Rel=None):
        bag_output = []
        if Rel is not None:
            Rel = F.embedding(Rel, self.rel.weight)
            for i in range(X_Scope.shape[0]):
                bag_rep = X[X_Scope[i][0]: X_Scope[i][1]]
                att_score = F.softmax(bag_rep.matmul(Rel[i]), 0).view(1, -1)  # (1, Bag_size)
                att_output = att_score.matmul(bag_rep)  # (1, dim)
                bag_output.append(att_output.squeeze())  # (dim, )
            bag_output = torch.stack(bag_output)
            bag_output = self.drop(bag_output)
            bag_output = self.rel(bag_output)
        else:
            att_score = X.matmul(self.rel.weight.t())  # (Batch_size, dim) -> (Batch_size, R)
            for s in X_Scope:
                bag_rep = X[s[0]:s[1]]  # (Bag_size, dim)
                bag_score = F.softmax(att_score[s[0]:s[1]], 0).t()  # (R, Bag_size)
                att_output = bag_score.matmul(bag_rep)  # (R, dim)
                bag_output.append(torch.diagonal(F.softmax(self.rel(att_output), -1)))
            bag_output = torch.stack(bag_output)
            # bag_output = F.softmax(bag_output, -1)
        return bag_output
