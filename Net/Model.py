import torch
import torch.nn as nn
from torch.nn import functional as F
from Net import CNN, PCNN, BiGRU
import numpy as np

class Model(nn.Module):
    def __init__(self, pre_word_vec, rel_num, opt, pos_dim=5, hidden_size=230):
        super(Model, self).__init__()
        word_embedding = torch.from_numpy(np.load(pre_word_vec))
        pos_len = opt['max_pos_length']
        emb_dim = word_embedding.shape[1] + 2 * pos_dim
        self.encoder_name = opt['encoder']
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=-1)
        self.pos1_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.drop = nn.Dropout(opt['dropout'])

        if self.encoder_name == 'CNN':
            self.encoder = CNN(emb_dim)
            self.rel = nn.Linear(hidden_size, rel_num)

        elif self.encoder_name == 'BiGRU':
            self.encoder = BiGRU(emb_dim)
            self.rel = nn.Linear(hidden_size * 2, rel_num)

        else:
            self.encoder = PCNN(emb_dim)
            self.rel = nn.Linear(hidden_size * 3, rel_num)

        self.init_weight()

    def forward(self, X, X_Pos1, X_Pos2, X_Mask, X_Len, X_Scope, X_Rel=None):
        X = self.word_pos_embedding(X, X_Pos1, X_Pos2)
        if self.encoder_name == 'CNN':
            X = self.encoder(X)
        elif self.encoder_name == 'BiGRU':
            X = self.encoder(X, X_Len)
        else:
            X = self.encoder(X, X_Mask)
        X = self.drop(X)
        X = self.sentence_attention(X, X_Scope, X_Rel)
        return X

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.zeros_(self.rel.bias)

    def word_pos_embedding(self, X, X_Pos1, X_Pos2):
        X = self.word_embedding(X)
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        X = torch.cat([X, X_Pos1, X_Pos2], -1)
        return X

    def sentence_attention(self, X, X_Scope, Rel=None):
        bag_output = []
        if Rel is not None:  # For training
            Rel = F.embedding(Rel, self.rel.weight)
            for i in range(X_Scope.shape[0]):
                bag_rep = X[X_Scope[i][0]: X_Scope[i][1]]
                att_score = F.softmax(bag_rep.matmul(Rel[i]), 0).view(1, -1)  # (1, Bag_size)
                att_output = att_score.matmul(bag_rep)  # (1, dim)
                bag_output.append(att_output.squeeze())  # (dim, )
            bag_output = torch.stack(bag_output)
            bag_output = self.drop(bag_output)
            bag_output = self.rel(bag_output)
        else:  # For testing
            att_score = X.matmul(self.rel.weight.t())  # (Batch_size, dim) -> (Batch_size, R)
            for s in X_Scope:
                bag_rep = X[s[0]:s[1]]  # (Bag_size, dim)
                bag_score = F.softmax(att_score[s[0]:s[1]], 0).t()  # (R, Bag_size)
                att_output = bag_score.matmul(bag_rep)  # (R, dim)
                bag_output.append(torch.diagonal(F.softmax(self.rel(att_output), -1)))
            bag_output = torch.stack(bag_output)
            # bag_output = F.softmax(bag_output, -1)
        return bag_output
