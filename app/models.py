# Load all modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt


class SkipgramNegSampling(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.v_embed = nn.Embedding(vocab_size, emb_size)
        self.u_embed = nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, pos_words, neg_words):
        batch_size = center_words.size(0)
        v = self.v_embed(center_words)
        u_pos = self.u_embed(pos_words)
        u_neg = self.u_embed(neg_words)

        pos_score = torch.sum(v * u_pos, dim=1)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)

        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(1)
        loss = -(pos_loss + neg_loss).mean()
        return loss

class GloVeModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.v_embed = nn.Embedding(vocab_size, emb_size)
        self.u_embed = nn.Embedding(vocab_size, emb_size)
        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, coocs, weighting):
        v = self.v_embed(center_words)
        u = self.u_embed(target_words)
        v_bias = self.v_bias(center_words).squeeze(1)
        u_bias = self.u_bias(target_words).squeeze(1)

        inner_prod = torch.sum(v * u, dim=1)
        loss = weighting * (inner_prod + v_bias + u_bias - coocs) ** 2
        return loss.sum()
