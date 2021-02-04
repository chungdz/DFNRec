import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules import FM_Part, VanillaAttend

class DEERS(nn.Module):

    def __init__(self, cfg):
        super(DEERS, self).__init__()
        self.cfg = cfg
        
        self.word_emb = nn.Embedding(cfg.word_num, cfg.hidden_size)
        self.pos_gru = nn.GRU(cfg.hidden_size, cfg.hidden_size)
        self.neg_gru = nn.GRU(cfg.hidden_size, cfg.hidden_size)

        self.pos_linear = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU()
        )
        self.neg_linear = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU()
        )
        self.concat_linear = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size)

    def forward(self, inputs):
        uid = inputs[:, 0]
        news_id = inputs[:, 1]

        target_seq = inputs[:, 2: 2 + self.cfg.max_title_len]
        clicked_seq = inputs[:, 2 + self.cfg.max_title_len: (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length * self.cfg.max_title_len)]
        neg_seq = inputs[:, (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length * self.cfg.max_title_len): (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length + self.cfg.neg_hist_length) * self.cfg.max_title_len]
        weak_neg_seq = inputs[:, (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length + self.cfg.neg_hist_length) * self.cfg.max_title_len:]

        target_seq = self.word_emb(target_seq)
        clicked_seq = self.word_emb(clicked_seq.reshape(-1, self.cfg.pos_hist_length, self.cfg.max_title_len))
        neg_seq = self.word_emb(neg_seq.reshape(-1, self.cfg.neg_hist_length, self.cfg.max_title_len))

        target = target_seq.mean(dim=-2)
        clicked_seq = clicked_seq.mean(dim=-2)
        neg_seq = neg_seq.mean(dim=-2)

        clicked_seq = clicked_seq.permute(1, 0, 2)
        poutput, phn = self.pos_gru(clicked_seq)
        pos = phn.permute(1, 0, 2).squeeze(1)

        neg_seq = neg_seq.permute(1, 0, 2)
        noutput, nhn = self.neg_gru(neg_seq)
        neg = nhn.permute(1, 0, 2).squeeze(1)

        pos = self.pos_linear(pos)
        neg = self.neg_linear(neg)
        
        user = self.concat_linear(torch.cat([pos, neg], dim=-1))
        res = torch.sum(user * target, dim=-1)

        return torch.sigmoid(res)