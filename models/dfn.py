import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules import FM_Part, VanillaAttend

class DFN(nn.Module):

    def __init__(self, cfg):
        super(DFN, self).__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(cfg.user_num, cfg.hidden_size)
        self.news_emb = nn.Embedding(cfg.news_num, cfg.hidden_size)
        self.user_wide = nn.Embedding(cfg.user_num, cfg.hidden_size)
        self.news_wide = nn.Embedding(cfg.news_num, cfg.hidden_size)
        self.word_emb = nn.Embedding(cfg.word_num, cfg.hidden_size)

        self.click_linear = nn.Linear(self.cfg.max_title_len * self.cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.neg_linear = nn.Linear(self.cfg.max_title_len * self.cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.weak_neg_linear = nn.Linear(self.cfg.max_title_len * self.cfg.hidden_size * 2, cfg.hidden_size, bias=False)

        self.click_mt = nn.MultiheadAttention(cfg.hidden_size, cfg.head_num)
        self.click_mt_linear_1 = nn.Linear(cfg.hidden_size, cfg.hidden_size * 4)
        self.click_mt_linear_2 = nn.Linear(cfg.hidden_size * 4, cfg.hidden_size)
        self.neg_mt = nn.MultiheadAttention(cfg.hidden_size, cfg.head_num)
        self.neg_mt_linear_1 = nn.Linear(cfg.hidden_size, cfg.hidden_size * 4)
        self.neg_mt_linear_2 = nn.Linear(cfg.hidden_size * 4, cfg.hidden_size)
        self.weak_neg_mt = nn.MultiheadAttention(cfg.hidden_size, cfg.head_num)
        self.weak_neg_linear_1 = nn.Linear(cfg.hidden_size, cfg.hidden_size * 4)
        self.weak_neg_linear_2 = nn.Linear(cfg.hidden_size * 4, cfg.hidden_size)

        self.click_va = VanillaAttend(cfg.hidden_size)
        self.neg_va = VanillaAttend(cfg.hidden_size)

        self.fm = FM_Part()
        self.deep = nn.Sequential(
            nn.Linear(self.cfg.hidden_size * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.fc = nn.Linear(self.cfg.hidden_size + 16 + 2 * cfg.hidden_size, 1)

    def forward(self, inputs):
        uid = inputs[:, 0]
        news_id = inputs[:, 1]

        target_seq = inputs[:, 2: 2 + self.cfg.max_title_len]
        clicked_seq = inputs[:, 2 + self.cfg.max_title_len: (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length * self.cfg.max_title_len)]
        neg_seq = inputs[:, (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length * self.cfg.max_title_len): (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length + self.cfg.neg_hist_length) * self.cfg.max_title_len]
        weak_neg_seq = inputs[:, (2 + self.cfg.max_title_len) + (self.cfg.pos_hist_length + self.cfg.neg_hist_length) * self.cfg.max_title_len:]

        uid_deep = self.user_emb(uid)
        news_deep = self.news_emb(news_id)
        # wide part
        uid_wide = self.user_wide(uid)
        news_wide = self.news_wide(news_id)

        target_seq = self.word_emb(target_seq)
        clicked_seq = self.word_emb(clicked_seq.reshape(-1, self.cfg.pos_hist_length, self.cfg.max_title_len))
        neg_seq = self.word_emb(neg_seq.reshape(-1, self.cfg.neg_hist_length, self.cfg.max_title_len))
        weak_neg_seq = self.word_emb(weak_neg_seq.reshape(-1, self.cfg.unclicked_hist_length, self.cfg.max_title_len))

        target_seq = target_seq.reshape(-1, self.cfg.max_title_len * self.cfg.hidden_size)
        clicked_seq = clicked_seq.reshape(-1, self.cfg.pos_hist_length, self.cfg.max_title_len * self.cfg.hidden_size)
        neg_seq = neg_seq.reshape(-1, self.cfg.neg_hist_length, self.cfg.max_title_len * self.cfg.hidden_size)
        weak_neg_seq = weak_neg_seq.reshape(-1, self.cfg.unclicked_hist_length, self.cfg.max_title_len * self.cfg.hidden_size)

        target_seq = target_seq.repeat(1, self.cfg.pos_hist_length).view(-1, self.cfg.pos_hist_length, self.cfg.max_title_len * self.cfg.hidden_size)
        clicked_seq = torch.cat([clicked_seq, target_seq], dim=-1)
        clicked_seq = self.click_linear(clicked_seq)
        neg_seq = torch.cat([neg_seq, target_seq], dim=-1)
        neg_seq = self.neg_linear(neg_seq)
        weak_neg_seq = torch.cat([weak_neg_seq, target_seq], dim=-1)
        weak_neg_seq = self.weak_neg_linear(weak_neg_seq)

        out_clicked_seq = self.transformer(clicked_seq, self.click_mt, self.click_mt_linear_1, self.click_mt_linear_2)
        out_neg_seq = self.transformer(neg_seq, self.neg_mt, self.neg_mt_linear_1, self.neg_mt_linear_2)
        out_weak_neg_seq = self.transformer(weak_neg_seq, self.weak_neg_mt, self.weak_neg_linear_1, self.weak_neg_linear_2)
        out_enhanced_clicked = self.click_va(weak_neg_seq, out_clicked_seq.repeat(1, self.cfg.unclicked_hist_length).reshape(-1, self.cfg.unclicked_hist_length, self.cfg.hidden_size))
        out_enhanced_neg = self.neg_va(weak_neg_seq, out_neg_seq.repeat(1, self.cfg.unclicked_hist_length).reshape(-1, self.cfg.unclicked_hist_length, self.cfg.hidden_size))

        input_embedding = torch.cat([uid_deep, news_deep, out_clicked_seq, out_weak_neg_seq, out_neg_seq, out_enhanced_clicked, out_enhanced_neg], dim=-1)

        # fm part
        fm = self.fm(input_embedding.reshape(-1, 7, self.cfg.hidden_size))
        # deep part
        deep = self.deep(input_embedding)

        z = torch.cat([deep, fm, uid_wide, news_wide], dim=-1)
        res = self.fc(z)

        return torch.sigmoid(res)

    def transformer(self, seq, multi_head_self_attention, linear_1, linear_2):

        hiddens = seq.permute(1, 0, 2)
        user_hiddens, _ = multi_head_self_attention(hiddens, hiddens, hiddens)
        user_hiddens = user_hiddens.permute(1, 0, 2)

        user_hiddens = linear_1(user_hiddens)
        user_hiddens = F.relu(user_hiddens)
        user_hiddens = linear_2(user_hiddens)
        user_hiddens = user_hiddens.sum(dim=1, keepdim=False)

        return user_hiddens