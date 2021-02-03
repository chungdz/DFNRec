import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FM_Part(nn.Module):

    def __init__(self):
        super(FM_Part, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=False), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=False)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * cross_term

        return cross_term

class VanillaAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(VanillaAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size * 4, 32),
            nn.ReLU()
        )
        
        self.gate_layer = nn.Linear(32, 1)

    def forward(self, target, info):
        seqs = torch.cat([info, target, info * target, info - target], dim=-1)
        
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = target * p_attn
        output = torch.sum(h, dim=1)
        return output