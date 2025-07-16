import torch
import torch.nn as nn
import math

class LearnableTemporalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnableTemporalEncoding, self).__init__()
        # 可学习的位置编码，随机初始化
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TemporalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TemporalEncoding, self).__init__()
        # 固定位置编码，基于正弦和余弦函数
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x