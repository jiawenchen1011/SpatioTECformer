import torch
import torch.nn as nn

class AdaptiveFusion(nn.Module):
    def __init__(self, d_model):
        super(AdaptiveFusion, self).__init__()
        # 自适应特征融合：通过门控机制动态调整 CNN 特征和外部特征的权重
        self.linear_aux = nn.Linear(d_model, d_model)  # 映射外部特征到 d_model 维度
        self.fusion_linear1 = nn.Linear(2 * d_model, d_model)
        self.gate_linear = nn.Linear(d_model, 1)
        self.fusion_linear2 = nn.Linear(2 * d_model, d_model)

    def forward(self, x_tec, x_phys):
        # x_tec: [batch_size, seq_len, d_model] (CNN 输出)
        # x_phys: [batch_size, seq_len, d_model] (外部特征)
        x_phys = self.linear_aux(x_phys)  # 映射外部特征到 d_model 维度
        combined = torch.cat([x_tec, x_phys], dim=-1)  # [batch_size, seq_len, 2 * d_model]
        gate = torch.sigmoid(self.gate_linear(torch.tanh(self.fusion_linear1(combined))))  # [batch_size, seq_len, 1]
        gate = gate.expand_as(x_phys)  # [batch_size, seq_len, d_model]
        fused = x_tec + gate * x_phys  # 门控仅作用于 x_phys，x_tec 直接加
        return self.fusion_linear2(torch.cat([x_tec, fused], dim=-1))  # [batch_size, seq_len, d_model]