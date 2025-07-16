import torch
import torch.nn as nn

class EnhancedCNN2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(EnhancedCNN2D, self).__init__()
        # 多尺度卷积：使用 3x3、5x5、7x7 卷积核，每条路径输出 out_channels // len(kernel_sizes) 个通道
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(kernel_sizes), kernel_size=(k, k), padding=(k//2, k//2)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels // len(kernel_sizes)),
                nn.Conv2d(out_channels // len(kernel_sizes), out_channels // len(kernel_sizes), kernel_size=(k, k), padding=(k//2, k//2)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels // len(kernel_sizes))
            )
            self.convs.append(conv)
        # 残差连接：若输入输出通道数不同，使用 1x1 卷积调整维度；否则直接使用输入
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residuals = x if self.residual is None else self.residual(x)
        conv_outs = []
        for conv in self.convs:
            out = conv(x)
            conv_outs.append(out)
        out = torch.cat(conv_outs, dim=1)  # 合并多尺度特征
        out = out + residuals  # 残差连接
        out = self.dropout(out)
        return out