import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np
import math

class Model(nn.Module):
    """
    TimesNet: 一个用于时间序列预测的深度学习模型
    该模型使用时间维度的注意力机制和多层感知机来处理时序数据
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 保存配置参数
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.top_k = configs.top_k
        self.num_kernels = configs.num_kernels

        # 初始化模型组件
        # 输入投影层：将输入特征维度映射到模型维度
        self.enc_embedding = nn.Linear(self.enc_in, self.d_model)

        # 位置编码
        self.position_encoding = PositionalEncoding(self.d_model)

        # 时间块层
        self.time_blocks = nn.ModuleList([
            TimeBlock(
                d_model=self.d_model,
                d_ff=self.d_ff,
                num_kernels=self.num_kernels,
                top_k=self.top_k
            ) for _ in range(self.e_layers)
        ])

        # 输出投影层：将模型维度映射回输出特征维度
        self.projection = nn.Linear(self.d_model, self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        模型的前向传播

        Args:
            x_enc: 编码器输入序列 [batch_size, seq_len, num_features]
            x_mark_enc: 编码器的时间特征标记
            x_dec: 解码器输入序列
            x_mark_dec: 解码器的时间特征标记
        """
        # 确保输入维度正确
        B, L, M = x_enc.shape

        # 特征投影
        enc_out = self.enc_embedding(x_enc)  # [B, L, d_model]

        # 添加位置编码 - 直接传递投影后的张量
        enc_out = self.position_encoding(enc_out)
        # 通过时间块处理
        for block in self.time_blocks:
            enc_out = block(enc_out)

        # 预测未来值
        dec_out = self.projection(enc_out)  # [B, L, M]

        # 只返回预测长度的输出
        dec_out = dec_out[:, -self.pred_len:, :]

        return dec_out


class TimeBlock(nn.Module):
    """
    时间块：TimesNet的核心组件
    包含多头注意力机制和前馈神经网络
    """

    def __init__(self, d_model, d_ff, num_kernels, top_k):
        super(TimeBlock, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.top_k = top_k

        # 多头注意力层
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # inception时间卷积
        self.inception = InceptionBlock(d_model, num_kernels)

    def forward(self, x):
        # 自注意力机制
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Inception时间卷积
        x = self.inception(x)

        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class InceptionBlock(nn.Module):
    """
    Inception块：使用不同核大小的卷积来捕获多尺度特征
    """

    def __init__(self, d_model, num_kernels):
        super(InceptionBlock, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=(k - 1) // 2)
            for k in range(1, num_kernels + 1, 2)
        ])

    def forward(self, x):
        # 转换维度顺序以适应卷积操作
        x = x.transpose(1, 2)

        # 应用不同核大小的卷积
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))

        # 合并所有卷积结果
        x = torch.stack(outputs).mean(0)

        # 转换回原始维度顺序
        x = x.transpose(1, 2)

        return x


class PositionalEncoding(nn.Module):
    """
    位置编码类：为序列中的每个位置添加唯一的位置信息
    这种编码帮助模型理解序列中元素的相对位置关系
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度并注册为缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        添加位置编码到输入张量

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]

        Returns:
            带有位置编码的张量，形状与输入相同
        """
        # 从缓冲区中获取对应长度的位置编码
        return x + self.pe[:, :x.size(1)]