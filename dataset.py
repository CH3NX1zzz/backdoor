import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
import pandas as pd


class TimeDataset(Dataset):
    def __init__(self, data, window_size=12, stride=1, timestamps=None):
        """
        改进的TimeDataset，使用滑动窗口来覆盖整个时间序列

        Args:
            data: 输入数据，形状为 [timestamps, nodes, features]
            window_size: 窗口大小
            stride: 滑动步长，默认为1表示每次移动一个时间步
            timestamps: 时间戳信息（如果有的话）
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.timestamps = timestamps

        # 计算可能的窗口数量
        self.num_windows = (len(data) - window_size - window_size) // stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        # 计算当前窗口的起始和结束位置
        start_idx = idx * self.stride
        input_end_idx = start_idx + self.window_size
        target_end_idx = input_end_idx + self.window_size

        # 获取输入窗口和目标窗口
        hist_data = self.data[start_idx:input_end_idx]
        futr_data = self.data[input_end_idx:target_end_idx]

        # 转换为张量
        hist_data = torch.FloatTensor(hist_data)
        futr_data = torch.FloatTensor(futr_data)

        if hist_data.shape[-1] == 1:
            hist_data = hist_data.squeeze(-1)
            futr_data = futr_data.squeeze(-1)

        # 返回当前时间步的索引
        return hist_data, futr_data, futr_data.clone(), torch.tensor(start_idx)

    def denormalize(self, normalized_data):
        """
        将标准化的数据转换回原始尺度
        这个方法在需要时可以根据具体的标准化方式进行修改

        Parameters:
            normalized_data (torch.Tensor): 标准化后的数据

        Returns:
            torch.Tensor: 转换回原始尺度的数据
        """
        return normalized_data