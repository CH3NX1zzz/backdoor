import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
import pandas as pd
from typing import Tuple, Optional, Union
import random

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


class MissingDataGenerator:
    """
    时间序列数据缺失值生成器
    支持多种缺失机制：完全随机缺失(MCAR)、随机缺失(MAR)、非随机缺失(MNAR)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化缺失值生成器

        Args:
            seed: 随机种子，用于结果复现
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def generate_mcar(
            self,
            data: torch.Tensor,
            missing_rate: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成完全随机缺失(MCAR)的数据

        Args:
            data: 输入数据张量，形状为 [timestamps, features]
            missing_rate: 缺失比例，范围[0, 1]

        Returns:
            missing_data: 包含缺失值的数据
            missing_mask: 缺失值掩码，1表示缺失
        """
        # 创建缺失值掩码
        missing_mask = torch.zeros_like(data, dtype=torch.bool)

        # 计算需要生成的缺失值数量
        n_missing = int(data.numel() * missing_rate)

        # 随机选择缺失位置
        missing_indices = torch.randperm(data.numel())[:n_missing]
        missing_mask.view(-1)[missing_indices] = True

        # 生成缺失数据
        missing_data = data.clone()
        missing_data[missing_mask] = float('nan')

        return missing_data, missing_mask

    def generate_mar(
            self,
            data: torch.Tensor,
            missing_rate: float = 0.2,
            ref_features: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成随机缺失(MAR)的数据
        缺失概率依赖于其他完整特征的值

        Args:
            data: 输入数据张量
            missing_rate: 总体缺失比例
            ref_features: 用于确定缺失概率的参考特征索引列表

        Returns:
            missing_data: 包含缺失值的数据
            missing_mask: 缺失值掩码
        """
        if ref_features is None:
            ref_features = list(range(data.shape[1]))

        # 基于参考特征计算缺失概率
        ref_values = data[:, ref_features].mean(dim=1)
        probs = torch.sigmoid(ref_values)
        probs = probs / probs.sum() * missing_rate * data.numel()

        # 生成缺失掩码
        missing_mask = torch.zeros_like(data, dtype=torch.bool)
        for i in range(len(probs)):
            if probs[i] > torch.rand(1):
                missing_mask[i] = True

        # 生成缺失数据
        missing_data = data.clone()
        missing_data[missing_mask] = float('nan')

        return missing_data, missing_mask

    def generate_mnar(
            self,
            data: torch.Tensor,
            missing_rate: float = 0.2,
            threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成非随机缺失(MNAR)的数据
        缺失概率依赖于值本身的大小

        Args:
            data: 输入数据张量
            missing_rate: 缺失比例
            threshold: 触发缺失的阈值，默认为数据的中位数

        Returns:
            missing_data: 包含缺失值的数据
            missing_mask: 缺失值掩码
        """
        if threshold is None:
            threshold = torch.median(data)

        # 基于值的大小确定缺失概率
        probs = torch.where(
            data > threshold,
            torch.ones_like(data) * missing_rate * 1.5,
            torch.ones_like(data) * missing_rate * 0.5
        )

        # 生成缺失掩码
        missing_mask = torch.zeros_like(data, dtype=torch.bool)
        missing_mask = torch.rand_like(data) < probs

        # 生成缺失数据
        missing_data = data.clone()
        missing_data[missing_mask] = float('nan')

        return missing_data, missing_mask

    def generate_missing_blocks(
            self,
            data: torch.Tensor,
            block_size: int = 3,
            num_blocks: Optional[int] = None,
            missing_rate: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成连续块状缺失的数据

        Args:
            data: 输入数据张量
            block_size: 每个缺失块的大小
            num_blocks: 缺失块的数量，如果为None则根据缺失率计算
            missing_rate: 总体缺失比例

        Returns:
            missing_data: 包含缺失值的数据
            missing_mask: 缺失值掩码
        """
        if num_blocks is None:
            num_blocks = int(data.shape[0] * missing_rate / block_size)

        # 生成可能的块起始位置
        valid_starts = torch.arange(0, data.shape[0] - block_size + 1)

        # 随机选择块的起始位置
        block_starts = torch.randperm(len(valid_starts))[:num_blocks]

        # 生成缺失掩码
        missing_mask = torch.zeros_like(data, dtype=torch.bool)
        for start in block_starts:
            missing_mask[start:start + block_size] = True

        # 生成缺失数据
        missing_data = data.clone()
        missing_data[missing_mask] = float('nan')

        return missing_data, missing_mask

    def generate_mixed_missing(
            self,
            data: torch.Tensor,
            missing_rates: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成混合类型的缺失数据

        Args:
            data: 输入数据张量
            missing_rates: 不同缺失类型的比例字典，例如
                {'mcar': 0.1, 'mar': 0.1, 'blocks': 0.1}

        Returns:
            missing_data: 包含缺失值的数据
            missing_mask: 缺失值掩码
        """
        missing_data = data.clone()
        missing_mask = torch.zeros_like(data, dtype=torch.bool)

        # 依次应用不同的缺失机制
        if 'mcar' in missing_rates:
            mcar_data, mcar_mask = self.generate_mcar(
                data,
                missing_rates['mcar']
            )
            missing_mask = missing_mask | mcar_mask

        if 'mar' in missing_rates:
            # 在未缺失的位置上应用MAR
            valid_data = data.clone()
            valid_data[missing_mask] = float('nan')
            mar_data, mar_mask = self.generate_mar(
                valid_data,
                missing_rates['mar']
            )
            missing_mask = missing_mask | mar_mask

        if 'blocks' in missing_rates:
            # 计算剩余数据中需要的块数
            remaining_rate = missing_rates['blocks']
            block_data, block_mask = self.generate_missing_blocks(
                data,
                missing_rate=remaining_rate
            )
            missing_mask = missing_mask | block_mask

        # 生成最终的缺失数据
        missing_data[missing_mask] = float('nan')

        return missing_data, missing_mask

    @staticmethod
    def get_missing_statistics(
            missing_mask: torch.Tensor
    ) -> dict:
        """
        计算缺失值的统计信息

        Args:
            missing_mask: 缺失值掩码

        Returns:
            统计信息字典
        """
        total_missing = missing_mask.sum().item()
        total_elements = missing_mask.numel()
        missing_rate = total_missing / total_elements

        # 计算连续缺失的统计信息
        consecutive_missing = []
        current_streak = 0

        for i in range(missing_mask.shape[0]):
            if missing_mask[i].any():
                current_streak += 1
            elif current_streak > 0:
                consecutive_missing.append(current_streak)
                current_streak = 0

        if current_streak > 0:
            consecutive_missing.append(current_streak)

        return {
            'total_missing': total_missing,
            'missing_rate': missing_rate,
            'max_consecutive': max(consecutive_missing) if consecutive_missing else 0,
            'avg_consecutive': np.mean(consecutive_missing) if consecutive_missing else 0
        }

