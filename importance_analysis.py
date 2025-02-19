import torch
import torch.nn as nn
from TimesTrogan import TimesTroganFGSM, visualize_importance
from dataset import TimeDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class ImportanceAnalyzer:
    def __init__(self, model, device, epsilon=0.01, seq_len=12):
        """
        初始化重要性分析器

        Args:
            model: 训练好的模型
            device: 计算设备（CPU/GPU）
            epsilon: 扰动大小
            seq_len: 序列长度
        """
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.seq_len = seq_len
        self.model.eval()  # 设置为评估模式

    def analyze_importance(self, data: torch.Tensor) -> torch.Tensor:
        """
        分析时间序列中每个时间步的重要性

        Args:
            data: 输入数据 [timesteps, features]

        Returns:
            importance_scores: 每个时间步的重要性分数
        """
        dataset = TimeDataset(
            data=data.numpy(),
            window_size=self.seq_len,
            stride=1
        )
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0  # 避免多进程引起的问题
        )

        importance_scores = torch.zeros(data.shape[0])
        sample_count = torch.zeros(data.shape[0])

        with torch.no_grad():
            for batch_x, batch_y, _, batch_idx in dataloader:
                # 将数据移动到正确的设备
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 生成时间特征标记（如果模型需要）
                x_mark = torch.zeros(batch_x.shape[0], batch_x.shape[1], 4).to(self.device)
                dec_inp = torch.zeros(batch_x.shape[0], self.seq_len, batch_x.shape[2]).to(self.device)

                # 计算原始输出
                original_output = self.model(batch_x, x_mark, dec_inp, None)

                # 对每个时间步计算重要性
                for t in range(batch_x.shape[1]):
                    # 创建扰动版本
                    perturbed_x = batch_x.clone()
                    perturbed_x[:, t, :] += self.epsilon * torch.randn_like(perturbed_x[:, t, :])

                    # 计算扰动后的输出
                    perturbed_output = self.model(perturbed_x, x_mark, dec_inp, None)

                    # 计算输出变化
                    output_diff = torch.norm(perturbed_output - original_output, dim=2).mean(1)

                    # 更新重要性分数
                    for idx, diff in zip(batch_idx, output_diff):
                        actual_idx = idx.item() + t
                        if actual_idx < data.shape[0]:
                            importance_scores[actual_idx] += diff.item()
                            sample_count[actual_idx] += 1

        # 计算平均重要性分数
        valid_mask = sample_count > 0
        importance_scores[valid_mask] /= sample_count[valid_mask]

        # 归一化分数
        if torch.sum(importance_scores) > 0:
            importance_scores = importance_scores / torch.sum(importance_scores)

        return importance_scores

    def visualize_importance(self, importance_scores: torch.Tensor, save_path: str = None):
        """
        可视化重要性分数

        Args:
            importance_scores: 重要性分数
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))
        plt.plot(importance_scores.cpu().numpy(), marker='o', markersize=4)
        plt.title('时间步重要性分数')
        plt.xlabel('时间步')
        plt.ylabel('重要性分数')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def run_importance_analysis(model, data, device, save_path=None, **kwargs):
    """
    运行重要性分析的便捷函数

    Args:
        model: 训练好的模型
        data: 输入数据
        device: 计算设备
        save_path: 结果保存路径
        **kwargs: 其他参数

    Returns:
        tuple: (重要性分数, 最重要的时间步)
    """
    # 确保数据是torch.Tensor类型
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)

    # 初始化分析器
    analyzer = ImportanceAnalyzer(model, device, **kwargs)

    # 计算重要性分数
    importance_scores = analyzer.analyze_importance(data)

    # 可视化结果
    if save_path:
        analyzer.visualize_importance(importance_scores, save_path)

    # 获取最重要的5个时间步
    top_k = min(5, len(importance_scores))
    top_indices = torch.topk(importance_scores, top_k).indices.tolist()

    return importance_scores, top_indices