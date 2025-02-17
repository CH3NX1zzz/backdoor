import torch
import torch.nn as nn
from TimesTrogan import TimesTroganFGSM, visualize_importance
from dataset import TimeDataset
from torch.utils.data import DataLoader


class ImportanceAnalyzer:
    def __init__(self, model, device, epsilon=0.01, seq_len=12):
        """
        初始化重要性分析器

        Args:
            model: 训练好的模型
            device: 计算设备
            epsilon: 扰动大小
            seq_len: 序列长度
        """
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.seq_len = seq_len
        self.timestrogan = TimesTroganFGSM(
            model=model,
            epsilon=epsilon,
            seq_len=seq_len
        )

    def prepare_data(self, data, window_size=12):
        """
        准备用于分析的数据加载器

        Args:
            data: 输入数据
            window_size: 窗口大小
        Returns:
            DataLoader: 数据加载器
        """
        dataset = TimeDataset(
            data=data,
            window_size=window_size,
            stride=1,
            timestamps=None
        )

        return DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )

    def analyze(self, data, save_path=None, top_k=5):
        """
        分析时间序列的重要性

        Args:
            data: 输入数据
            save_path: 结果保存路径
            top_k: 返回前k个最重要的时间步

        Returns:
            tuple: (重要性分数, 最重要的k个时间步)
        """
        # 准备数据
        dataloader = self.prepare_data(data)

        # 计算重要性分数
        importance_scores, _ = self.timestrogan.analyze_timeseries(dataloader, self.device)

        # 可视化结果
        if save_path:
            visualize_importance(importance_scores, save_path)

        # 获取最重要的时间步
        top_indices = torch.topk(importance_scores, top_k).indices

        return importance_scores, top_indices.tolist()


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
    analyzer = ImportanceAnalyzer(model, device, **kwargs)
    return analyzer.analyze(data, save_path)