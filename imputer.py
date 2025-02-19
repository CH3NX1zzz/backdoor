import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from dataset import TimeDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from importance_analysis import ImportanceAnalyzer

class TemporalImputer:
    """
    时间序列缺失值填补与后门攻击实现
    结合时间步重要性的缺失值填补优化器
    """

    def __init__(
            self,
            model: nn.Module,
            window_size: int = 12,
            alpha_smooth: float = 0.1,  # 平滑性损失权重
            alpha_temporal: float = 0.1,  # 时间一致性损失权重
            learning_rate: float = 0.01,
            max_iterations: int = 3,
            convergence_threshold: float = 1e-6
    ):
        self.model = model
        self.window_size = window_size
        self.alpha_smooth = alpha_smooth
        self.alpha_temporal = alpha_temporal
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def calculate_temporal_weights(self, missing_indices: torch.Tensor, importance_scores: torch.Tensor,
                                   top_k: int = 5) -> torch.Tensor:
        """
        计算重要时间步之间的距离权重
        只针对最重要的k个时间步计算权重矩阵

        Args:
            missing_indices: 缺失值的索引张量
            importance_scores: 每个时间步的重要性分数
            top_k: 选择的重要时间步数量

        Returns:
            weights: 距离权重张量
        """
        # 确保top_k不超过时间步数量
        n_timesteps = missing_indices.size(0)
        top_k = min(top_k, n_timesteps)

        # 找出最重要的k个时间步的索引
        _, top_indices = torch.topk(importance_scores, top_k)

        # 创建一个较小的权重矩阵，只包含重要时间步
        weights = torch.zeros(n_timesteps, top_k)

        # 计算这些重要时间步与所有时间步之间的距离权重
        for i in range(n_timesteps):
            for j, top_idx in enumerate(top_indices):
                if i != top_idx:
                    # 计算时间步距离的指数衰减权重
                    distance = abs(missing_indices[i] - missing_indices[top_idx])
                    weights[i, j] = torch.exp(-distance / self.window_size)

        # 归一化权重
        row_sums = weights.sum(dim=1, keepdim=True)
        weights = torch.where(row_sums > 0, weights / row_sums, weights)

        return weights

    def smoothness_loss(self, values: torch.Tensor) -> torch.Tensor:
        """
        计算时间序列的平滑性损失
        使用一阶差分来衡量序列的平滑程度

        Args:
            values: 时间序列值

        Returns:
            loss: 平滑性损失值
        """
        differences = values[1:] - values[:-1]
        return torch.mean(differences.pow(2))

    def temporal_consistency_loss(
            self,
            values: torch.Tensor,
            weights: torch.Tensor
    ) -> torch.Tensor:
        """
        计算时间一致性损失
        基于时间距离权重的加权一致性约束

        Args:
            values: 时间序列值
            weights: 时间距离权重

        Returns:
            loss: 时间一致性损失值
        """
        n = values.size(0)
        loss = 0.0

        for i in range(n):
            for j in range(n):
                if i != j:
                    # 加权的值差异
                    diff = (values[i] - values[j]).pow(2)
                    loss += weights[i, j] * diff

        return loss / (n * n)

    def calculate_weights(self, data_length: int, missing_indices: torch.Tensor,
                          importance_scores: torch.Tensor) -> torch.Tensor:
        """
        计算缺失值之间的时间权重关系
        使用简化且稳健的方法计算权重，避免索引越界问题

        Args:
            data_length: 数据总长度
            missing_indices: 缺失值的位置索引
            importance_scores: 时间步重要性分数

        Returns:
            weights: 权重矩阵 [n_missing, n_missing]
        """
        # 1. 确保缺失索引在有效范围内
        valid_indices = missing_indices[missing_indices < data_length]
        if len(valid_indices) == 0:
            return torch.ones(1, 1)  # 返回默认权重

        # 2. 调整重要性分数长度，确保与数据长度匹配
        if len(importance_scores) < data_length:
            # 如果重要性分数不够长，用1进行补齐
            pad_length = data_length - len(importance_scores)
            importance_scores = torch.cat([
                importance_scores,
                torch.ones(pad_length, device=importance_scores.device)
            ])
        else:
            # 如果重要性分数过长，进行截断
            importance_scores = importance_scores[:data_length]

        # 3. 创建距离矩阵，使用向量化操作
        n_missing = len(valid_indices)
        # 将索引展开成矩阵形式 [n_missing, 1] - [1, n_missing]
        indices_matrix = valid_indices.view(-1, 1) - valid_indices.view(1, -1)
        # 计算绝对距离
        distances = torch.abs(indices_matrix)
        # 使用广播计算权重矩阵
        weights = 1.0 / (1.0 + distances)
        # 将对角线置为0（不考虑自身）
        weights.fill_diagonal_(0.0)

        # 4. 计算重要性权重，使用索引操作
        # 确保索引在范围内
        valid_importance_indices = valid_indices[valid_indices < len(importance_scores)]
        # 获取重要性分数
        importance_weights = torch.ones(n_missing, device=weights.device)
        importance_weights[:len(valid_importance_indices)] = importance_scores[valid_importance_indices]

        # 6. 结合距离权重和重要性权重
        weights = weights * importance_weights.unsqueeze(1)

        # 7. 归一化权重
        row_sums = weights.sum(dim=1, keepdim=True)
        weights = torch.where(row_sums > 0, weights / row_sums, weights)

        return weights

    def impute_missing_values(
            self,
            data: torch.Tensor,
            missing_mask: torch.Tensor,
            importance_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行缺失值填补

        Args:
            data: 原始数据张量
            missing_mask: 缺失值掩码
            importance_scores: 时间步重要性分数

        Returns:
            imputed_data: 填补后的数据
            metrics: 优化过程的指标
        """
        # 1. 将数据迁移到正确的设备上
        device = next(self.model.parameters()).device
        data = data.to(device)
        missing_mask = missing_mask.to(device)
        importance_scores = importance_scores.to(device)

        # 2. 初始化填补数据
        imputed_data = data.clone()

        # 3. 找出包含缺失值的时间步
        missing_indices = torch.where(missing_mask.any(dim=-1))[0]
        if len(missing_indices) == 0:
            return imputed_data, {'total_loss': [], 'smoothness_loss': []}

        # 4. 初始填充：使用窗口平均值
        for idx in missing_indices:
            # 定义时间窗口范围
            window_start = max(0, idx - self.window_size // 2)
            window_end = min(data.shape[0], idx + self.window_size // 2)

            # 获取窗口内的数据
            window_data = data[window_start:window_end]
            window_mask = missing_mask[window_start:window_end]

            # 对每个特征分别处理
            for feature in range(data.shape[-1]):
                if missing_mask[idx, feature]:
                    # 获取当前特征的有效值
                    valid_data = window_data[~window_mask[:, feature], feature]
                    if len(valid_data) > 0:
                        # 使用窗口内的平均值
                        imputed_data[idx, feature] = valid_data.mean()
                    else:
                        # 如果窗口内没有有效值，使用整体平均值
                        imputed_data[idx, feature] = data[~missing_mask[:, feature], feature].mean()

        # 5. 计算时间权重
        weights = self.calculate_weights(
            data_length=data.shape[0],
            missing_indices=missing_indices,
            importance_scores=importance_scores
        )

        # 6. 优化填补值
        missing_values = imputed_data[missing_indices].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([missing_values], lr=self.learning_rate)

        metrics = {'total_loss': [], 'smoothness_loss': []}

        # 7. 迭代优化
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # 更新填补数据
            current_data = imputed_data.clone()
            current_data[missing_indices] = missing_values

            # 计算损失函数
            smooth_loss = torch.mean(torch.diff(current_data, dim=0).pow(2))
            consistency_loss = torch.mean(weights * torch.cdist(missing_values, missing_values))

            # 计算总损失
            total_loss = (
                    smooth_loss * self.alpha_smooth +
                    consistency_loss * self.alpha_temporal
            )

            # 记录损失值
            metrics['total_loss'].append(total_loss.item())
            metrics['smoothness_loss'].append(smooth_loss.item())

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 检查收敛性
            if iteration > 0:
                if abs(metrics['total_loss'][-1] - metrics['total_loss'][-2]) < self.convergence_threshold:
                    break

        # 8. 更新最终填补值
        imputed_data[missing_indices] = missing_values.detach()

        return imputed_data, metrics

class ImputationTrainer:
    """
    缺失值填补训练器
    结合时间步重要性的缺失值填补优化
    """

    def __init__(
            self,
            model: nn.Module,
            imputer: TemporalImputer,
            device: torch.device,
            learning_rate: float = 0.001,
            num_epochs: int = 1,
            batch_size: int = 32,
            patience: int = 10
    ):
        self.model = model
        self.imputer = imputer
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience

    def train_imputation(
            self,
            data: torch.Tensor,
            missing_mask: torch.Tensor,
            importance_scores: torch.Tensor,
            val_data: torch.Tensor = None,
            val_mask: torch.Tensor = None
    ):
        """
        训练缺失值填补

        Args:
            data: 原始数据 [timesteps, features]
            missing_mask: 缺失值掩码
            importance_scores: 时间步重要性分数
            val_data: 验证数据（可选）
            val_mask: 验证集缺失掩码（可选）

        Returns:
            imputed_data: 填补后的数据
            training_info: 训练过程信息
        """
        print("开始缺失值填补训练...")

        # 初始填补
        imputed_data, _ = self.imputer.impute_missing_values(
            data, missing_mask, importance_scores
        )

        # 创建数据集
        dataset = TimeDataset(
            data=imputed_data.numpy(),
            window_size=self.imputer.window_size,
            stride=1
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        # 优化器
        optimizer = optim.Adam([
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': [imputed_data[missing_mask]], 'lr': self.learning_rate * 0.1}
        ])

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 训练记录
        best_loss = float('inf')
        best_imputed = None
        patience_counter = 0
        training_info = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }

        for epoch in range(self.num_epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0

            for batch_x, batch_y, clean_target, idx in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                x_mark = torch.zeros(batch_x.shape[0], batch_x.shape[1], 4).to(self.device)
                dec_inp = torch.zeros(batch_x.shape[0], self.imputer.window_size, batch_x.shape[2]).to(self.device)

                outputs = self.model(batch_x, x_mark, dec_inp, None)

                # 计算损失
                reconstruction_loss = nn.MSELoss()(outputs, batch_y)

                # 计算平滑性损失
                smoothness_loss = self.imputer.smoothness_loss(imputed_data)

                # 计算时间一致性损失
                temporal_weights = self.imputer.calculate_weights(
                    torch.where(missing_mask)[0]
                )
                temporal_loss = self.imputer.temporal_consistency_loss(
                    imputed_data[missing_mask],
                    temporal_weights
                )

                # 总损失
                loss = (
                        reconstruction_loss +
                        self.imputer.alpha_smooth * smoothness_loss +
                        self.imputer.alpha_temporal * temporal_loss
                )

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader)
            training_info['train_losses'].append(avg_train_loss)

            # 验证阶段
            if val_data is not None and val_mask is not None:
                val_loss = self.validate_imputation(val_data, val_mask)
                training_info['val_losses'].append(val_loss)

                # 早停检查
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_imputed = imputed_data.clone()
                    patience_counter = 0
                    training_info['best_epoch'] = epoch
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                scheduler.step(val_loss)
            else:
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    best_imputed = imputed_data.clone()
                    training_info['best_epoch'] = epoch

                scheduler.step(avg_train_loss)

            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}")
            if val_data is not None:
                print(f"Validation Loss = {val_loss:.6f}")

        return best_imputed if best_imputed is not None else imputed_data, training_info

    def validate_imputation(self, val_data: torch.Tensor, val_mask: torch.Tensor) -> float:
        """
        验证填补效果

        Args:
            val_data: 验证数据
            val_mask: 验证集缺失掩码

        Returns:
            float: 验证损失
        """
        self.model.eval()
        with torch.no_grad():
            # 使用当前模型进行预测
            x_mark = torch.zeros(1, val_data.shape[0], 4).to(self.device)
            dec_inp = torch.zeros(1, self.imputer.window_size, val_data.shape[1]).to(self.device)

            val_data_input = val_data.unsqueeze(0).to(self.device)
            outputs = self.model(val_data_input, x_mark, dec_inp, None)

            # 计算验证损失
            val_loss = nn.MSELoss()(
                outputs.squeeze(0)[val_mask],
                val_data[val_mask]
            )

        return val_loss.item()

    def plot_training_progress(self, training_info: dict, save_path: str = None):
        """
        绘制训练进度

        Args:
            training_info: 训练信息字典
            save_path: 保存路径（可选）
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(training_info['train_losses'], label='Training Loss')
        if 'val_losses' in training_info:
            plt.plot(training_info['val_losses'], label='Validation Loss')

        plt.axvline(x=training_info['best_epoch'], color='r', linestyle='--', label='Best Model')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()