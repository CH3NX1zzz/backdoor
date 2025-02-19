import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from dataset import TimeDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    def calculate_temporal_weights(self, missing_indices: torch.Tensor) -> torch.Tensor:
        """
        计算时间步之间的距离权重
        对于每个缺失值，计算其与最近的非缺失值的距离权重

        Args:
            missing_indices: 缺失值的索引张量

        Returns:
            weights: 距离权重张量
        """
        n_timesteps = missing_indices.size(0)
        weights = torch.zeros(n_timesteps, n_timesteps)

        for i in range(n_timesteps):
            for j in range(n_timesteps):
                if i != j:
                    # 计算时间步距离的指数衰减权重
                    distance = abs(missing_indices[i] - missing_indices[j])
                    weights[i, j] = torch.exp(-distance / self.window_size)

        # 归一化权重
        weights = F.normalize(weights, p=1, dim=1)
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

    def impute_missing_values(
            self,
            data: torch.Tensor,
            missing_mask: torch.Tensor,
            importance_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        优化填补缺失值

        Args:
            data: 原始数据张量
            missing_mask: 缺失值掩码
            importance_scores: 时间步重要性分数

        Returns:
            imputed_data: 填补后的数据
            metrics: 优化过程的指标
        """
        # 初始化缺失值（使用临近值的平均值）
        if len(data.shape) == 2:
            data = data.unsqueeze(1)  # 添加nodes维度
            missing_mask = missing_mask.unsqueeze(1)

        device = next(self.model.parameters()).device
        data = data.to(device)
        missing_mask = missing_mask.to(device)
        importance_scores = importance_scores.to(device)

        imputed_data = data.clone()
        missing_indices = torch.where(missing_mask)[0].unique()

        # 初始填充
        for idx in missing_indices:
            window_start = max(0, idx - self.window_size)
            window_end = min(data.size(0), idx + self.window_size)
            window_values = data[window_start:window_end]
            valid_mask = ~missing_mask[window_start:window_end]
            valid_values = window_values[~missing_mask[window_start:window_end]]

            if len(valid_values) > 0:
                imputed_data[idx] = valid_values.mean()
            else:
                valid_mask_all = ~missing_mask
                imputed_data[idx] = data[~missing_mask].mean()

        # 计算时间距离权重
        temporal_weights = self.calculate_temporal_weights(missing_indices)
        temporal_weights = temporal_weights.to(device)

        # 优化过程
        imputed_values = imputed_data[missing_indices].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([imputed_values], lr=self.learning_rate)

        metrics = {
            'total_loss': [],
            'smoothness_loss': [],
            'temporal_loss': [],
            'reconstruction_loss': []
        }

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # 更新缺失值
            current_data = imputed_data.clone()
            current_data[missing_indices] = imputed_values
            if len(current_data.shape) == 2:
                model_input = current_data.unsqueeze(0)  # [1, timestamps, features]
            else:
                model_input = current_data

            print(f"Model input shape: {model_input.shape}")  # 调试信息


            seq_len = model_input.shape[1]
            x_mark = torch.zeros(1, seq_len, 4).to(device)
            dec_inp = torch.zeros(1, seq_len, model_input.shape[-1]).to(device)

            # 计算重建损失
            output = self.model(model_input, x_mark, dec_inp, None)
            reconstruction_loss = F.mse_loss(output.squeeze(0), current_data)

            # 计算平滑性损失
            smooth_loss = self.smoothness_loss(current_data)

            # 计算时间一致性损失
            temporal_loss = self.temporal_consistency_loss(
                imputed_values,
                temporal_weights
            )

            # 基于重要性分数加权的总损失
            importance_weights = importance_scores[missing_indices]
            total_loss = (
                    reconstruction_loss +
                    self.alpha_smooth * smooth_loss * importance_weights.mean() +
                    self.alpha_temporal * temporal_loss * importance_weights.mean()
            )

            # 记录损失
            metrics['total_loss'].append(total_loss.item())
            metrics['smoothness_loss'].append(smooth_loss.item())
            metrics['temporal_loss'].append(temporal_loss.item())
            metrics['reconstruction_loss'].append(reconstruction_loss.item())

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 检查收敛
            if iteration > 0:
                loss_diff = abs(metrics['total_loss'][-1] - metrics['total_loss'][-2])
                if loss_diff < self.convergence_threshold:
                    break

        # 更新最终的填补值
        imputed_data[missing_indices] = imputed_values.detach()

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
            num_epochs: int = 100,
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
                temporal_weights = self.imputer.calculate_temporal_weights(
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