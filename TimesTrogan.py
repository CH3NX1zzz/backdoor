import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class TimesTroganFGSM:
    def __init__(self, model, epsilon=0.3,seq_len=12, alpha=2 / 255, num_steps=40):
        """
        TimesTrogan-FGSM方法实现
        Args:
            model: 目标TimesNet模型
            epsilon: 扰动大小上限
            alpha: 每步扰动大小
            num_steps: 迭代步数
        """
        self.model = model
        self.epsilon = epsilon
        self.seq_len = seq_len
        self.alpha = alpha
        self.num_steps = num_steps
        self.importance_scores = None

    def calculate_importance(self, x, y):
        """
        计算时间步重要性分数
        Args:
            x: 输入序列 [batch_size, seq_len, feature_dim]
            y: 目标值
        Returns:
            importance_scores: 每个时间步的重要性分数
        """
        x_perturbed = x.clone().detach().requires_grad_(True)
        self.model.eval()

        # 记录每个时间步的梯度
        gradients = []

        # 对每个时间步进行FGSM攻击
        for t in range(x.shape[1]):
            # 前向传播
            output = self.model(x_perturbed, None, None, None)
            loss = F.mse_loss(output, y)

            # 反向传播
            loss.backward()

            # 获取该时间步的梯度
            grad = x_perturbed.grad[:, t, :].clone().detach()
            gradients.append(torch.norm(grad, dim=1).mean().item())

            # 清除梯度
            x_perturbed.grad.zero_()

        # 计算重要性分数
        importance_scores = torch.tensor(gradients)
        importance_scores = F.softmax(importance_scores, dim=0)

        self.importance_scores = importance_scores
        return importance_scores

    def generate_adversarial(self, x, y):
        """
        生成对抗样本
        Args:
            x: 输入序列
            y: 目标值
        Returns:
            x_adv: 对抗样本
        """
        x_adv = x.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad = True

            # 前向传播
            output = self.model(x_adv, None, None, None)
            loss = F.mse_loss(output, y)

            # 反向传播
            loss.backward()

            # 使用重要性分数加权的FGSM更新
            if self.importance_scores is not None:
                weighted_grad = x_adv.grad * self.importance_scores.view(1, -1, 1)
            else:
                weighted_grad = x_adv.grad

            # FGSM更新
            with torch.no_grad():
                grad_sign = weighted_grad.sign()
                x_adv = x_adv.detach() + self.alpha * grad_sign

                # 限制扰动范围
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv

    def analyze_timeseries(self, dataloader, device):
        """
        分析时间序列数据中各个时间步的重要性

        Args:
            dataloader: 数据加载器
            device: 计算设备

        Returns:
            tuple: (重要性分数, 对抗样本)
        """
        from collections import defaultdict
        self.model.eval()

        # 使用defaultdict来收集每个时间步的重要性分数
        importance_by_timestep = defaultdict(list)
        total_timesteps = len(dataloader.dataset.data)  # 总时间步数

        for batch_x, batch_y, _, timesteps in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 对每个窗口计算重要性
            for i in range(batch_x.shape[1]):  # 遍历窗口中的每个时间步
                # 创建扰动
                perturbed_x = batch_x.clone()
                perturbed_x[:, i, :] += self.epsilon * torch.randn_like(perturbed_x[:, i, :])

                # 计算原始输出和扰动输出
                with torch.no_grad():
                    x_mark = torch.zeros(batch_x.shape[0], batch_x.shape[1], 4).to(device)
                    dec_inp = torch.zeros_like(batch_y).to(device)

                    orig_output = self.model(batch_x, x_mark, dec_inp, None)
                    pert_output = self.model(perturbed_x, x_mark, dec_inp, None)

                    # 计算输出差异
                    output_diff = torch.norm(pert_output - orig_output, dim=2).mean()

                    # 将重要性分数与实际时间步关联
                    actual_timestep = timesteps[0].item() + i
                    if actual_timestep < total_timesteps:
                        importance_by_timestep[actual_timestep].append(output_diff.item())

        # 计算每个时间步的平均重要性
        final_importance = torch.zeros(total_timesteps)
        for timestep in range(total_timesteps):
            if timestep in importance_by_timestep:
                final_importance[timestep] = torch.tensor(np.mean(importance_by_timestep[timestep]))

        # 归一化重要性分数
        if torch.sum(final_importance) > 0:
            final_importance = final_importance / torch.sum(final_importance)

        return final_importance, None


def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for x, y, clean_y, idx in val_loader:
            x = x.to(device)
            y = y.to(device)

            x_mark = torch.zeros(x.shape[0], x.shape[1], 4).to(device)
            dec_inp = torch.zeros_like(y).to(device)
            outputs = model(x, x_mark, dec_inp, None)

            loss = criterion(outputs, y)
            total_val_loss += loss.item()

    return total_val_loss / len(val_loader)

def visualize_importance(importance_scores, save_path=None):
    """
    可视化重要性分数
    Args:
        importance_scores: 重要性分数
        save_path: 保存路径
    """


    plt.figure(figsize=(10, 6))
    plt.plot(importance_scores.cpu().numpy(), marker='o')
    plt.title('Time Step Importance Scores')
    plt.xlabel('Time Step')
    plt.ylabel('Importance Score')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


