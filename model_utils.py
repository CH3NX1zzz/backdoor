import torch
import json
import os
from datetime import datetime


class ModelManager:
    def __init__(self, save_dir='saved_models'):
        """
        初始化模型管理器，用于处理模型的保存和加载

        参数:
            save_dir: 保存模型和结果的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_model(self, model, model_name, metadata=None):
        """
        保存模型和相关元数据

        参数:
            model: 训练好的模型
            model_name: 模型名称
            metadata: 额外的元数据信息，如训练参数、性能指标等
        """
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建保存路径
        model_dir = os.path.join(self.save_dir, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型状态
        model_path = os.path.join(model_dir, "saved_models/parameter")
        torch.save(model.state_dict(), model_path)

        # 保存元数据
        if metadata:
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        return model_dir

    def load_model(self, model, model_path):
        """
        加载保存的模型

        参数:
            model: 模型实例
            model_path: 模型文件路径
        """
        model.load_state_dict(torch.load(model_path))
        return model

    def save_importance_analysis(self, model_dir, importance_scores, top_timesteps):
        """
        保存重要性分析结果

        参数:
            model_dir: 模型目录
            importance_scores: 重要性分数
            top_timesteps: 最重要的时间步
        """
        results = {
            'importance_scores': importance_scores.tolist(),
            'top_timesteps': top_timesteps
        }

        results_path = os.path.join(model_dir, "importance_analysis.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)