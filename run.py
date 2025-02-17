from TimesTrogan import *
from sklearn.preprocessing import StandardScaler
from dataset import *
import yaml
from easydict import EasyDict as edict
from model import TimesNet
from types import SimpleNamespace
from torch import optim
from importance_analysis import run_importance_analysis
from model_utils import ModelManager
import os
import yaml
import argparse  # 添加参数解析功能

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict['TimesNet'])
def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    训练模型的函数
    """

    data = np.load('./data/PEMS03/PEMS03.npz')
    raw_data = data['data']  # 假设数据格式为 [timestamps, nodes, features]

    # 数据归一化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_data.reshape(-1, raw_data.shape[-1])).reshape(raw_data.shape)
    train_ratio = 0.6  # 训练集比例改为60%
    val_ratio = 0.2  # 验证集使用20%

    train_length = int(len(raw_data) * train_ratio)
    val_length = int(len(raw_data) * (train_ratio + val_ratio))

    # 三部分数据集的划分
    train_data = data_scaled[:train_length]
    val_data = data_scaled[train_length:val_length]
    test_data = data_scaled[val_length:]

    # 创建三个数据集的加载器
    train_dataset = TimeDataset(
        data=train_data,
        window_size=12,
        stride=1,  # 每次移动一个时间步
        timestamps=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,  # 重要：不要打乱顺序
        num_workers=2
    )

    val_dataset = TimeDataset(
        data=val_data,
        window_size=12,  # 替换 num_for_hist
        stride=1,  # 新增参数
        timestamps=None
    )

    test_dataset = TimeDataset(
        data=test_data,
        window_size=12,  # 替换 num_for_hist
        stride=1,  # 新增参数
        timestamps=None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,  # 验证集不需要打乱顺序
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # 添加早停机制的相关变量
    best_val_loss = float('inf')
    patience = 5  # 如果验证集损失在5个epoch内没有改善，就停止训练
    patience_counter = 0
    best_model_state = None
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam优化器，学习率设为0.0001
    criterion = nn.MSELoss()  # 均方误差损失函数
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    num_epochs = 10
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for batch_idx, (x, y, clean_target, idx) in enumerate(train_loader):
            if len(x.shape) == 4:
                x = x.squeeze(-1)

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()  # 记得在每个batch开始时清零梯度

            x_mark = torch.zeros(x.shape[0], x.shape[1], 4).to(device)
            dec_inp = torch.zeros(x.shape[0], config.pred_len, x.shape[2]).to(device)

            outputs = model(x, x_mark, dec_inp, None)

            loss = criterion(outputs, y)
            loss.backward()

            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0

        with torch.no_grad():  # 在验证时不需要计算梯度
            for x, y, clean_target, idx in val_loader:
                if len(x.shape) == 4:
                    x = x.squeeze(-1)

                x = x.to(device)
                y = y.to(device)

                x_mark = torch.zeros(x.shape[0], x.shape[1], 4).to(device)
                dec_inp = torch.zeros(x.shape[0], config.pred_len, x.shape[2]).to(device)

                outputs = model(x, x_mark, dec_inp, None)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 打印训练和验证的损失
        print(f'Epoch {epoch} completed:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                # 恢复最佳模型状态
                model.load_state_dict(best_model_state)
                break

    # 在测试集上评估最终模型
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y, clean_target, idx in test_loader:
            if len(x.shape) == 4:
                x = x.squeeze(-1)

            x = x.to(device)
            y = y.to(device)

            x_mark = torch.zeros(x.shape[0], x.shape[1], 4).to(device)
            dec_inp = torch.zeros(x.shape[0], config.pred_len, x.shape[2]).to(device)

            outputs = model(x, x_mark, dec_inp, None)
            loss = criterion(outputs, y)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.4f}')

    # 运行重要性分析
    importance_scores, top_timesteps = run_importance_analysis(
        model=model,
        data=train_data,
        device=device,
        save_path='importance_scores.png',
        epsilon=0.01,
        seq_len=12
    )

    print(f"Top 5 most important time steps: {top_timesteps}")
    # [原有的训练代码]
    return model, {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}

def main():
    # 设置随机种子确保实验可重复性


    torch.manual_seed(42)
    np.random.seed(42)



    # 数据归一化
    data = np.load('./data/PEMS03/PEMS03.npz')
    raw_data = data['data']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_data.reshape(-1, raw_data.shape[-1])).reshape(raw_data.shape)

    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_manager = ModelManager()

    parser = argparse.ArgumentParser(description='Train or load TimesNet model')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    args = parser.parse_args()

    # 加载和处理数据
    model = TimesNet(config)
    if args.train:
        # 训练新模型
        train_dataset = TimeDataset(data=data_scaled[:int(len(raw_data) * 0.6)], window_size=12, stride=1)
        val_dataset = TimeDataset(data=data_scaled[int(len(raw_data) * 0.6):int(len(raw_data) * 0.8)], window_size=12,
                                  stride=1)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        model, training_metrics = train_model(model, train_loader, val_loader, device)

        # 保存模型和训练结果
        model_dir = model_manager.save_model(
            model,
            "TimesNet",
            metadata={
                'training_metrics': training_metrics,
                'config': config
            }
        )
    else:
        # 加载最新的已训练模型
        try:
            latest_model_dir = max(
                [d for d in os.listdir(model_manager.save_dir) if d.startswith("TimesNet")],
                key=lambda x: os.path.getctime(os.path.join(model_manager.save_dir, x))
            )
            model_path = os.path.join(model_manager.save_dir, latest_model_dir, "model.pth")
            model = model_manager.load_model(model, model_path)
            model_dir = os.path.join(model_manager.save_dir, latest_model_dir)
            print(f"Loaded model from: {model_path}")
        except (FileNotFoundError, ValueError):
            print("No saved model found. Please train a new model using --train flag.")
            return

        # 运行重要性分析
    importance_scores, top_timesteps = run_importance_analysis(
        model=model,
        data=data_scaled[:int(len(raw_data)*0.6)],
        device=device,
        save_path=os.path.join(model_dir, 'importance_scores.png')
    )

    # 保存分析结果
    model_manager.save_importance_analysis(model_dir, importance_scores, top_timesteps)

    print(f"Top 5 most important time steps: {top_timesteps}")
    print(f"Results saved in: {model_dir}")

    # 初始化模型配置


    # 初始化模型



if __name__ == "__main__":
    main()