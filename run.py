from TimesTrogan import *
from sklearn.preprocessing import StandardScaler
from dataset import *
import yaml
from easydict import EasyDict as edict
from model import TimesNet
from types import SimpleNamespace
from torch import optim



def main():
    # 设置随机种子确保实验可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载和处理数据
    data = np.load('./data/PEMS03/PEMS03.npz')
    raw_data = data['data']  # 假设数据格式为 [timestamps, nodes, features]

    # 数据归一化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_data.reshape(-1, raw_data.shape[-1])).reshape(raw_data.shape)


    # 初始化模型配置
    model_config = {
        'task_name': 'short_term_forecast',
        'seq_len': 12,
        'label_len': 12,
        'pred_len': 12,
        'e_layers': 2,
        'embed': 'timeF',
        'freq': 'h',
        'dropout': 0.1,
        'top_k': 5,
        'd_model': 256,
        'd_ff': 256,
        'factor': 3,
        'num_kernels': 6,
        'output_attention': False,
        'distil': True,
        'enc_in': 358,  # num_of_vertices from dataset config
        'dec_in': 358,
        'c_out': 358
    }
    config = SimpleNamespace(**model_config)
    # 初始化模型
    model = TimesNet(config)
    model = model.to(device)


    # 划分训练集和测试集
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

    # 初始化 TimesTrogan-FGSM
    timestrogan = TimesTroganFGSM(
        model=model,
        epsilon=0.01,  # 扰动大小
        seq_len=12  # 确保这与您的数据序列长度匹配
    )

    # 分析时间序列
    importance_scores, adv_examples = timestrogan.analyze_timeseries(train_loader, device)

    # 可视化结果
    visualize_importance(importance_scores, 'importance_scores.png')

    # 打印最重要的时间步
    top_k = 5
    top_indices = torch.topk(importance_scores, top_k).indices
    print(f"Top {top_k} most important time steps: {top_indices.tolist()}")


if __name__ == "__main__":
    main()