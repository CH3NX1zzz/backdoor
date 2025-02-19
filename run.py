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
from dataset import MissingDataGenerator
from imputer import TemporalImputer,ImputationTrainer

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict['TimesNet'])

data = np.load('./data/PEMS03/PEMS03.npz')
raw_data = data['data']  # 假设数据格式为 [timestamps, nodes, features]

    # 数据归一化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(raw_data.reshape(-1, raw_data.shape[-1])).reshape(raw_data.shape)
data_2d = data_scaled.reshape(data_scaled.shape[0], -1)
data_tensor = torch.FloatTensor(data_2d)

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


def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    训练模型的函数
    """



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
    num_epochs = 1

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
    config.enc_in = data_tensor.shape[1]
    if args.train:
        # 训练新模型
        print("Training new model...")
        train_dataset = TimeDataset(data=data_scaled[:int(len(raw_data) * 0.6)], window_size=12, stride=1)
        val_dataset = TimeDataset(data=data_scaled[int(len(raw_data) * 0.6):int(len(raw_data) * 0.8)], window_size=12,
                                  stride=1)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        model, training_metrics = train_model(model, train_loader, val_loader, device)

        # 保存模型和训练结果
        model_dir = model_manager.save_model(
            model=model,
            model_name="TimesNet",
            metadata={
                'training_metrics': {
                    'train_loss': float(training_metrics['train_loss']),
                    'val_loss': float(training_metrics['val_loss'])
                },
                'config': {
                    'pred_len': config.pred_len,
                    'seq_len': config.seq_len,
                    'label_len': config.label_len,
                    'd_model': config.d_model,
                    'enc_in': config.enc_in,
                    'e_layers': config.e_layers,
                    'd_ff': config.d_ff,
                    'top_k': config.top_k,
                    'num_kernels': config.num_kernels
                }
            }
        )
        print(f"模型已保存到: {model_dir}")
    else:
        print("Loading model...")
        # 加载最新的已训练模型
        try:
            # 检查保存目录是否存在
            if not os.path.exists(model_manager.save_dir):
                raise FileNotFoundError(f"模型保存目录不存在: {model_manager.save_dir}")

            # 获取所有TimesNet开头的目录
            model_dirs = [
                d for d in os.listdir(model_manager.save_dir)
                if os.path.isdir(os.path.join(model_manager.save_dir, d))
                   and d.startswith("TimesNet")
            ]

            if not model_dirs:
                raise FileNotFoundError("未找到已保存的模型目录")

            # 获取最新的模型目录
            latest_model_dir = max(
                model_dirs,
                key=lambda x: os.path.getctime(os.path.join(model_manager.save_dir, x))
            )

            model_path = os.path.join(model_manager.save_dir, latest_model_dir, "saved_models")

            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            # 加载模型
            model = model_manager.load_model(model, model_path)
            model_dir = os.path.join(model_manager.save_dir, latest_model_dir)

            print(f"已加载模型: {model_path}")
            print(f"模型目录: {model_dir}")

        except FileNotFoundError as e:
            print(f"错误: {e}")
            print("请使用 --train 参数训练新模型")
            return
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            print("请使用 --train 参数训练新模型")
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


    # 初始化生成器
    generator = MissingDataGenerator(seed=42)

    # 生成缺失值
    missing_data, missing_mask = generator.generate_mixed_missing(
        data_tensor,
        missing_rates={'mcar': 0.1, 'blocks': 0.1}
    )
    print(f"生成的缺失数据形状: {missing_data.shape}")
    print(f"缺失率: {missing_mask.float().mean().item():.2%}")
    # 验证结果
    validation_results = generator.get_missing_statistics(missing_mask)
    print("验证结果:", validation_results)

    print("\n初始化填补训练...")
    imputer = TemporalImputer(
        model=model,
        window_size=12,
        alpha_smooth=0.1,
        alpha_temporal=0.1
    )

    trainer = ImputationTrainer(
        model=model,
        imputer=imputer,
        device=device,
        learning_rate=0.001,
        num_epochs=100,
        batch_size=32,
        patience=10
    )



    # 15. 训练填补模型
    print("\n开始填补训练...")
    imputed_data, training_info = trainer.train_imputation(
        data=missing_data,
        missing_mask=missing_mask,
        importance_scores=importance_scores,
        val_data=torch.FloatTensor(val_data),
        val_mask=missing_mask[train_length:val_length]
    )
    imputed_data_3d = imputed_data.reshape(data_scaled.shape)

    # 16. 绘制训练进度
    print("\n绘制训练进度...")
    trainer.plot_training_progress(
        training_info,
        save_path=os.path.join(model_dir, 'imputation_training.png')
    )

    # 17. 保存填补结果
    print("\n保存结果...")
    results_path = os.path.join(model_dir, 'imputation_results.npz')
    np.savez(
        results_path,
        original_data=data_tensor.numpy(),
        missing_data=missing_data.numpy(),
        imputed_data=imputed_data.numpy(),
        missing_mask=missing_mask.numpy()
    )
    print(f"结果已保存到: {results_path}")

    # 18. 计算和显示评估指标
    print("\n计算评估指标...")
    mse = np.mean((data_tensor.numpy()[missing_mask.numpy()] -
                   imputed_data.numpy()[missing_mask.numpy()]) ** 2)
    mae = np.mean(np.abs(data_tensor.numpy()[missing_mask.numpy()] -
                         imputed_data.numpy()[missing_mask.numpy()]))
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")


if __name__ == "__main__":
    main()