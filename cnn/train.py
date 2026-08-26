# random：设置随机种子用（保证可复现）
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import get_ds
from model import build_cnn


def set_seed(seed):
    """把所有随机源设成同一个种子，让每次运行结果完全一致（可复现性）。"""
    random.seed(seed)                # Python 内置 random
    np.random.seed(seed)             # numpy（数据增强、打乱可能用到）
    torch.manual_seed(seed)          # PyTorch CPU/GPU 的随机数
    torch.cuda.manual_seed_all(seed) # 所有 CUDA 设备


def evaluate(model, loader, loss_fn, device):
    """在给定 DataLoader 上计算平均损失和准确率。

    只做前向、不做反向，也不更新权重，是"检验模型当前水平"用的。
    """
    model.eval()  # 切换到评估模式：关掉 Dropout、BatchNorm 用全局统计量
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():  # 关闭梯度计算：省显存、省时间（推理不需要梯度）
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # 数据搬到目标设备
            logits = model(x)  # 前向传播 -> (batch, num_classes) 每类的原始分数

            # loss_fn 默认 reduction='mean'，是这批的平均损失；
            # 乘 x.size(0)（batch 大小）还原成"这批的总损失"，方便下面统一加权平均
            total_loss += loss_fn(logits, y).item() * x.size(0)

            # argmax(dim=1)：每个样本取分数最高的那个类作为预测
            # == y：和真实标签逐元素比较，得到 bool 张量；sum() 统计预测正确的个数
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)  # 累加见过的样本总数

    # 总损失 / 总样本数 = 整个数据集的平均损失；correct / total = 准确率
    return total_loss / total, correct / total


def get_model(config):
    """按配置构建模型。"""
    return build_cnn(config)


def train_model(config):
    # ── step1: 选择设备 ──────────────────────────────────────────────
    # 优先级: CUDA (NVIDIA GPU) > MPS (Mac 的 Apple Silicon GPU) > CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    # ── step2: 固定随机种子，保证可复现 ───────────────────────────────
    set_seed(config["seed"])

    # ── step3: 创建模型权重保存目录 ──────────────────────────────────
    # 目录名形如 "cifar10_weights"，用于存放每个 epoch 的 .pt 权重
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # ── step4: 获取数据 ──────────────────────────────────────────────
    # CNN 版本只返回 train_loader 和 test_loader（没切验证集）
    train_loader, test_loader = get_ds(config)

    # ── step5: 构建模型并搬到设备 ────────────────────────────────────
    model = get_model(config).to(device)

    # ── step6: 打印参数量 ────────────────────────────────────────────
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── step7: 创建优化器 ────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # ── step8: 创建损失函数 ──────────────────────────────────────────
    # CrossEntropyLoss 内部自带 Softmax，所以模型最后一层不要自己加激活函数
    loss_fn = nn.CrossEntropyLoss().to(device)

    # ── step9: 学习率调度器（CNN 相比 MLP 多出的部分）─────────────────
    # StepLR: 每训练 lr_scheduler_step 个 epoch，学习率乘上 gamma（衰减一次）
    # 目的是训练后期学习率变小，帮助收敛到更精细的位置
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_scheduler_step'], gamma=config['lr_scheduler_gamma']
    )

    # ── step10: 断点续训 ─────────────────────────────────────────────
    # 如果之前训练过、有权重文件，就加载它接着训，而不是从头开始
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    # preload='latest' 找最新的权重；preload=具体数字则找对应 epoch 的权重；preload=None 则从头训
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)  # 读出 checkpoint（一个 dict）
        model.load_state_dict(state['model_state_dict'])      # 恢复模型权重
        initial_epoch = state['epoch'] + 1                    # 从下一个 epoch 继续
        optimizer.load_state_dict(state['optimizer_state_dict'])  # 恢复优化器状态（动量等）
        global_step = state['global_step']                    # 恢复全局步数
    else:
        print('No model to preload, starting from scratch')

    # ── step11: 训练主循环 ───────────────────────────────────────────
    # 外层循环: 每个 epoch 完整过一遍训练集
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()  # 清一次显存缓存（主要对 GPU 有意义）
        model.train()  # 切换到训练模式：开启 Dropout、BatchNorm 用当前 batch 统计量

        total_loss = 0.0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")  # 进度条

        # 内层循环: 遍历训练集的每一个 batch
        for x, y in batch_iterator:
            x, y = x.to(device), y.to(device)  # 数据搬到设备

            # ① 前向: 输入 -> 输出每类的 logits
            logits = model(x)
            # ② 计算 loss: 预测和真实标签的差距（标量，batch 平均）
            loss = loss_fn(logits, y)

            # ③ 反向传播: 从 loss 反推出每个参数的梯度（存进 .grad）
            loss.backward()
            # ④ 更新权重: 优化器用 .grad 里的梯度真正修改参数
            optimizer.step()
            # ⑤ 清空梯度: 否则下次 backward 会在旧梯度上累加
            optimizer.zero_grad(set_to_none=True)

            # 记账: 平均 loss × batch 大小 = 这批总 loss，累加到 epoch 维度
            total_loss += loss.item() * x.size(0)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})  # 进度条上显示当前 loss
            global_step += 1  # 全局步数，跨 epoch 不重置

        # ── epoch 结束收尾 ──────────────────────────────────────────
        scheduler.step()  # 学习率衰减步进（StepLR 每 step_size 个 epoch 衰减一次）

        # 整个训练集的平均损失 = 总损失 / 总样本数
        train_loss = total_loss / len(train_loader.dataset)

        # 在测试集上评估（不更新权重）；CNN 这版没切验证集，直接用测试集观察
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        # 打印里顺带显示当前学习率，方便确认调度器在起作用
        print(f"  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc * 100:.2f}%  lr={optimizer.param_groups[0]['lr']:.2e}")

        # 保存这一个 epoch 的 checkpoint，文件名带 epoch 编号
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,                              # 第几个 epoch（断点续训要用）
            'model_state_dict': model.state_dict(),      # 模型权重
            'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
            'global_step': global_step                   # 全局步数
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # 忽略告警，输出更干净
    config = get_config()  # 读取配置
    train_model(config)
