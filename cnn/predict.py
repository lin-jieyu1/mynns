# torch：加载模型、推理
import torch
# matplotlib：把图片和预测结果画出来
import matplotlib.pyplot as plt

from config import get_config, latest_weights_file_path
from dataset import get_ds, CIFAR_MEAN, CIFAR_STD
from model import build_cnn

# 只覆盖 CIFAR-10 的 10 类；换成 CIFAR-100 时改用数字下标显示
CIFAR10_NAMES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]


def main():
    # ── 选择设备（和 train.py 里同一套逻辑）───────────────────────────
    device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    config = get_config()

    # ── 找训练好的权重文件，没有就报错提示先训练 ───────────────────────
    checkpoint_path = latest_weights_file_path(config)
    if checkpoint_path is None:
        raise FileNotFoundError("没有找到训练好的权重，请先运行 train.py")

    # ── 加载模型 + 权重 ─────────────────────────────────────────────
    model = build_cnn(config).to(device)  # 先按相同配置建一个空模型
    # 把 checkpoint 里的权重填进模型；map_location=device 允许权重在 CPU 上加载
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()  # 评估模式：关闭 Dropout（推理时不需要随机丢弃）

    # ── 取测试集的一个 batch 做推理 ─────────────────────────────────
    _, test_loader = get_ds(config)
    images, labels = next(iter(test_loader))  # next(iter(...)) 取出第一个 batch

    with torch.no_grad():  # 推理不需要梯度，省内存
        # model(...) -> (batch, 10) logits
        # argmax(dim=1): 每个样本取分数最高的类 -> 预测类别
        # .cpu(): 结果搬回 CPU，方便和 labels 比较、画图
        preds = model(images.to(device)).argmax(dim=1).cpu()

    # ── 反归一化，把 (0,1) 之间的像素还原成可显示的图片 ─────────────────
    # 训练时做了 Normalize，像素被标准化到 0 附近（有负数），
    # 显示前要乘 std 加 mean 还原回 [0,1]；view(1,3,1,1) 是凑成可广播的形状
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)

    # ── 展示前 10 张图片，绿色 = 预测正确，红色 = 预测错误 ──────────────
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # 2 行 5 列子图
    for i, ax in enumerate(axes.flat):  # axes.flat 把 2x5 的子图展平成 1 维
        # 反归一化 + clamp(0,1) 截断越界值 + permute(1,2,0) 把 (C,H,W) 变 (H,W,C) 给 matplotlib
        img = (images[i] * std[0] + mean[0]).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        color = "green" if preds[i] == labels[i] else "red"  # 对错决定标题颜色
        # CIFAR-10 显示中文类别名，CIFAR-100 类别太多就显示数字
        true_name = CIFAR10_NAMES[labels[i]] if config["datasource"] == "cifar10" else str(labels[i].item())
        pred_name = CIFAR10_NAMES[preds[i]] if config["datasource"] == "cifar10" else str(preds[i].item())
        ax.set_title(f"真:{true_name}\n预:{pred_name}", color=color, fontsize=9)
        ax.axis("off")  # 不显示坐标轴刻度
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


if __name__ == "__main__":
    main()
