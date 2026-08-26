# torch：加载模型、推理
import torch
# matplotlib：把图片和预测结果画出来
import matplotlib.pyplot as plt

from config import get_config, latest_weights_file_path
from dataset import get_ds
from model import build_mlp


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
    model = build_mlp(config).to(device)  # 先按相同配置建一个空模型
    # 把 checkpoint 里的权重填进模型；map_location=device 允许权重在 CPU 上加载
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()  # 评估模式：关闭 Dropout（推理时不需要随机丢弃）

    # ── 取测试集的一个 batch 做推理 ─────────────────────────────────
    _, _, test_loader = get_ds(config)
    images, labels = next(iter(test_loader))  # next(iter(...)) 取出第一个 batch

    with torch.no_grad():  # 推理不需要梯度，省内存
        # model(...) -> (batch, 10) logits
        # argmax(dim=1): 每个样本取分数最高的类 -> 预测类别
        # .cpu(): 结果搬回 CPU，方便和 labels 比较、画图
        preds = model(images.to(device)).argmax(dim=1).cpu()

    # ── 展示前 10 张图片，绿色 = 预测正确，红色 = 预测错误 ──────────────
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))  # 2 行 5 列子图
    for i, ax in enumerate(axes.flat):  # axes.flat 把 2x5 的子图展平成 1 维
        ax.imshow(images[i].squeeze(), cmap="gray")  # squeeze() 去掉 (1,28,28) 里的通道维度 -> (28,28)
        color = "green" if preds[i] == labels[i] else "red"  # 对错决定标题颜色
        ax.set_title(f"true:{labels[i].item()} pred:{preds[i].item()}", color=color)
        ax.axis("off")  # 不显示坐标轴刻度
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig("predict.png")  # 远程环境没有图形窗口，保存成文件再看
    plt.show()


if __name__ == "__main__":
    main()
