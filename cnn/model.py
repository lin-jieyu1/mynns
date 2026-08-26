import torch.nn as nn


class CNN(nn.Module):
    """用于图像分类的小型卷积神经网络。

    结构: [Conv3x3 + BN + ReLU + MaxPool] x 3 -> AdaptiveAvgPool -> Linear

    CIFAR-10 输入是 (3, 32, 32)，每过一个 block 特征图边长减半：
        32 -> 16 -> 8 -> 4
    通道数逐渐增加: 3 -> 32 -> 64 -> 128，对应"信息不断抽象、空间不断压缩"。

    最后用 AdaptiveAvgPool2d(1) 把任意尺寸的特征图压成 1x1，
    这样即使换更大/更小的输入图片也能接上全连接层。
    """

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # ── 特征提取部分: 从原始像素里提取越来越抽象的特征 ──────────────
        # 每个 block: Conv(提取局部特征) -> BN(稳定训练) -> ReLU(非线性) -> MaxPool(压缩尺寸)
        self.features = nn.Sequential(
            # Block 1: (3, 32, 32) -> (32, 32, 32) -> MaxPool -> (32, 16, 16)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            # 3x3 卷积, padding=1 保证输出尺寸不变；32 个卷积核 = 输出 32 个通道
            nn.BatchNorm2d(32),            # 批归一化：加速收敛、缓解过拟合
            nn.ReLU(inplace=True),         # 激活函数，引入非线性
            nn.MaxPool2d(2),               # 下采样，边长减半（2x2 窗口取最大值）

            # Block 2: (32, 16, 16) -> (64, 16, 16) -> MaxPool -> (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (64, 8, 8) -> (128, 8, 8) -> MaxPool -> (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # ── 分类部分: 把特征图压缩成"每一类的分数" ──────────────────────
        self.classifier = nn.Sequential(
            # 全局平均池化: (batch, 128, h, w) -> (batch, 128, 1, 1)
            # 不管 h,w 是多少，都压成 1x1，所以输入尺寸可以随便换
            nn.AdaptiveAvgPool2d(1),       # (batch, 128, h, w) -> (batch, 128, 1, 1)
            nn.Flatten(),                  # 拉平成 (batch, 128)
            nn.Dropout(0.3),               # 随机丢弃 30%，防止过拟合
            nn.Linear(128, num_classes),   # 128 个特征 -> 10 类分数（logits）
            # 注意: 不加激活函数，因为 CrossEntropyLoss 自带 Softmax
        )

    def forward(self, x):
        # x: (batch, in_channels, h, w)，例如 (batch, 3, 32, 32)
        # 先过特征提取，再过分类器，一条直线串下来
        return self.classifier(self.features(x))


def build_cnn(config):
    """根据配置字典构建 CNN。"""
    return CNN(
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
    )
