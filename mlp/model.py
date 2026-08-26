import torch.nn as nn


class MLP(nn.Module):
    """多层感知机（全连接神经网络）。

    结构: 输入 -> [Linear + ReLU + Dropout] * len(hidden_dims) -> Linear(num_classes)

    Fashion-MNIST 的输入是 28x28 灰度图，所以 input_dim = 28 * 28 = 784，
    每一张图片被拉平成 784 维向量再送入网络。
    """

    def __init__(self, input_dim, hidden_dims : list[int], num_classes, dropout=0.2):
        super().__init__()
        layers = []  # 先存到普通 list，最后展开塞进 Sequential（详见后面注释）
        dims = [input_dim] + hidden_dims  # 所有线性层的输入输出维度链
        # 例如 dims = [784, 512, 256, 128]，两两配对形成 3 个 Linear
        # zip 是Python内置函数，用于将两个列表合并成一个元组列表
        # 例如：zip([1, 2, 3], [4, 5, 6]) -> [(1, 4), (2, 5), (3, 6)]
        for in_features, out_features in zip(dims[:-1], dims[1:]): # 遍历每一层
            # 每一层都添加一个线性层，一个ReLU激活函数，一个Dropout层
            layers.append(nn.Linear(in_features, out_features))  # 线性变换: x @ W^T + b
            layers.append(nn.ReLU(inplace=True))                 # 激活函数: 引入非线性，否则多层线性叠加还是线性
            layers.append(nn.Dropout(dropout))                   # 随机丢弃部分神经元，防止过拟合
        # 最后一层不加激活函数，分类任务交给 CrossEntropyLoss 处理
        layers.append(nn.Linear(dims[-1], num_classes))
        # nn.Sequential(*layers): 把 list 里的层按顺序串成一条"流水线"，
        # forward 时自动前一层输出喂给后一层。这就是"更大的模型"。
        # 为什么不能直接 self.layers = layers？因为普通 list 里的模块不会被
        # PyTorch 注册（parameters()/.to(device)/state_dict() 都找不到），
        # 必须放进 Sequential / ModuleList 或作为属性，参数才算数。
        self.net = nn.Sequential(*layers) # 将所有层打包成一个Sequential模型

    def forward(self, x):
        # x: (batch, 1, 28, 28) -> (batch, input_dim)
        # 这里的1是channel数，28是高度，28是宽度
        # (Batch, Channel, Height, Width) -> (Batch, Height * Width)
        x = x.view(x.size(0), -1)  # 把每张 28x28 的图拉平成 784 维向量
        # 这里的-1是自动计算维度，例如x.size(0)是batch_size，x.size(1)是input_dim
        # 例如：x.size(0) = 100, x.size(1) = 784, x.view(x.size(0), -1) -> (100, 784)
        return self.net(x)  # (batch, num_classes) 每个类别的原始分数（logits）


def build_mlp(config):
    """根据配置字典构建 MLP。"""
    return MLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
