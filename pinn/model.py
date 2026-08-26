import torch
import torch.nn as nn


class Sine(nn.Module):
    """正弦激活函数（SIREN 论文提出），适合表示振荡类解。"""

    def forward(self, x):
        return torch.sin(x)


def get_activation(name):
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "sine":
        return Sine
    raise ValueError(f"未知激活函数: {name}")


class PINN(nn.Module):
    """物理信息神经网络：输入坐标 x，输出对解的预测 u(x)。

    结构就是 MLP，但激活函数默认用 tanh：PDE 残差要计算二阶导，
    ReLU 的二阶导恒为 0，会导致梯度无法传到 PDE 项。
    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation="tanh"):
        super().__init__()
        act = get_activation(activation)
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_features, out_features in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_features, out_features), act()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_pinn(config):
    return PINN(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        activation=config["activation"],
    )
