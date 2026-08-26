# DataLoader：负责把 Dataset 批量打包、随机打乱、多进程预取
# random_split：把数据集按比例随机切成多份（如训练集/验证集）
from torch.utils.data import DataLoader, random_split
# torchvision 自带常用 CV 数据集（MNIST、FashionMNIST...）和图像预处理工具
from torchvision import datasets, transforms


def get_ds(config):
    """加载数据集，返回 (train_loader, val_loader, test_loader)。"""
    # transforms.Compose：把多个图像预处理按顺序串成一条流水线
    transform = transforms.Compose([
        # ToTensor：把 PIL 图片 / numpy 数组转成 Tensor，
        #   形状从 (H, W, C) 变 (C, H, W)，且像素值从 0~255 缩放到 0.0~1.0
        transforms.ToTensor(),
        # Normalize(mean, std)：按 (像素 - mean) / std 做标准化，
        #   (0.5,) 表示灰度图通道的均值和标准差都取 0.5，
        #   所以像素值从 [0,1] 变成 [-1,1]，让输入落在 0 附近，梯度更稳、收敛更快
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # 根据配置选择用哪个数据集类（都是 torchvision 内置的）
    if config["datasource"] == "fashion_mnist":
        ds_cls = datasets.FashionMNIST
    elif config["datasource"] == "mnist":
        ds_cls = datasets.MNIST
    else:
        raise ValueError(f"未知数据集: {config['datasource']}")

    # train=True 拿训练集（6 万张），train=False 拿测试集（1 万张）
    # download=True：本地没有就自动从网上下载
    # transform=transform：下载后每张图都会先过一遍上面的预处理
    train_ds = ds_cls(root=config["data_root"], train=True, download=True, transform=transform)
    test_ds = ds_cls(root=config["data_root"], train=False, download=True, transform=transform)

    # 从训练集切出 10% 作为验证集（只用来调超参，不参与梯度更新）
    val_size = int(len(train_ds) * 0.1)  # 60000 * 0.1 = 6000 张作验证集
    train_size = len(train_ds) - val_size  # 剩下 54000 张作真正的训练集
    # random_split 返回两个新的子数据集，互不重叠
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    # DataLoader：迭代它时每次返回一个 batch 的 (x, y)
    # shuffle=True：每个 epoch 打乱一次顺序，避免模型学到样本顺序的假规律（训练集才需要）
    # shuffle=False：验证/测试不用打乱，保证评估稳定
    # num_workers：用几个子进程并行预取数据，加快喂数据速度
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    return train_loader, val_loader, test_loader
