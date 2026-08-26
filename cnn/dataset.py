# DataLoader：负责把 Dataset 批量打包、随机打乱、多进程预取
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR 系列数据集的均值和标准差（在训练集上统计得到）
# Normalize 需要这三个数，写在模块级常量里供训练和推理共用，
# 保证"训练时的预处理"和"预测时的预处理"完全一致
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)  # 三个通道 (R, G, B) 各自的均值
CIFAR_STD = (0.2023, 0.1994, 0.2010)   # 三个通道各自的标准差


def get_num_classes(datasource):
    """根据数据集名称返回类别数：CIFAR-10 有 10 类，CIFAR-100 有 100 类。"""
    return 100 if datasource == "cifar100" else 10


def get_ds(config):
    # ── 配置自检 ─────────────────────────────────────────────────────
    # 防止把 num_classes 配错（比如选了 cifar100 却配成 10），
    # 配置不一致直接报错，比训到一半才发现好
    expected = get_num_classes(config["datasource"])
    assert config["num_classes"] == expected, \
        f"数据集 {config['datasource']} 需要 num_classes={expected}，当前配置为 {config['num_classes']}"

    # ── 训练集预处理: 数据增强 + 归一化 ────────────────────────────────
    # 数据增强 = 每次取图时随机做点"变形"，相当于免费扩大了训练集，
    # 让模型对不同位置/翻转的物体都鲁棒，缓解过拟合
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # 先四周补 4 像素，再随机裁回 32x32
        transforms.RandomHorizontalFlip(),      # 以 50% 概率水平翻转（对物体分类很有效）
        transforms.ToTensor(),                  # PIL -> Tensor，像素值缩放到 [0,1]
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  # 按 CIFAR 统计量标准化
    ])
    # ── 测试集预处理: 只归一化，不做增强 ────────────────────────────────
    # 测试时不能随机变形，否则每次评估结果都不一样；只用归一化保证稳定
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # ── 选择数据集类 ─────────────────────────────────────────────────
    if config["datasource"] == "cifar10":
        ds_cls = datasets.CIFAR10
    elif config["datasource"] == "cifar100":
        ds_cls = datasets.CIFAR100
    else:
        raise ValueError(f"未知数据集: {config['datasource']}")

    # train=True 拿训练集（5 万张），train=False 拿测试集（1 万张）
    # download=True：本地没有就自动下载；transform：下载后每张图先过预处理
    train_ds = ds_cls(root=config["data_root"], train=True, download=True, transform=train_transform)
    test_ds = ds_cls(root=config["data_root"], train=False, download=True, transform=test_transform)

    # ── 包成 DataLoader ──────────────────────────────────────────────
    # shuffle=True：每个 epoch 打乱顺序，避免模型学到样本顺序的假规律（仅训练集）
    # shuffle=False：测试不打乱，评估结果稳定
    # num_workers：用几个子进程并行预取数据，加快喂数据速度
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # 注意: 这版 CNN 没有切验证集，直接用测试集观察训练效果
    return train_loader, test_loader
