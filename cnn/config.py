from pathlib import Path

def get_config():
    """所有超参数集中在这里，改配置不需要动训练代码。"""
    return {
        "batch_size": 128,      # 每个 batch 的样本数；CIFAR 图片小，可以比 MLP 用更大的 batch
        "num_epochs": 30,       # 训练轮数（完整过几遍训练集）
        "lr": 10**-3,           # 学习率：每次更新权重的步长
        "datasource": "cifar10",        # 数据集名称（dataset.py 里根据它选类）
        "data_root": "./data",  # 数据集下载/缓存的目录
        "model_folder": "weights",      # 权重存放子目录
        "model_basename": "cnn_",       # 权重文件名前缀
        "preload": "latest",            # 断点续训: 'latest'=最新 / 数字=指定epoch / None=从头训
        "lr_scheduler_step": 10,  # 每训练 10 个 epoch 学习率衰减一次
        "lr_scheduler_gamma": 0.1,      # 衰减系数: 学习率每次乘 0.1（降一个量级）
        "in_channels": 3,       # 输入通道数: CIFAR-10 是彩色图 (R, G, B) 三通道
        "num_classes": 10,      # 输出类别数: CIFAR-10 有 10 类物体
        "num_workers": 2,       # DataLoader 用几个子进程预取数据
        "seed": 42,             # 随机种子: 保证实验可复现
    }

def get_weights_file_path(config, epoch):
    """构造某个 epoch 的权重文件完整路径。"""
    # 目录: "cifar10_weights"；文件名: "cnn_00.pt"、"cnn_01.pt"...
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    """在权重目录里找到编号最大的（即最新的）权重文件；目录为空返回 None。"""
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"  # 通配符匹配所有 epoch 文件
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()  # 按文件名排序，"cnn_09.pt" > "cnn_08.pt"
    return str(weights_files[-1])  # 取最后一个（最新）
