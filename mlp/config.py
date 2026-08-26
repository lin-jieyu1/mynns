from pathlib import Path

def get_config():
    """所有超参数集中在这里，改配置不需要动训练代码。"""
    return {
        "batch_size": 64,       # 每个 batch 的样本数；越大越稳但越吃显存
        "num_epochs": 15,       # 训练轮数（完整过几遍训练集）
        "lr": 10**-3,           # 学习率：每次更新权重的步长
        "datasource": "fashion_mnist",  # 数据集名称（dataset.py 里根据它选类）
        "data_root": "./data",  # 数据集下载/缓存的目录
        "model_folder": "weights",      # 权重存放子目录
        "model_basename": "mlp_",       # 权重文件名前缀
        "preload": "latest",            # 断点续训: 'latest'=最新 / 数字=指定epoch / None=从头训
        "input_dim": 28 * 28,   # 输入维度: 28x28 灰度图拉平后 784
        "hidden_dims": [512, 256, 128], # 隐藏层神经元数（3 个全连接层）
        "num_classes": 10,      # 输出类别数: Fashion-MNIST 有 10 类衣服
        "dropout": 0.2,         # Dropout 概率: 每个神经元有 20% 概率被随机丢弃
        "num_workers": 2,       # DataLoader 用几个子进程预取数据
        "seed": 42,             # 随机种子: 保证实验可复现
    }

def get_weights_file_path(config, epoch):
    """构造某个 epoch 的权重文件完整路径。"""
    # 目录: "fashion_mnist_weights"；文件名: "mlp_00.pt"、"mlp_01.pt"...
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
    weights_files.sort()  # 按文件名排序，"mlp_09.pt" > "mlp_08.pt"
    return str(weights_files[-1])  # 取最后一个（最新）
