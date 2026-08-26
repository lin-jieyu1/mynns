from pathlib import Path

def get_config():
    return {
        # ===== 问题定义（1D 泊松方程）=====
        "domain": [0.0, 1.0],      # x ∈ [0, 1]
        "left_bc": 0.0,            # u(0) = 0
        "right_bc": 0.0,           # u(1) = 0
        "datasource": "poisson1d",

        # ===== 采样点数 =====
        "n_collocation": 2048,     # 每步重新采样的配点（PDE 残差点）
        "n_boundary": 64,          # 边界点

        # ===== 训练超参数 =====
        "num_epochs": 3000,
        "lr": 10**-3,
        "lambda_pde": 1.0,         # PDE 残差损失的权重
        "lambda_bc": 10.0,         # 边界条件损失的权重

        # ===== 模型结构 =====
        "input_dim": 1,
        "hidden_dims": [64, 64, 64, 64],
        "output_dim": 1,
        "activation": "tanh",      # PINN 推荐 tanh / sine，ReLU 二阶导恒为 0 不适合

        # ===== 训练环境 =====
        "seed": 42,

        # ===== 权重保存 =====
        "model_folder": "weights",
        "model_basename": "pinn_",
        "preload": "latest",
    }

def get_weights_file_path(config, epoch):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
