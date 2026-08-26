import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings

from config import get_config, get_weights_file_path, latest_weights_file_path
from model import build_pinn
from pde import Poisson1D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(config):
    return build_pinn(config)


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    set_seed(config["seed"])
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    problem = Poisson1D(
        config["domain"][0], config["domain"][1],
        config["left_bc"], config["right_bc"],
    )
    model = get_model(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 断点续训：如果已有权重就接着训
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        # 每次迭代重新采样配点：PINN 的数据是"无限"的
        model.train()
        x_collocation = problem.sample_collocation(config["n_collocation"]).to(device)
        x_bc, u_bc = problem.sample_boundary(config["n_boundary"])
        x_bc, u_bc = x_bc.to(device), u_bc.to(device)

        # PDE 残差损失：让网络在整个区域内满足方程
        residual = problem.residual(model, x_collocation)
        loss_pde = torch.mean(residual**2)

        # 边界条件损失：让网络在边界处满足约束
        u_pred_bc = model(x_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)

        loss = config["lambda_pde"] * loss_pde + config["lambda_bc"] * loss_bc

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # 每 500 步与精确解对比一次，观察收敛情况
        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                x_ref = torch.linspace(config["domain"][0], config["domain"][1], 1001, device=device).unsqueeze(1)
                u_exact = problem.exact_solution(x_ref)
                u_pred = model(x_ref)
                l2_err = torch.mean((u_pred - u_exact) ** 2).sqrt().item()
            print(f"  相对 L2 误差: {l2_err:.6f}")

            model_filename = get_weights_file_path(config, f"{epoch:04d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
