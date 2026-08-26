import torch
import matplotlib.pyplot as plt

from config import get_config, latest_weights_file_path
from model import build_pinn
from pde import Poisson1D


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    config = get_config()

    checkpoint_path = latest_weights_file_path(config)
    if checkpoint_path is None:
        raise FileNotFoundError("没有找到训练好的权重，请先运行 train.py")

    model = build_pinn(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()

    problem = Poisson1D(
        config["domain"][0], config["domain"][1],
        config["left_bc"], config["right_bc"],
    )

    with torch.no_grad():
        x = torch.linspace(config["domain"][0], config["domain"][1], 1001, device=device).unsqueeze(1)
        u_exact = problem.exact_solution(x).cpu()
        u_pred = model(x).cpu()

    x = x.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, u_exact.numpy(), "k--", label="精确解 u(x) = sin(πx)")
    axes[0].plot(x, u_pred.numpy(), "r-", label="PINN 预测")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].legend()
    axes[0].set_title("解对比")

    axes[1].plot(x, (u_pred - u_exact).numpy().squeeze())
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("误差")
    axes[1].set_title("预测 - 精确解")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
