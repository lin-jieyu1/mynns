import torch

PI = torch.pi


class Poisson1D:
    """一维泊松方程：-u''(x) = f(x)，x ∈ (a, b)。

    默认取 f(x) = π² sin(πx)，配合边界条件 u(a) = u(b) = 0，
    精确解为 u(x) = sin(πx)，方便训练后验证误差。

    想换方程只需要改 residual / exact_solution / boundary 三个方法。
    """

    def __init__(self, left, right, left_bc, right_bc):
        self.left = left
        self.right = right
        self.left_bc = left_bc
        self.right_bc = right_bc

    def residual(self, model, x):
        """PDE 残差：-u''(x) - π² sin(πx)，理想情况下处处为 0。"""
        x = x.requires_grad_(True)
        u = model(x)
        # 用自动微分求一阶、二阶导
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = PI**2 * torch.sin(PI * x)
        return -u_xx - f  # (n, 1)

    def exact_solution(self, x):
        return torch.sin(PI * x)  # (n, 1)

    def sample_collocation(self, n):
        """在区间内均匀随机采样 n 个配点。"""
        x = torch.rand(n, 1) * (self.right - self.left) + self.left
        return x

    def sample_boundary(self, n):
        """左右边界各采样 n 个点，并带上边界值作为监督标签。"""
        x = torch.tensor([[self.left], [self.right]]).repeat(n, 1)
        u_true = torch.tensor([[self.left_bc], [self.right_bc]]).repeat(n, 1)
        return x, u_true
