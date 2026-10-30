"""Muon Newton-Schulz orthogonalization task."""

TASK = {
    "title": "Muon Newton-Schulz Matrix Orthogonalization",
    "title_zh": "Muon 优化器 Newton-Schulz 矩阵正交化",
    "difficulty": "Hard",
    "category": "优化器与学习率",
    "function_name": "muon_orthogonalize",
    "starter_code": '''import torch

def muon_orthogonalize(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    pass
''',
    "hint": "Normalize first. Transpose a tall matrix so the Gram matrix uses the smaller axis, iterate the cubic polynomial, then restore the orientation.",
    "hint_zh": "先归一化。将高矩阵转置，使 Gram 矩阵使用较小维度；迭代三次多项式后恢复方向。",
    "description_en": "Implement Muon's Newton-Schulz-style matrix orthogonalization for a 2D tensor. Normalize by the Frobenius norm, orient the matrix to minimize the Gram matrix, and apply the given polynomial iteration.",
    "description_zh": "为二维张量实现 Muon 的 Newton-Schulz 风格矩阵正交化。先按 Frobenius 范数归一化，调整矩阵方向以减小 Gram 矩阵，再执行给定多项式迭代。",
    "theory_en": "The iteration applies a polynomial in `XX^T` to `X`. Its coefficients rapidly push singular values toward one while matrix multiplication preserves singular vectors.",
    "theory_zh": "该迭代把 `XX^T` 的多项式作用于 `X`。系数会快速将奇异值推向 1，而矩阵乘法保持奇异向量。",
    "diagram_en": "```mermaid\nflowchart LR\nG[G] --> N[Frobenius normalize]\nN --> T[orient smaller axis]\nT --> A[A=XX^T]\nA --> I[X=aX+(bA+cA^2)X]\nI --> A\nI --> R[restore orientation]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nG[G] --> N[Frobenius 归一化]\nN --> T[调整较小轴]\nT --> A[A=XX^T]\nA --> I[X=aX+(bA+cA^2)X]\nI --> A\nI --> R[恢复方向]\n```",
    "tests": [
        {"name": "exact polynomial reference", "code": """import torch
torch.manual_seed(0)
G = torch.randn(3, 5)
out = {fn}(G, steps=2)
X = G / (G.norm() + 1e-7)
for _ in range(2):
    A = X @ X.T
    B = -4.7750 * A + 2.0315 * (A @ A)
    X = 3.4445 * X + B @ X
assert out.shape == G.shape
assert torch.allclose(out, X, atol=1e-7, rtol=1e-6)
"""},
        {"name": "tall matrix transposes internally", "code": """import torch
torch.manual_seed(1)
G = torch.randn(7, 2)
out = {fn}(G, steps=1)
X = (G / (G.norm() + 1e-7)).T
A = X @ X.T
expected = (3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X).T
assert out.shape == (7, 2)
assert torch.allclose(out, expected)
"""},
        {"name": "zero steps only normalizes", "code": """import torch
G = torch.tensor([[3.0, 4.0]])
out = {fn}(G, steps=0, eps=0.0)
assert out.shape == G.shape
assert torch.equal(out, torch.tensor([[0.6, 0.8]]))
"""},
        {"name": "gradient flows", "code": """import torch
G = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
out = {fn}(G, steps=2)
out.square().sum().backward()
assert out.shape == G.shape
assert G.grad is not None and G.grad.shape == G.shape
assert torch.isfinite(G.grad).all()
"""},
    ],
    "solution": '''import torch

def muon_orthogonalize(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    a, b, c = coeffs
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X
''',
    "demo": '''G = torch.randn(8, 4)
X = muon_orthogonalize(G)
print(X.shape, X.T @ X)
''',
    "explanation": "Transposing tall inputs reduces the Gram matrix from M-by-M to N-by-N. All operations remain differentiable, and transposing back preserves the input shape.",
}
