"""PiSSA initialization task."""

TASK = {
    "title": "PiSSA (Principal Singular Value Adaptation) Initialization",
    "title_zh": "PiSSA 主奇异值低秩初始化",
    "difficulty": "Medium",
    "category": "参数高效微调",
    "function_name": "pissa_init",
    "starter_code": '''import torch

def pissa_init(W, rank, alpha=1.0):
    pass
''',
    "hint": "Split each retained singular value equally between B and A after dividing by LoRA scaling. Subtract the resulting scaled product from W.",
    "hint_zh": "先除以 LoRA 缩放，再将每个保留奇异值的平方根分别放入 B 和 A，最后从 W 中减去缩放后的乘积。",
    "description_en": "Initialize PiSSA factors from the leading singular components of a weight matrix. Return a residual base weight and low-rank factors whose scaled sum exactly reconstructs the original matrix.",
    "description_zh": "使用权重矩阵的主奇异分量初始化 PiSSA 因子。返回残差基础权重和低秩因子，使其缩放后的和精确重建原矩阵。",
    "theory_en": "The truncated SVD is the optimal rank-r approximation in Frobenius norm. PiSSA assigns that principal component to the adapter at initialization and leaves the complementary residual in the frozen base.",
    "theory_zh": "截断 SVD 是 Frobenius 范数下的最优秩 r 近似。PiSSA 在初始化时将主分量分配给适配器，并把其余残差留在冻结基础权重中。",
    "diagram_en": "```mermaid\nflowchart LR\nW[W] --> S[SVD]\nS --> P[top-r principal part]\nP --> BA[B and A factors]\nW --> R[residual Wres]\nBA --> C[Wres+sBA=W]\nR --> C\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nW[W] --> S[SVD]\nS --> P[前 r 个主分量]\nP --> BA[B 与 A 因子]\nW --> R[残差 Wres]\nBA --> C[Wres+sBA=W]\nR --> C\n```",
    "tests": [
        {"name": "shapes and reconstruction", "code": """import torch
torch.manual_seed(0)
W = torch.randn(6, 4)
W_res, B, A = {fn}(W, rank=2, alpha=4.0)
assert W_res.shape == W.shape and B.shape == (6, 2) and A.shape == (2, 4)
reconstructed = W_res + 2.0 * (B @ A)
assert torch.allclose(reconstructed, W, atol=1e-6, rtol=1e-6)
"""},
        {"name": "principal product exact reference", "code": """import torch
torch.manual_seed(1)
W = torch.randn(5, 3, dtype=torch.double)
W_res, B, A = {fn}(W, rank=2, alpha=1.0)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
s = 0.5
expected = (U[:, :2] * S[:2].unsqueeze(0)) @ Vh[:2]
assert B.shape == (5, 2) and A.shape == (2, 3)
assert torch.allclose(s * (B @ A), expected, atol=1e-10)
assert torch.allclose(W_res, W - expected, atol=1e-10)
"""},
        {"name": "full rank leaves zero residual", "code": """import torch
W = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
W_res, B, A = {fn}(W, rank=3, alpha=3.0)
assert B.shape == (3, 3) and A.shape == (3, 3)
assert torch.allclose(W_res, torch.zeros_like(W), atol=1e-6)
assert torch.allclose(B @ A, W, atol=1e-6)
"""},
        {"name": "gradient flow", "code": """import torch
torch.manual_seed(2)
W = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
W_res, B, A = {fn}(W, rank=2, alpha=2.0)
loss = W_res.square().sum() + B.square().sum() + A.square().sum()
loss.backward()
assert W_res.shape == W.shape
assert W.grad is not None and W.grad.shape == W.shape
assert torch.isfinite(W.grad).all()
"""},
    ],
    "solution": '''import torch

def pissa_init(W, rank, alpha=1.0):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    s = alpha / rank
    sqrt_S = torch.sqrt(S_r / s)
    B = U_r * sqrt_S.unsqueeze(0)
    A = sqrt_S.unsqueeze(1) * Vh_r
    W_res = W - s * (B @ A)
    return W_res, B, A
''',
    "demo": '''W = torch.randn(8, 6)
W_res, B, A = pissa_init(W, rank=2, alpha=2.0)
print(torch.max(torch.abs(W_res + B @ A - W)))
''',
    "explanation": "Because each factor receives `sqrt(S_r / s)`, their product equals the principal rank-r component divided by `s`. Multiplication by the LoRA scaling therefore restores the truncated SVD component.",
}
