"""Muon Newton-Schulz Matrix Orthogonalization task."""

TASK = {
    'title': 'Muon Newton-Schulz Matrix Orthogonalization',
    'title_zh': 'Muon 优化器 Newton-Schulz 矩阵正交化',
    'difficulty': 'Hard',
    'category': '优化器与学习率',
    'function_name': 'muon_orthogonalize',
    'starter_code': """def muon_orthogonalize(G, steps=5, coeffs=(3.4445, -4.775, 2.0315), eps=1e-07):
    pass""",
    'hint': """Normalize first. Transpose a tall matrix so the Gram matrix uses the smaller axis, iterate the cubic polynomial, then restore the orientation.""",
    'hint_zh': '先归一化。将高矩阵转置，使 Gram 矩阵使用较小维度；迭代三次多项式后恢复方向。',
    'description_en': """Implement Muon's Newton-Schulz-style matrix orthogonalization for a 2D tensor. Normalize by the Frobenius norm, orient the matrix to minimize the Gram matrix, and apply the given polynomial iteration.""",
    'description_zh': '为二维张量实现 Muon 的 Newton-Schulz 风格矩阵正交化。先按 Frobenius 范数归一化，调整矩阵方向以减小 Gram 矩阵，再执行给定多项式迭代。',
    'theory_en': """The iteration applies a polynomial in `XX^T` to `X`. Its coefficients rapidly push singular values toward one while matrix multiplication preserves singular vectors.""",
    'theory_zh': '该迭代把 `XX^T` 的多项式作用于 `X`。系数会快速将奇异值推向 1，而矩阵乘法保持奇异向量。',
    'diagram_en': """```mermaid
flowchart LR
G[G] --> N[Frobenius normalize]
N --> T[orient smaller axis]
T --> A[A=XX^T]
A --> I[X=aX+(bA+cA^2)X]
I --> A
I --> R[restore orientation]
```""",
    'diagram_zh': """```mermaid
flowchart LR
G[G] --> N[Frobenius 归一化]
N --> T[调整较小轴]
T --> A[A=XX^T]
A --> I[X=aX+(bA+cA^2)X]
I --> A
I --> R[恢复方向]
```""",
    "tests": [
        {
            "name": 'exact polynomial reference',
            "code": """import torch
torch.manual_seed(0)
G = torch.randn(3, 5)
out = {fn}(G, steps=2)
X = G / (G.norm() + 1e-7)
for _ in range(2):
    A = X @ X.T
    B = -4.7750 * A + 2.0315 * (A @ A)
    X = 3.4445 * X + B @ X
assert out.shape == G.shape
assert torch.allclose(out, X, atol=1e-7, rtol=1e-6)""",
        },
        {
            "name": 'tall matrix transposes internally',
            "code": """import torch
torch.manual_seed(1)
G = torch.randn(7, 2)
out = {fn}(G, steps=1)
X = (G / (G.norm() + 1e-7)).T
A = X @ X.T
expected = (3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X).T
assert out.shape == (7, 2)
assert torch.allclose(out, expected)""",
        },
        {
            "name": 'zero steps only normalizes',
            "code": """import torch
G = torch.tensor([[3.0, 4.0]])
out = {fn}(G, steps=0, eps=0.0)
assert out.shape == G.shape
assert torch.equal(out, torch.tensor([[0.6, 0.8]]))""",
        },
        {
            "name": 'gradient flows',
            "code": """import torch
G = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
out = {fn}(G, steps=2)
out.square().sum().backward()
assert out.shape == G.shape
assert G.grad is not None and G.grad.shape == G.shape
assert torch.isfinite(G.grad).all()""",
        },
    ],
    'solution': '''def muon_orthogonalize(G, steps=5, coeffs=(3.4445, -4.775, 2.0315), eps=1e-07):
    """Muon 优化器 Newton-Schulz 矩阵正交化 (Muon Newton-Schulz Matrix Orthogonalization)。"""
    # Step 1: 沿矩阵 Frobenius 范数做初始归一化，并把较短轴置于行维
    orthogonalized = G / (G.norm() + eps)
    transposed = orthogonalized.shape[0] > orthogonalized.shape[1]
    if transposed:
        orthogonalized = orthogonalized.T
    coeff_a, coeff_b, coeff_c = coeffs
    # Step 2: 迭代应用 Newton-Schulz 五次奇多项式逼近矩阵的正交极因子
    for _ in range(steps):
        gram = orthogonalized @ orthogonalized.T
        polynomial = coeff_b * gram + coeff_c * (gram @ gram)
        orthogonalized = coeff_a * orthogonalized + polynomial @ orthogonalized
    # Step 3: 若内部为降低 Gram 代价做过转置，则恢复输入矩阵的原始方向
    if transposed:
        orthogonalized = orthogonalized.T
    return orthogonalized''',
    'demo': """torch.manual_seed(42)
G = torch.randn(8, 4)
X = muon_orthogonalize(G)
print('输入/输出形状:', tuple(G.shape), tuple(X.shape))
print('列 Gram 对角线:', (X.T @ X).diag().tolist())
print('列 Gram 最大非对角量:', f'{((X.T @ X) - torch.diag((X.T @ X).diag())).abs().max().item():.4f}')""",
    'explanation': r"""### 核心思路
Muon 不用显式 SVD，而以多项式迭代近似 $G$ 的正交极因子 $UV^\top$。先令 $X=G/(\|G\|_F+\varepsilon)$ 保证奇异值处于迭代有效区间；每轮取 $A=XX^\top$ 并更新 $X\leftarrow aX+(bA+cA^2)X$。这等价于把标量五次奇多项式 $a\sigma+b\sigma^3+c\sigma^5$ 作用于每个奇异值，而保留奇异向量。

### 代码步骤拆解
- **Step 1（归一化与定向）**：`G.norm()` 是 Frobenius 范数；若原矩阵高于宽，转置后行数 $r=\min(M,N)$，使 `gram` 只需构造 $r\times r$。
- **Step 2（多项式迭代）**：每轮由 `gram=X@X.T` 计算 $bA+cA^2$，再左乘 X；默认五轮快速把非零奇异值推向约 1。
- **Step 3（恢复布局）**：`transposed` 只改变内部计算方向，最后转置回来，保证返回形状与 `G` 完全一致。

### 重点与易错点分析
- **注意点 1（不是逐列归一化）**：初始尺度使用整个矩阵的 $\|G\|_F$，不是行范数、列范数或谱范数；`steps=0` 应只返回这一全局归一化结果。
- **注意点 2（Gram 方向决定成本）**：始终计算 `X @ X.T`，所以必须保证内部行数不大于列数；否则高瘦矩阵会产生昂贵的 $M\times M$ Gram。
- **注意点 3（近似而非精确 QR）**：给定系数优先追求有限步内的奇异值整平，输出不一定达到机器精度正交；与 QR 不同，它由矩阵乘法组成，适合加速器且全程可微。

### 复杂度分析
- **时间复杂度**：令 $r=\min(M,N),c=\max(M,N)$，T 轮成本为 $O(T(r^2c+r^3))=O(Tr^2c)$，瓶颈是 Gram 与多项式乘 X。
- **空间复杂度**：前向额外保存 `gram`、`polynomial` 各 $r\times r$，为 $O(r^2)$；输出矩阵为 $O(MN)$。""",
}
