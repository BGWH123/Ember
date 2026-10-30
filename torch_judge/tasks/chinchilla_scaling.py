"""Chinchilla Compute-Optimal Scaling Law task."""

TASK = {
    'title': 'Chinchilla Compute-Optimal Scaling Law',
    'title_zh': 'Chinchilla 算力最优缩放定律',
    'difficulty': 'Medium',
    'category': '高效训练',
    'function_name': 'chinchilla_optimal_allocation',
    'starter_code': """def chinchilla_optimal_allocation(C, alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69, k=6.0):
    pass""",
    'hint': """Convert `C` with `torch.as_tensor`, compute the constant `G`, then apply the two complementary powers whose exponents sum to one.""",
    'hint_zh': '用 `torch.as_tensor` 转换 `C`，计算常数 `G`，再应用两个指数和为 1 的互补幂。',
    'description_en': """Compute the parameter count, token count, and predicted loss at the Chinchilla compute-optimal allocation.""",
    'description_zh': '计算 Chinchilla 算力最优分配下的参数量、token 数和预测损失。',
    'theory_en': """Balancing the marginal loss reduction from parameters and data under the product constraint `C=kND` yields power-law allocations in compute.""",
    'theory_zh': '在乘积约束 `C=kND` 下平衡参数和数据带来的边际损失下降，可得到关于算力的幂律分配。',
    'diagram_en': """```text
compute C --C=kND--> balance A/N^alpha and B/D^beta -> N_opt, D_opt -> L_opt
```""",
    'diagram_zh': """```text
算力 C --C=kND--> 平衡 A/N^alpha 与 B/D^beta -> N_opt、D_opt -> L_opt
```""",
    "tests": [
        {
            "name": 'scalar tensor outputs',
            "code": """import torch
torch.tensor(0.0)  # Initialize PyTorch storage types before invoking sandboxed code.
N, D, L = {fn}(1e20)
assert N.shape == D.shape == L.shape == torch.Size([])
assert N > 0 and D > 0 and L > 1.69""",
        },
        {
            "name": 'exact formula reference',
            "code": """import torch
C = torch.tensor([1e18, 1e20], dtype=torch.float64)
N, D, L = {fn}(C)
alpha, beta, A, B, E, k = 0.34, 0.28, 406.4, 410.7, 1.69, 6.0
G = ((alpha * A) / (beta * B)) ** (1.0 / (alpha + beta))
rN = G * (C / k) ** (beta / (alpha + beta))
rD = (1.0 / G) * (C / k) ** (alpha / (alpha + beta))
rL = E + A / rN ** alpha + B / rD ** beta
assert torch.equal(N, rN) and torch.equal(D, rD) and torch.equal(L, rL)""",
        },
        {
            "name": 'compute constraint edge',
            "code": """import torch
C = torch.tensor([6.0, 60.0, 600.0])
N, D, L = {fn}(C)
assert N.shape == D.shape == L.shape == C.shape
assert torch.allclose(6.0 * N * D, C, rtol=1e-6, atol=1e-6)
assert torch.all(L[1:] < L[:-1])""",
        },
        {
            "name": 'gradient with respect to compute',
            "code": """import torch
C = torch.tensor(1e12, dtype=torch.float64, requires_grad=True)
N, D, L = {fn}(C)
L.backward()
assert N.requires_grad and D.requires_grad
assert C.grad is not None and torch.isfinite(C.grad)
assert C.grad.item() < 0""",
        },
    ],
    'solution': '''def chinchilla_optimal_allocation(C, alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69, k=6.0):
    """Chinchilla 算力最优分配 (Compute-Optimal Scaling)。"""
    # Step 1: 统一算力输入并计算最优比例常数
    C = torch.as_tensor(C)
    allocation_ratio = (
        (alpha * A) / (beta * B)
    ) ** (1.0 / (alpha + beta))

    # Step 2: 按互补幂律分配参数量与数据量
    N_opt = allocation_ratio * (C / k) ** (beta / (alpha + beta))
    D_opt = (1.0 / allocation_ratio) * (
        C / k
    ) ** (alpha / (alpha + beta))

    # Step 3: 将最优规模代回经验损失模型
    L_opt = E + A / (N_opt ** alpha) + B / (D_opt ** beta)
    return N_opt, D_opt, L_opt''',
    'demo': """torch.manual_seed(42)
compute = torch.tensor(1e23)
N_opt, D_opt, L_opt = chinchilla_optimal_allocation(compute)
print("Parameter count:", N_opt.item())
print("Token count:", D_opt.item())
print("Predicted loss:", L_opt.item())""",
    'explanation': r"""### 核心思路
在约束 $C=kND$ 下平衡参数项 $A/N^\alpha$ 与数据项 $B/D^\beta$，得到关于算力的互补幂律最优解。

### 代码步骤拆解
- **Step 1（比例常数）**：由损失系数和指数计算 `allocation_ratio`。
- **Step 2（资源分配）**：分别按 $\beta/(\alpha+\beta)$ 与 $\alpha/(\alpha+\beta)$ 次幂计算 `N_opt/D_opt`。
- **Step 3（损失预测）**：把最优规模代回经验模型得到 `L_opt`。

### 重点与易错点分析
- **注意点 1（约束一致性）**：两个幂指数之和为 1，确保 `k * N_opt * D_opt == C`。
- **注意点 2（张量转换）**：`torch.as_tensor` 同时支持标量和张量，并保留已有张量的梯度链。
- **注意点 3（输入域）**：幂律公式要求正算力；非正输入会产生无定义或非有限结果。

### 复杂度分析
- **时间复杂度**：$O(N)$，对 `C` 的每个元素执行常数次标量运算。
- **空间复杂度**：$O(N)$，保存三个同形状输出。""",
}
