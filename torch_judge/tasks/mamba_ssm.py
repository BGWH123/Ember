"""Mamba SSM Step task."""

TASK = {
    'title': 'Mamba SSM Step',
    'title_zh': 'Mamba SSM 步骤',
    'difficulty': 'Hard',
    'category': '状态空间模型',
    'function_name': 'mamba_ssm_step',
    'starter_code': """def mamba_ssm_step(x, h, A, B, C, delta):
    pass""",
    'hint': """1. `dA = exp(delta.unsqueeze(-1) * A)`  shape `(B,D,N)`
2. `dB = delta.unsqueeze(-1) * B`
3. `h_new = dA * h + dB * x.unsqueeze(-1)`
4. `y = (h_new * C).sum(dim=-1)`  shape `(B,D)`""",
    'hint_zh': """1. `dA = exp(delta.unsqueeze(-1) * A)`  形状 `(B,D,N)`
2. `dB = delta.unsqueeze(-1) * B`
3. `h_new = dA * h + dB * x.unsqueeze(-1)`
4. `y = (h_new * C).sum(dim=-1)`  形状 `(B,D)`""",
    'description_en': """Implement a single recurrent step of the Mamba selective state space model (SSM).

Mamba's key innovation is that B, C, and delta are **input-dependent** (selective), unlike classical SSMs where they are fixed. This allows the model to selectively remember or forget information.

**Signature:** `mamba_ssm_step(x, h, A, B, C, delta) -> (y, h_new)`

**Parameters:**
- `x` — input at current step, shape (B, D)
- `h` — hidden state, shape (B, D, N)
- `A` — state transition, shape (D, N) — used as `A = -exp(A_log)` upstream
- `B` — input projection (selective), shape (B, D, N)
- `C` — output projection (selective), shape (B, D, N)
- `delta` — discretization step (selective, after softplus), shape (B, D)

**Discretization (Zero-Order Hold):**
- `dA = exp(delta.unsqueeze(-1) * A)` — shape (B, D, N)
- `dB = delta.unsqueeze(-1) * B` — shape (B, D, N)

**Recurrence:**
- `h_new = dA * h + dB * x.unsqueeze(-1)`
- `y = (h_new * C).sum(dim=-1)` — shape (B, D)

**Returns:** tuple `(y, h_new)`""",
    'description_zh': """实现 Mamba 选择性状态空间模型（SSM）的单步递推。

Mamba 的核心创新在于 B、C 和 delta 是**输入相关的**（选择性的），不同于经典 SSM 中的固定参数。这使模型能够选择性地记忆或遗忘信息。

**签名:** `mamba_ssm_step(x, h, A, B, C, delta) -> (y, h_new)`

**参数:**
- `x` — 当前步输入，形状 (B, D)
- `h` — 隐藏状态，形状 (B, D, N)
- `A` — 状态转移矩阵，形状 (D, N)（上游使用 `A = -exp(A_log)`）
- `B` — 输入投影（选择性），形状 (B, D, N)
- `C` — 输出投影（选择性），形状 (B, D, N)
- `delta` — 离散化步长（选择性，经过 softplus），形状 (B, D)

**离散化（零阶保持）:**
- `dA = exp(delta.unsqueeze(-1) * A)` — 形状 (B, D, N)
- `dB = delta.unsqueeze(-1) * B` — 形状 (B, D, N)

**递推:**
- `h_new = dA * h + dB * x.unsqueeze(-1)`
- `y = (h_new * C).sum(dim=-1)` — 形状 (B, D)

**返回:** 元组 `(y, h_new)`""",
    "tests": [
        {
            "name": 'Output shapes',
            "code": """import torch
B_sz, D, N = 2, 8, 4
x = torch.randn(B_sz, D)
h = torch.randn(B_sz, D, N)
A = torch.randn(D, N)
B = torch.randn(B_sz, D, N)
C = torch.randn(B_sz, D, N)
delta = torch.rand(B_sz, D) + 0.1
y, h_new = {fn}(x, h, A, B, C, delta)
assert y.shape == (B_sz, D), f'y shape: {y.shape}'
assert h_new.shape == (B_sz, D, N), f'h_new shape: {h_new.shape}'""",
        },
        {
            "name": 'h_new depends on both h and x',
            "code": """import torch
torch.manual_seed(0)
B_sz, D, N = 2, 4, 3
x = torch.randn(B_sz, D)
h = torch.randn(B_sz, D, N)
A = -torch.rand(D, N)  # negative for stability
B = torch.randn(B_sz, D, N)
C = torch.randn(B_sz, D, N)
delta = torch.rand(B_sz, D) + 0.1
_, h1 = {fn}(x, h, A, B, C, delta)
_, h2 = {fn}(x, torch.zeros_like(h), A, B, C, delta)
_, h3 = {fn}(torch.zeros_like(x), h, A, B, C, delta)
assert not torch.allclose(h1, h2), 'h_new must depend on h'
assert not torch.allclose(h1, h3), 'h_new must depend on x'""",
        },
        {
            "name": 'Gradient flows through x, B, C, delta',
            "code": """import torch
B_sz, D, N = 2, 4, 3
x = torch.randn(B_sz, D, requires_grad=True)
h = torch.randn(B_sz, D, N)
A = torch.randn(D, N)
Bt = torch.randn(B_sz, D, N, requires_grad=True)
C = torch.randn(B_sz, D, N, requires_grad=True)
delta = torch.ones(B_sz, D, requires_grad=True)  # leaf tensor, no +0.1 op
y, h_new = {fn}(x, h, A, Bt, C, delta)
(y.sum() + h_new.sum()).backward()
assert x.grad is not None, 'x.grad is None'
assert Bt.grad is not None, 'B.grad is None'
assert C.grad is not None, 'C.grad is None'
assert delta.grad is not None, 'delta.grad is None'""",
        },
        {
            "name": 'With A=-inf, h_new equals dB * x (no memory)',
            "code": """import torch
torch.manual_seed(1)
B_sz, D, N = 1, 4, 3
x = torch.randn(B_sz, D)
h = torch.randn(B_sz, D, N)
A = torch.full((D, N), -1e9)  # exp(-inf * delta) ~ 0
B = torch.randn(B_sz, D, N)
C = torch.randn(B_sz, D, N)
delta = torch.rand(B_sz, D) + 0.1
_, h_new = {fn}(x, h, A, B, C, delta)
expected = delta.unsqueeze(-1) * B * x.unsqueeze(-1)
assert torch.allclose(h_new, expected, atol=1e-4), 'With A=-inf, h_new should be dB*x only'""",
        },
        {
            "name": 'Numerical correctness of dA formula',
            "code": """import torch
torch.manual_seed(3)
B_sz, D, N = 2, 4, 3
x = torch.randn(B_sz, D)
h = torch.zeros(B_sz, D, N)
A = -torch.rand(D, N)
B = torch.randn(B_sz, D, N)
C = torch.randn(B_sz, D, N)
delta = torch.rand(B_sz, D) + 0.1
_, h_new = {fn}(x, h, A, B, C, delta)
dA_ref = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
dB_ref = delta.unsqueeze(-1) * B
expected_h = dA_ref * h + dB_ref * x.unsqueeze(-1)
assert torch.allclose(h_new, expected_h, atol=1e-5), 'h_new does not match ZOH formula'""",
        },
    ],
    'solution': '''def mamba_ssm_step(x, h, A, B, C, delta):
    """执行 Mamba 选择性状态空间单步递推 (Selective SSM Step)。"""
    # Step 1: 将连续状态矩阵与输入映射离散化
    delta_expanded = delta.unsqueeze(-1)
    discrete_A = torch.exp(delta_expanded * A.unsqueeze(0))
    discrete_B = delta_expanded * B

    # Step 2: 融合历史状态与当前输入更新隐藏状态
    h_new = discrete_A * h + discrete_B * x.unsqueeze(-1)

    # Step 3: 用选择性输出投影读出当前结果
    y = (h_new * C).sum(dim=-1)
    return y, h_new''',
    'demo': """torch.manual_seed(42)
B_size, D, N = 2, 8, 4
x = torch.randn(B_size, D)
h = torch.zeros(B_size, D, N)
A = -torch.rand(D, N)
B = torch.randn(B_size, D, N)
C = torch.randn(B_size, D, N)
delta = torch.rand(B_size, D) + 0.1
y, h_new = mamba_ssm_step(x, h, A, B, C, delta)
print("Output shape:", y.shape)
print("State shape:", h_new.shape)
print("Sample values:", y[0, :4])""",
    'explanation': r"""### 核心思路
Mamba 单步先按 `delta` 离散化连续状态参数，得到 $\bar A=e^{\Delta A}$ 与 $\bar B=\Delta B$；随后递推 `h_new` 并通过输入相关的 `C` 求和读出 `(B, D)` 输出。

### 代码步骤拆解
- **Step 1（离散化）**：扩展 `delta` 到 `(B, D, 1)`，广播生成 `(B, D, N)` 参数。
- **Step 2（状态更新）**：将衰减历史 `discrete_A * h` 与输入注入项相加。
- **Step 3（选择性读出）**：对 `h_new * C` 的状态维求和得到 `y`。

### 重点与易错点分析
- **注意点 1（广播对齐）**：`A` 是 `(D, N)`，显式 `unsqueeze(0)` 可清晰对齐批次维。
- **注意点 2（稳定性）**：上游通常令 `A` 为负数且 `delta` 为正数，使 `exp(delta * A)` 不超过 1。
- **注意点 3（梯度路径）**：`B`、`C`、`delta` 都是输入相关选择参数，不能被 `.detach()`。

### 复杂度分析
- **时间复杂度**：$O(BDN)$，离散化、状态更新与读出均逐元素。
- **空间复杂度**：$O(BDN)$，主要保存离散参数和新状态。""",
}
