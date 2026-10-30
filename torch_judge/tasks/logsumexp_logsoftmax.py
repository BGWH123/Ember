"""Numerically Stable LogSumExp & Log-Softmax task."""

TASK = {
    'title': 'Numerically Stable LogSumExp & Log-Softmax',
    'title_zh': '数值稳定 LogSumExp 与 Log-Softmax',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'logsumexp_logsoftmax',
    'starter_code': """def logsumexp_logsoftmax(x, dim=-1):
    pass""",
    'hint': """Subtract the maximum with `keepdim=True` before exponentiation; retain that dimension for log-probabilities and squeeze it only for LogSumExp.""",
    'hint_zh': '指数运算前减去 `keepdim=True` 的最大值；计算对数概率时保留该维，仅对 LogSumExp 结果压缩该维。',
    'description_en': 'Implement numerically stable LogSumExp and log-softmax together using the max-subtraction trick.',
    'description_zh': '使用减最大值技巧，同时实现数值稳定的 LogSumExp 和 log-softmax。',
    'theory_en': """Because `log(sum(exp(x)))` is shift-equivariant, subtracting the largest entry prevents overflow. Adding the maximum back recovers the exact result, while `x-lse` yields log-softmax.""",
    'theory_zh': '由于 `log(sum(exp(x)))` 对平移具有等变性，先减最大值可避免溢出；随后加回最大值得到原结果，而 `x-lse` 即 log-softmax。',
    'diagram_en': """```text
x -> max m -> exp(x-m) -> sum -> log -> +m -> lse
x --------------------------------------------- lse -> x-lse -> log_probs
```""",
    'diagram_zh': """```text
x -> 最大值 m -> exp(x-m) -> 求和 -> log -> +m -> lse
x ---------------------------------------------- lse -> x-lse -> log_probs
```""",
    "tests": [
        {
            "name": 'output shapes',
            "code": """import torch
x = torch.randn(2, 3, 4)
lse, logp = {fn}(x, dim=1)
assert lse.shape == (2, 4) and logp.shape == x.shape""",
        },
        {
            "name": 'matches reference',
            "code": """import torch
torch.manual_seed(3)
x = torch.randn(2, 5, 7, dtype=torch.float64)
lse, logp = {fn}(x, -1)
assert torch.allclose(lse, torch.logsumexp(x, -1), atol=1e-12)
assert torch.allclose(logp, torch.log_softmax(x, -1), atol=1e-12)""",
        },
        {
            "name": 'extreme shift stability',
            "code": """import torch
x = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
lse, logp = {fn}(x)
assert torch.isfinite(lse).all() and torch.isfinite(logp).all()
assert torch.allclose(logp.exp().sum(-1), torch.ones(2), atol=2e-5)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(3, 6, requires_grad=True)
lse, logp = {fn}(x, 1)
(lse.sum() + logp.square().mean()).backward()
assert x.grad is not None and torch.isfinite(x.grad).all()""",
        },
    ],
    'solution': '''def logsumexp_logsoftmax(x, dim=-1):
    """联合计算稳定 LogSumExp 与 Log-Softmax。"""
    # Step 1: 保留归约维并减去该维最大值
    row_max = x.max(dim=dim, keepdim=True).values
    shifted = x - row_max

    # Step 2: 在安全范围内计算 LogSumExp
    lse_keepdim = row_max + torch.log(
        torch.exp(shifted).sum(dim=dim, keepdim=True)
    )

    # Step 3: 广播得到对数概率并压缩 LSE 归约维
    log_probs = x - lse_keepdim
    lse = lse_keepdim.squeeze(dim)
    return lse, log_probs''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 5) + 1000.0
lse, log_probs = logsumexp_logsoftmax(x)
print("LogSumExp shape:", lse.shape)
print("Log-probability shape:", log_probs.shape)
print("Probability sums:", log_probs.exp().sum(dim=-1))""",
    'explanation': r"""### 核心思路
利用 $\log\sum_i e^{x_i}=m+\log\sum_i e^{x_i-m}$，其中 `m` 是归约维最大值，避免指数溢出；`x - lse_keepdim` 同时得到 log-softmax。

### 代码步骤拆解
- **Step 1（最大值平移）**：以 `keepdim=True` 计算 `row_max` 并得到 `shifted`。
- **Step 2（对数和指数）**：对平移值求指数和，再取对数并加回最大值。
- **Step 3（双输出）**：广播计算 `log_probs`，仅对 `lse` 压缩归约维。

### 重点与易错点分析
- **注意点 1（数值稳定）**：减最大值后所有指数不超过 1，可避免大正数导致 `inf`。
- **注意点 2（维度保留）**：`lse_keepdim` 必须保留归约轴，才能与原张量正确广播。
- **注意点 3（极端负值）**：统一平移也避免所有指数同时下溢为零。

### 复杂度分析
- **时间复杂度**：$O(N)$，每个元素参与最大值、指数和求和。
- **空间复杂度**：$O(N)$，主要中间量是 `shifted` 与 `log_probs`。""",
}
