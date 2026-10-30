"""World Models: Symlog Transform & Dreamer Lambda-Return task."""

TASK = {
    'title': 'World Models: Symlog Transform & Dreamer Lambda-Return',
    'title_zh': '世界模型：Symlog 变换与 Dreamer Lambda 回报',
    'difficulty': 'Hard',
    'category': '世界模型',
    'function_name': 'dreamer_lambda_return',
    'starter_code': """def dreamer_lambda_return(rewards, values, continues, gamma=0.99, lam=0.95):
    pass""",
    'hint': """Initialize the final return from the final value, scan backward through the horizon, then apply the sign-preserving log and exponential elementwise.""",
    'hint_zh': '用最后一个价值初始化最终回报，沿时间范围反向递推，再逐元素应用保号对数和指数变换。',
    'description_en': """Compute finite-horizon Dreamer lambda-returns together with their symlog encoding and exact symexp reconstruction.""",
    'description_zh': '计算有限时域 Dreamer lambda 回报，并返回其 symlog 编码和精确 symexp 重建。',
    'theory_en': """Lambda-returns interpolate between one-step value targets and long rollouts. Symlog compresses large signed magnitudes smoothly, while symexp is its inverse.""",
    'theory_zh': 'Lambda 回报在一步价值目标与长时域展开之间插值。Symlog 平滑压缩大幅有符号数值，symexp 则是其逆变换。',
    'diagram_en': """```text
last value -> backward lambda recursion -> G -> symlog(G) -> symexp -> G
```""",
    'diagram_zh': """```text
末端价值 -> 反向 lambda 递推 -> G -> symlog(G) -> symexp -> G
```""",
    "tests": [
        {
            "name": 'tuple shapes',
            "code": """import torch
r = torch.randn(2, 5)
v = torch.randn(2, 5)
c = torch.ones(2, 5)
G, y, x = {fn}(r, v, c)
assert G.shape == y.shape == x.shape == (2, 5)""",
        },
        {
            "name": 'exact recursion',
            "code": """import torch
r = torch.tensor([[1., 2., 3.]])
v = torch.tensor([[.5, .6, .7]])
c = torch.tensor([[1., .5, 1.]])
G, y, x = {fn}(r, v, c, gamma=.9, lam=.8)
ref = torch.empty_like(r)
ref[:, 2] = v[:, 2]
ref[:, 1] = r[:, 1] + .9 * c[:, 1] * (.2 * v[:, 2] + .8 * ref[:, 2])
ref[:, 0] = r[:, 0] + .9 * c[:, 0] * (.2 * v[:, 1] + .8 * ref[:, 1])
assert torch.allclose(G, ref) and torch.allclose(y, torch.sign(ref) * torch.log1p(ref.abs())) and torch.allclose(x, ref)""",
        },
        {
            "name": 'horizon one edge case',
            "code": """import torch
r = torch.tensor([[99.], [-2.]])
v = torch.tensor([[3.], [-4.]])
G, y, x = {fn}(r, v, torch.ones_like(r))
assert torch.equal(G, v) and torch.allclose(x, G, atol=1e-6)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
r = torch.randn(2, 4, requires_grad=True)
v = torch.randn(2, 4, requires_grad=True)
c = torch.sigmoid(torch.randn(2, 4)).requires_grad_()
G, y, x = {fn}(r, v, c)
(G.sum() + y.sum() + x.sum()).backward()
assert r.grad is not None and v.grad is not None and c.grad is not None""",
        },
    ],
    'solution': '''def dreamer_lambda_return(rewards, values, continues, gamma=0.99, lam=0.95):
    """计算 Dreamer Lambda 回报与 Symlog 变换 (Dreamer Lambda-Return)。"""
    B, T = rewards.shape

    # Step 1: 以末端价值为 bootstrap，从后向前递推 Lambda 回报
    reversed_returns = [values[:, T - 1]]
    for t in range(T - 2, -1, -1):
        next_value = (1.0 - lam) * values[:, t + 1] + lam * reversed_returns[-1]
        current_return = rewards[:, t] + gamma * continues[:, t] * next_value
        reversed_returns.append(current_return)
    returns = torch.stack(reversed_returns[::-1], dim=1)

    # Step 2: 用保号对数压缩大幅值回报
    encoded = torch.sign(returns) * torch.log1p(returns.abs())

    # Step 3: 用保号指数重建原始回报
    reconstructed = torch.sign(encoded) * torch.expm1(encoded.abs())
    return returns, encoded, reconstructed''',
    'demo': """torch.manual_seed(42)
rewards = torch.randn(3, 15)
values = torch.randn(3, 15)
continues = torch.ones(3, 15)
returns, encoded, reconstructed = dreamer_lambda_return(rewards, values, continues)
print("Return shape:", returns.shape)
print("Max reconstruction error:", (returns - reconstructed).abs().max().item())""",
    'explanation': r"""### 核心思路
Dreamer 的有限时域 Lambda 回报反向递推 $G_t=r_t+\gamma c_t[(1-\lambda)V_{t+1}+\lambda G_{t+1}]$，再用 symlog 压缩有符号幅值并以 symexp 精确还原。

### 代码步骤拆解
- **Step 1（反向递推）**：以 `values[:, T-1]` 初始化，从 $T-2$ 到 0 生成 `(B,)` 回报并反转堆叠为 `(B, T)`。
- **Step 2（Symlog 编码）**：逐元素计算 `sign(x) * log1p(abs(x))`，在零点连续并压缩大值。
- **Step 3（Symexp 解码）**：用 `sign(y) * expm1(abs(y))` 重建回报。

### 重点与易错点分析
- **注意点 1（边界定义）**：末位置按题意直接等于末端价值，`rewards[:, T-1]` 不进入递推。
- **注意点 2（自动微分）**：列表收集后 `stack` 可保留完整梯度，避免原地写入预分配张量破坏计算图。
- **注意点 3（数值函数）**：`log1p` 和 `expm1` 在接近零时比 `log(1+x)`、`exp(x)-1` 更精确。

### 复杂度分析
- **时间复杂度**：$O(BT)$，主要开销是时间维反向扫描。
- **空间复杂度**：$O(BT)$，需保存所有回报以及编码和重建张量。""",
}
