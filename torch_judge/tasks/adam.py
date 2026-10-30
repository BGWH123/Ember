"""Adam Optimizer task."""

TASK = {
    'title': 'Adam Optimizer',
    'title_zh': 'Adam 优化器',
    'difficulty': 'Medium',
    'category': '优化器与学习率',
    'function_name': 'MyAdam',
    'starter_code': """class MyAdam:
    pass""",
    'hint': """1. m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²
2. Bias-correct: m̂ = m/(1-β1ᵗ),  v̂ = v/(1-β2ᵗ)
3. p -= lr · m̂ / (√v̂ + ε)""",
    'hint_zh': """1. m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²
2. 偏差校正：m̂ = m/(1-β1ᵗ),  v̂ = v/(1-β2ᵗ)
3. p -= lr · m̂ / (√v̂ + ε)""",
    'description_en': """Implement the Adam optimizer from scratch.

Adam combines momentum (1st moment) and RMSProp (2nd moment) with bias correction for adaptive per-parameter learning rates.

**Signature:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`

**Methods:**
- `step()` — update parameters using bias-corrected moments
- `zero_grad()` — zero all parameter gradients

**Constraints:**
- Must match `torch.optim.Adam` numerically
- Bias correction: `m_hat = m / (1 - beta1^t)`""",
    'description_zh': """从零实现 Adam 优化器。

Adam 结合了动量（一阶矩）和 RMSProp（二阶矩），并通过偏差校正实现自适应的逐参数学习率。

**签名:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`

**方法:**
- `step()` — 使用偏差校正后的矩更新参数
- `zero_grad()` — 将所有参数梯度清零

**约束:**
- 必须与 `torch.optim.Adam` 数值一致
- 偏差校正: `m_hat = m / (1 - beta1^t)`""",
    'theory_en': r"""Adam combines momentum (first moment) and RMSProp (second moment) with bias correction.

**Update Rules:**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Hyperparameters:**
- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.999$ (second moment decay)
- $\epsilon = 10^{-8}$ (numerical stability)

Bias correction is critical in early steps when $m_t$ and $v_t$ are biased toward zero.""",
    'theory_zh': r"""Adam 结合了动量（一阶矩）和 RMSProp（二阶矩），并进行偏差校正。

**更新规则：**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**超参数：**
- $\beta_1 = 0.9$（动量衰减）
- $\beta_2 = 0.999$（二阶矩衰减）
- $\epsilon = 10^{-8}$（数值稳定性）

偏差校正在早期步骤至关重要，因为此时 $m_t$ 和 $v_t$ 偏向零。""",
    'diagram_en': """```mermaid
flowchart TD
    G[Gradient g_t] --> M[First moment<br/>m_t = beta1*m_t-1 + (1-beta1)*g_t]
    G --> V[Second moment<br/>v_t = beta2*v_t-1 + (1-beta2)*g_t^2]
    M --> BC1[Bias correct<br/>m_hat = m_t / (1-beta1^t)]
    V --> BC2[Bias correct<br/>v_hat = v_t / (1-beta2^t)]
    BC1 --> U[Update<br/>theta -= lr * m_hat / (sqrt(v_hat) + eps)]
    BC2 --> U
```""",
    'diagram_zh': """```mermaid
flowchart TD
    G[梯度 g_t] --> M[一阶矩<br/>m_t = beta1*m_t-1 + (1-beta1)*g_t]
    G --> V[二阶矩<br/>v_t = beta2*v_t-1 + (1-beta2)*g_t^2]
    M --> BC1[偏差校正<br/>m_hat = m_t / (1-beta1^t)]
    V --> BC2[偏差校正<br/>v_hat = v_t / (1-beta2^t)]
    BC1 --> U[参数更新<br/>theta -= lr * m_hat / (sqrt(v_hat) + eps)]
    BC2 --> U
```""",
    "tests": [
        {
            "name": 'Parameters change after step',
            "code": """import torch
torch.manual_seed(0)
w = torch.randn(4, 3, requires_grad=True)
opt = {fn}([w], lr=0.01)
(w ** 2).sum().backward()
w_before = w.data.clone()
opt.step()
assert not torch.equal(w.data, w_before), 'Should change after step'""",
        },
        {
            "name": 'Matches torch.optim.Adam',
            "code": """import torch
torch.manual_seed(0)
w1 = torch.randn(8, 4, requires_grad=True)
w2 = w1.data.clone().requires_grad_(True)
opt1 = {fn}([w1], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
opt2 = torch.optim.Adam([w2], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
for _ in range(5):
    (w1 ** 2).sum().backward()
    opt1.step(); opt1.zero_grad()
    (w2 ** 2).sum().backward()
    opt2.step(); opt2.zero_grad()
assert torch.allclose(w1.data, w2.data, atol=1e-5), f'Max diff: {(w1.data-w2.data).abs().max():.6f}'""",
        },
        {
            "name": 'zero_grad works',
            "code": """import torch
w = torch.randn(4, requires_grad=True)
opt = {fn}([w], lr=0.01)
(w ** 2).sum().backward()
assert w.grad.abs().sum() > 0
opt.zero_grad()
assert w.grad.abs().sum() == 0, 'zero_grad should zero all gradients'""",
        },
    ],
    'solution': '''class MyAdam:
    """自适应矩估计优化器 (Adam Optimizer)。"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.step_count = 0
        self.first_moments = [torch.zeros_like(param) for param in self.params]
        self.second_moments = [torch.zeros_like(param) for param in self.params]

    def step(self):
        # Step 1: 更新一阶矩与二阶矩
        self.step_count += 1
        with torch.no_grad():
            for index, param in enumerate(self.params):
                if param.grad is None:
                    continue
                grad = param.grad
                self.first_moments[index].mul_(self.beta_1).add_(
                    grad, alpha=1.0 - self.beta_1
                )
                self.second_moments[index].mul_(self.beta_2).addcmul_(
                    grad, grad, value=1.0 - self.beta_2
                )

                # Step 2: 校正零初始化偏差并更新参数
                first_hat = self.first_moments[index] / (
                    1.0 - self.beta_1 ** self.step_count
                )
                second_hat = self.second_moments[index] / (
                    1.0 - self.beta_2 ** self.step_count
                )
                param.sub_(self.lr * first_hat / (torch.sqrt(second_hat) + self.eps))

    def zero_grad(self):
        # Step 3: 原地清零已有梯度
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()''',
    'demo': """torch.manual_seed(42)
weights = torch.randn(4, 3, requires_grad=True)
optimizer = MyAdam([weights], lr=0.01)
loss = weights.square().sum()
loss.backward()
optimizer.step()
optimizer.zero_grad()
print("Output shape:", weights.shape)
print("Loss:", loss.item())
print("Gradient sum:", weights.grad.sum().item())""",
    'explanation': r"""### 核心思路
Adam 维护梯度的一阶、二阶指数移动平均，并用 $\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$ 构造逐元素自适应更新。

### 代码步骤拆解
- **Step 1（矩估计）**：`first_moments` 和 `second_moments` 分别累计梯度与梯度平方。
- **Step 2（偏差校正）**：除以 $1-\beta_1^t$ 和 $1-\beta_2^t$，补偿状态从零开始的偏差。
- **Step 3（梯度清理）**：`zero_grad` 原地清零已有梯度，避免下一轮反向传播累积。

### 重点与易错点分析
- **注意点 1（数值稳定）**：`eps` 位于二阶矩开方之后，与 PyTorch Adam 的默认公式一致。
- **注意点 2（无梯度参数）**：参数更新需跳过 `grad is None`，但全局步数按优化器调用次数递增以匹配测试语义。
- **注意点 3（自动微分隔离）**：参数更新必须置于 `torch.no_grad()` 中，避免把优化器步骤加入计算图。

### 复杂度分析
- **时间复杂度**：$O(P)$，每步逐元素处理全部有梯度参数。
- **空间复杂度**：$O(P)$，额外保存两份与参数等大的矩状态。""",
}
