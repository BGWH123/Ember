"""Adam Optimizer task."""

TASK = {
    "title": "Adam Optimizer",
    "title_zh": "Adam 优化器",
    "difficulty": "Medium",
    "category": "优化器与学习率",
    "description_en": (
        "Implement the Adam optimizer from scratch.\n\nAdam combines momentum (1st moment) and RMSProp (2nd moment) with bias correction for adaptive per-parameter learning rates.\n\n**Signature:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`\n\n**Methods:**\n- `step()` — update parameters using bias-corrected moments\n- `zero_grad()` — zero all parameter gradients\n\n**Constraints:**\n- Must match `torch.optim.Adam` numerically\n- Bias correction: `m_hat = m / (1 - beta1^t)`"
    ),
    "description_zh": (
        "从零实现 Adam 优化器。\n\nAdam 结合了动量（一阶矩）和 RMSProp（二阶矩），并通过偏差校正实现自适应的逐参数学习率。\n\n**签名:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`\n\n**方法:**\n- `step()` — 使用偏差校正后的矩更新参数\n- `zero_grad()` — 将所有参数梯度清零\n\n**约束:**\n- 必须与 `torch.optim.Adam` 数值一致\n- 偏差校正: `m_hat = m / (1 - beta1^t)`"
    ),
    "function_name": "MyAdam",
    "hint": (
        "1. m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²\n2. Bias-correct: m̂ = m/(1-β1ᵗ),  v̂ = v/(1-β2ᵗ)\n3. p -= lr · m̂ / (√v̂ + ε)"
    ),
    "hint_zh": (
        "1. m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²\n2. 偏差校正：m̂ = m/(1-β1ᵗ),  v̂ = v/(1-β2ᵗ)\n3. p -= lr · m̂ / (√v̂ + ε)"
    ),
    "theory_en": (
        "Adam combines momentum (first moment) and RMSProp (second moment) with bias correction.\n\n**Update Rules:**\n$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$\n$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$\n$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$\n$$\\theta_t = \\theta_{t-1} - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$\n\n**Hyperparameters:**\n- $\\beta_1 = 0.9$ (momentum decay)\n- $\\beta_2 = 0.999$ (second moment decay)\n- $\\epsilon = 10^{-8}$ (numerical stability)\n\nBias correction is critical in early steps when $m_t$ and $v_t$ are biased toward zero."
    ),
    "theory_zh": (
        "Adam 结合了动量（一阶矩）和 RMSProp（二阶矩），并进行偏差校正。\n\n**更新规则：**\n$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$\n$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$\n$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$\n$$\\theta_t = \\theta_{t-1} - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$\n\n**超参数：**\n- $\\beta_1 = 0.9$（动量衰减）\n- $\\beta_2 = 0.999$（二阶矩衰减）\n- $\\epsilon = 10^{-8}$（数值稳定性）\n\n偏差校正在早期步骤至关重要，因为此时 $m_t$ 和 $v_t$ 偏向零。"
    ),
    "diagram_en": (
        "```mermaid\nflowchart TD\n    G[Gradient g_t] --> M[First moment<br/>m_t = beta1*m_t-1 + (1-beta1)*g_t]\n    G --> V[Second moment<br/>v_t = beta2*v_t-1 + (1-beta2)*g_t^2]\n    M --> BC1[Bias correct<br/>m_hat = m_t / (1-beta1^t)]\n    V --> BC2[Bias correct<br/>v_hat = v_t / (1-beta2^t)]\n    BC1 --> U[Update<br/>theta -= lr * m_hat / (sqrt(v_hat) + eps)]\n    BC2 --> U\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart TD\n    G[梯度 g_t] --> M[一阶矩<br/>m_t = beta1*m_t-1 + (1-beta1)*g_t]\n    G --> V[二阶矩<br/>v_t = beta2*v_t-1 + (1-beta2)*g_t^2]\n    M --> BC1[偏差校正<br/>m_hat = m_t / (1-beta1^t)]\n    V --> BC2[偏差校正<br/>v_hat = v_t / (1-beta2^t)]\n    BC1 --> U[参数更新<br/>theta -= lr * m_hat / (sqrt(v_hat) + eps)]\n    BC2 --> U\n```"
    ),
    "tests": [
        {
            "name": "Parameters change after step",
            "code": """









import torch
torch.manual_seed(0)
w = torch.randn(4, 3, requires_grad=True)
opt = {fn}([w], lr=0.01)
(w ** 2).sum().backward()
w_before = w.data.clone()
opt.step()
assert not torch.equal(w.data, w_before), 'Should change after step'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Matches torch.optim.Adam",
            "code": """









import torch
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
assert torch.allclose(w1.data, w2.data, atol=1e-5), f'Max diff: {(w1.data-w2.data).abs().max():.6f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "zero_grad works",
            "code": """









import torch
w = torch.randn(4, requires_grad=True)
opt = {fn}([w], lr=0.01)
(w ** 2).sum().backward()
assert w.grad.abs().sum() > 0
opt.zero_grad()
assert w.grad.abs().sum() == 0, 'zero_grad should zero all gradients'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


class MyAdam:
    """
    Adam Optimizer implementation.
    结合动量（一阶矩估计）和 RMSProp（二阶矩估计），通过偏差校正实现自适应学习率。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)                       # 待优化的参数列表
        self.lr = lr                                     # 学习率 η
        self.beta1, self.beta2 = betas                   # 一阶矩衰减 β1=0.9, 二阶矩衰减 β2=0.999
        self.eps = eps                                   # 数值稳定常数 ε=1e-8，防止除以零
        self.t = 0                                       # 时间步计数器，用于偏差校正

        # 初始化一阶矩 m 和二阶矩 v 为零张量
        # m_t: 梯度的一阶矩（动量），v_t: 梯度的二阶矩（RMSProp）
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        """
        执行一步参数更新。
        更新公式:
            m_t = β1·m_{t-1} + (1-β1)·g_t
            v_t = β2·v_{t-1} + (1-β2)·g_t²
            m̂_t = m_t / (1-β1^t)   ← 偏差校正，早期步骤 m_t 偏向零
            v̂_t = v_t / (1-β2^t)   ← 偏差校正
            θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
        """
        self.t += 1                                      # 更新时间步
        with torch.no_grad():                            # 参数更新不需要梯度追踪
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue                             # 跳过无梯度的参数

                g = p.grad                               # 当前梯度 g_t

                # 更新一阶矩（动量估计）: m_t = β1·m_{t-1} + (1-β1)·g_t
                # 相当于指数移动平均，β1 控制历史梯度的衰减速度
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

                # 更新二阶矩（梯度平方的移动平均）: v_t = β2·v_{t-1} + (1-β2)·g_t²
                # 用于自适应调整每个参数的学习率
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

                # 偏差校正: m̂_t = m_t / (1 - β1^t)
                # 早期 t 较小时，(1-β1^t) 接近 0，m̂_t 会被放大，补偿初始化偏差
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)

                # 偏差校正: v̂_t = v_t / (1 - β2^t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 参数更新: θ -= η · m̂ / (√v̂ + ε)
                # m̂ 提供更新方向（类似动量），√v̂ 提供自适应缩放（大梯度参数学习率更小）
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """将所有参数的梯度清零。在每次 backward() 前调用，防止梯度累积。"""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()                           # 原地置零，避免内存分配

    
    
    ''',
    "demo": '''








torch.manual_seed(0)
w = torch.randn(4, 3, requires_grad=True)
opt = MyAdam([w], lr=0.01)
for i in range(5):
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f'Step {i}: loss={loss.item():.4f}')
    
    
    
    
    
    
    
    
    ''',
}
