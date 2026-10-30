"""AdamW Optimizer (Decoupled Weight Decay) task."""

TASK = {
    'title': 'AdamW Optimizer (Decoupled Weight Decay)',
    'title_zh': 'AdamW 优化器（解耦权重衰减）',
    'difficulty': 'Medium',
    'category': '优化器与学习率',
    'function_name': 'MyAdamW',
    'starter_code': """class MyAdamW:
    pass""",
    'hint': """Keep one first moment, second moment, and step counter per parameter. Apply weight decay directly to the parameter before the Adam update.""",
    'hint_zh': '为每个参数保存一阶矩、二阶矩和步数。在 Adam 更新前直接对参数应用权重衰减。',
    'description_en': """Implement AdamW with decoupled weight decay. For every parameter with a gradient, first multiply the parameter by `1 - lr * weight_decay`, then perform a bias-corrected Adam update. Provide `step()` and `zero_grad()` methods and match `torch.optim.AdamW` numerically.""",
    'description_zh': """实现带解耦权重衰减的 AdamW。对每个有梯度的参数，先乘以 `1 - lr * weight_decay`，再执行带偏差校正的 Adam 更新。提供 `step()` 与 `zero_grad()`，并与 `torch.optim.AdamW` 数值一致。""",
    'theory_en': """AdamW separates regularization from the gradient: `p <- (1-lr*wd)p`. It then computes exponential moving averages `m` and `v`, corrects their initialization bias, and scales the update by the root second moment.""",
    'theory_zh': 'AdamW 将正则化与梯度分离：`p <- (1-lr*wd)p`。随后计算一阶、二阶指数移动平均，进行初始化偏差校正，并用二阶矩平方根缩放更新。',
    'diagram_en': """```mermaid
flowchart LR
G[gradient] --> M[moment updates]
P[parameter] --> D[decoupled decay]
D --> U[Adam update]
M --> U
```""",
    'diagram_zh': """```mermaid
flowchart LR
G[梯度] --> M[矩更新]
P[参数] --> D[解耦衰减]
D --> U[Adam 更新]
M --> U
```""",
    "tests": [
        {
            "name": 'matches PyTorch',
            "code": """import torch
torch.manual_seed(0)
p = torch.randn(4, 3, requires_grad=True)
q = p.detach().clone().requires_grad_(True)
mine = {fn}([p], lr=3e-3, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1)
ref = torch.optim.AdamW([q], lr=3e-3, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1)
for _ in range(4):
    (p.square().sum()).backward(); (q.square().sum()).backward()
    mine.step(); ref.step(); mine.zero_grad(); ref.zero_grad()
assert p.shape == q.shape
assert torch.allclose(p, q, atol=1e-7, rtol=1e-6), (p-q).abs().max()""",
        },
        {
            "name": 'decay is decoupled',
            "code": """import torch
p = torch.tensor([2.0], requires_grad=True)
opt = {fn}([p], lr=0.1, betas=(0.0, 0.0), eps=0.0, weight_decay=0.2)
p.grad = torch.tensor([1.0])
opt.step()
expected = torch.tensor([2.0 * 0.98 - 0.1])
assert p.shape == (1,)
assert torch.equal(p.detach(), expected)""",
        },
        {
            "name": 'zero grad and gradient flow',
            "code": """import torch
p = torch.randn(5, requires_grad=True)
opt = {fn}([p])
p.square().sum().backward()
assert p.grad is not None and p.grad.abs().sum() > 0
opt.zero_grad()
assert torch.equal(p.grad, torch.zeros_like(p))""",
        },
        {
            "name": 'missing gradient is skipped',
            "code": """import torch
p = torch.tensor([1.0], requires_grad=True)
q = torch.tensor([2.0], requires_grad=True)
opt = {fn}([p, q], lr=0.1, weight_decay=0.5)
p.grad = torch.tensor([0.25])
before = q.detach().clone()
opt.step()
assert q.shape == before.shape
assert torch.equal(q.detach(), before)""",
        },
    ],
    'solution': '''class MyAdamW:
    """解耦权重衰减优化器 (AdamW Optimizer)。"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.first_moments = [torch.zeros_like(param) for param in self.params]
        self.second_moments = [torch.zeros_like(param) for param in self.params]
        self.step_counts = [0 for _ in self.params]

    def step(self):
        # Step 1: 对有梯度参数应用解耦权重衰减
        with torch.no_grad():
            for index, param in enumerate(self.params):
                if param.grad is None:
                    continue
                self.step_counts[index] += 1
                grad = param.grad
                param.mul_(1.0 - self.lr * self.weight_decay)

                # Step 2: 更新矩估计并进行偏差校正
                self.first_moments[index].mul_(self.beta_1).add_(
                    grad, alpha=1.0 - self.beta_1
                )
                self.second_moments[index].mul_(self.beta_2).addcmul_(
                    grad, grad, value=1.0 - self.beta_2
                )
                first_hat = self.first_moments[index] / (
                    1.0 - self.beta_1 ** self.step_counts[index]
                )
                second_hat = self.second_moments[index] / (
                    1.0 - self.beta_2 ** self.step_counts[index]
                )
                param.sub_(self.lr * first_hat / (torch.sqrt(second_hat) + self.eps))

    def zero_grad(self):
        # Step 3: 原地清零已有梯度
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()''',
    'demo': """torch.manual_seed(42)
weights = torch.randn(4, requires_grad=True)
optimizer = MyAdamW([weights], lr=0.01, weight_decay=0.1)
loss = weights.square().sum()
loss.backward()
optimizer.step()
optimizer.zero_grad()
print("Output shape:", weights.shape)
print("Loss:", loss.item())
print("Sample values:", weights[:3])""",
    'explanation': r"""### 核心思路
AdamW 先独立执行 $\theta\leftarrow(1-\eta\lambda)\theta$，再使用 Adam 的偏差校正矩更新，避免将权重衰减混入梯度统计。

### 代码步骤拆解
- **Step 1（解耦衰减）**：对每个有梯度的 `param` 乘以 `1 - lr * weight_decay`。
- **Step 2（自适应更新）**：更新两阶矩，按各参数的 `step_counts` 校正后执行 Adam 步骤。
- **Step 3（梯度清理）**：原地清零已有梯度，不为缺失梯度额外分配张量。

### 重点与易错点分析
- **注意点 1（衰减顺序）**：权重衰减直接作用于参数，不能写成梯度中的 L2 正则项。
- **注意点 2（参数独立状态）**：无梯度参数既不衰减也不增加计数，状态必须逐参数维护。
- **注意点 3（更新隔离）**：所有原地参数更新都置于 `torch.no_grad()`，防止污染计算图。

### 复杂度分析
- **时间复杂度**：$O(P)$，瓶颈是参数与矩状态的逐元素更新。
- **空间复杂度**：$O(P)$，额外维护两份矩状态和少量计数。""",
}
