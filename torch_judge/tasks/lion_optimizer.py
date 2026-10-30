"""Lion Optimizer (EvoLved Sign Momentum) task."""

TASK = {
    'title': 'Lion Optimizer (EvoLved Sign Momentum)',
    'title_zh': 'Lion 优化器（符号动量）',
    'difficulty': 'Medium',
    'category': '优化器与学习率',
    'function_name': 'LionOptimizer',
    'starter_code': """class LionOptimizer:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass""",
    'hint': """Compute the signed update from the old momentum and current gradient, step the parameter, and only then update the momentum with beta2.""",
    'hint_zh': '先用旧动量和当前梯度计算符号更新并更新参数，之后再用 beta2 更新动量。',
    'description_en': """Implement Lion, a memory-efficient optimizer using a signed interpolation of momentum and gradient. Apply decoupled weight decay before the signed parameter update.""",
    'description_zh': '实现 Lion：它使用动量与梯度插值的符号进行更新，内存占用较低。在符号参数更新前应用解耦权重衰减。',
    'theory_en': """Lion stores one momentum tensor. Its update direction is `sign(beta1*m + (1-beta1)*g)`, while the persistent momentum is updated with beta2 after the parameter step.""",
    'theory_zh': 'Lion 仅保存一个动量张量。更新方向为 `sign(beta1*m + (1-beta1)*g)`，持久动量在参数更新后使用 beta2 更新。',
    'diagram_en': """```mermaid
flowchart LR
M[old momentum] --> S[sign direction]
G[gradient] --> S
P[parameter] --> D[weight decay]
D --> U[parameter step]
S --> U
G --> N[new momentum]
```""",
    'diagram_zh': """```mermaid
flowchart LR
M[旧动量] --> S[符号方向]
G[梯度] --> S
P[参数] --> D[权重衰减]
D --> U[参数更新]
S --> U
G --> N[新动量]
```""",
    "tests": [
        {
            "name": 'first step reference',
            "code": """import torch
p = torch.tensor([1.0, -2.0], requires_grad=True)
opt = {fn}([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.2)
p.grad = torch.tensor([3.0, -4.0])
opt.step()
expected = torch.tensor([1.0, -2.0]) * 0.98 - 0.1 * torch.tensor([1.0, -1.0])
assert p.shape == (2,)
assert torch.equal(p.detach(), expected)""",
        },
        {
            "name": 'multiple steps match reference',
            "code": """import torch
p = torch.tensor([0.5, -0.25], requires_grad=True)
opt = {fn}([p], lr=0.03, betas=(0.8, 0.9))
ref = p.detach().clone(); m = torch.zeros_like(ref)
for g in [torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 1.0])]:
    p.grad = g.clone()
    ref.sub_(0.03 * torch.sign(0.8 * m + 0.2 * g))
    m.mul_(0.9).add_(g, alpha=0.1)
    opt.step()
assert torch.equal(p.detach(), ref)""",
        },
        {
            "name": 'zero grad supports autograd',
            "code": """import torch
p = torch.randn(3, requires_grad=True)
opt = {fn}([p])
p.sum().backward()
assert p.grad is not None and p.grad.shape == p.shape
opt.zero_grad()
assert torch.equal(p.grad, torch.zeros_like(p))""",
        },
        {
            "name": 'no gradient no update',
            "code": """import torch
p = torch.randn(2, 3, requires_grad=True)
before = p.detach().clone()
opt = {fn}([p], lr=1.0, weight_decay=1.0)
opt.step()
assert p.shape == (2, 3)
assert torch.equal(p.detach(), before)""",
        },
    ],
    'solution': '''class LionOptimizer:
    """实现符号动量优化器 (Lion Optimizer)。"""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.momentum = [torch.zeros_like(param) for param in self.params]

    def step(self):
        # Step 1: 在无梯度环境应用解耦权重衰减
        with torch.no_grad():
            for index, param in enumerate(self.params):
                if param.grad is None:
                    continue
                grad = param.grad
                param.mul_(1.0 - self.lr * self.weight_decay)

                # Step 2: 用旧动量与当前梯度的插值符号更新参数
                update = self.beta1 * self.momentum[index] + (1.0 - self.beta1) * grad
                param.add_(torch.sign(update), alpha=-self.lr)

                # Step 3: 参数更新后再刷新持久动量
                self.momentum[index].mul_(self.beta2).add_(grad, alpha=1.0 - self.beta2)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()''',
    'demo': """torch.manual_seed(42)
w = torch.randn(4, requires_grad=True)
optimizer = LionOptimizer([w], lr=1e-3)
before = w.detach().clone()
w.square().sum().backward()
optimizer.step()
print("Parameter shape:", w.shape)
print("Updated values:", w.detach())
print("Changed:", not torch.equal(before, w.detach()))""",
    'explanation': r"""### 核心思路
Lion 使用 $\operatorname{sign}(\beta_1m+(1-\beta_1)g)$ 决定更新方向，并在参数更新后用 $\beta_2$ 刷新唯一的动量状态。解耦权重衰减直接作用于参数。

### 代码步骤拆解
- **Step 1（权重衰减）**：仅对存在梯度的参数执行乘法衰减。
- **Step 2（符号更新）**：由旧动量和梯度插值生成 `update`，按学习率更新参数。
- **Step 3（动量刷新）**：使用 `beta2` 原地更新持久 `momentum`。

### 重点与易错点分析
- **注意点 1（更新顺序）**：动量必须在参数步之后更新，否则方向使用了错误的新动量。
- **注意点 2（无梯度参数）**：`grad is None` 时应完全跳过，包括权重衰减。
- **注意点 3（原地安全）**：所有参数修改都放在 `torch.no_grad()` 中，避免破坏叶子张量计算图。

### 复杂度分析
- **时间复杂度**：$O(P)$，每步遍历有梯度的 $P$ 个参数元素。
- **空间复杂度**：$O(P)$，每个参数只保存一份动量。""",
}
