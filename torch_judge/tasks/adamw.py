"""AdamW optimizer task."""

TASK = {
    "title": "AdamW Optimizer (Decoupled Weight Decay)",
    "title_zh": "AdamW 优化器（解耦权重衰减）",
    "difficulty": "Medium",
    "category": "优化器与学习率",
    "function_name": "MyAdamW",
    "starter_code": '''import torch

class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass
''',
    "hint": "Keep one first moment, second moment, and step counter per parameter. Apply weight decay directly to the parameter before the Adam update.",
    "hint_zh": "为每个参数保存一阶矩、二阶矩和步数。在 Adam 更新前直接对参数应用权重衰减。",
    "description_en": "Implement AdamW with decoupled weight decay. For every parameter with a gradient, first multiply the parameter by `1 - lr * weight_decay`, then perform a bias-corrected Adam update. Provide `step()` and `zero_grad()` methods and match `torch.optim.AdamW` numerically.",
    "description_zh": "实现带解耦权重衰减的 AdamW。对每个有梯度的参数，先乘以 `1 - lr * weight_decay`，再执行带偏差校正的 Adam 更新。提供 `step()` 与 `zero_grad()`，并与 `torch.optim.AdamW` 数值一致。",
    "theory_en": "AdamW separates regularization from the gradient: `p <- (1-lr*wd)p`. It then computes exponential moving averages `m` and `v`, corrects their initialization bias, and scales the update by the root second moment.",
    "theory_zh": "AdamW 将正则化与梯度分离：`p <- (1-lr*wd)p`。随后计算一阶、二阶指数移动平均，进行初始化偏差校正，并用二阶矩平方根缩放更新。",
    "diagram_en": "```mermaid\nflowchart LR\nG[gradient] --> M[moment updates]\nP[parameter] --> D[decoupled decay]\nD --> U[Adam update]\nM --> U\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nG[梯度] --> M[矩更新]\nP[参数] --> D[解耦衰减]\nD --> U[Adam 更新]\nM --> U\n```",
    "tests": [
        {"name": "matches PyTorch", "code": """import torch
torch.manual_seed(0)
p = torch.randn(4, 3, requires_grad=True)
q = p.detach().clone().requires_grad_(True)
mine = {fn}([p], lr=3e-3, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1)
ref = torch.optim.AdamW([q], lr=3e-3, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1)
for _ in range(4):
    (p.square().sum()).backward(); (q.square().sum()).backward()
    mine.step(); ref.step(); mine.zero_grad(); ref.zero_grad()
assert p.shape == q.shape
assert torch.allclose(p, q, atol=1e-7, rtol=1e-6), (p-q).abs().max()
"""},
        {"name": "decay is decoupled", "code": """import torch
p = torch.tensor([2.0], requires_grad=True)
opt = {fn}([p], lr=0.1, betas=(0.0, 0.0), eps=0.0, weight_decay=0.2)
p.grad = torch.tensor([1.0])
opt.step()
expected = torch.tensor([2.0 * 0.98 - 0.1])
assert p.shape == (1,)
assert torch.equal(p.detach(), expected)
"""},
        {"name": "zero grad and gradient flow", "code": """import torch
p = torch.randn(5, requires_grad=True)
opt = {fn}([p])
p.square().sum().backward()
assert p.grad is not None and p.grad.abs().sum() > 0
opt.zero_grad()
assert torch.equal(p.grad, torch.zeros_like(p))
"""},
        {"name": "missing gradient is skipped", "code": """import torch
p = torch.tensor([1.0], requires_grad=True)
q = torch.tensor([2.0], requires_grad=True)
opt = {fn}([p, q], lr=0.1, weight_decay=0.5)
p.grad = torch.tensor([0.25])
before = q.detach().clone()
opt.step()
assert q.shape == before.shape
assert torch.equal(q.detach(), before)
"""},
    ],
    "solution": '''import torch

class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.steps = [0 for _ in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.steps[i] += 1
                g = p.grad
                p.mul_(1.0 - self.lr * self.weight_decay)
                self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                m_hat = self.m[i] / (1.0 - self.beta1 ** self.steps[i])
                v_hat = self.v[i] / (1.0 - self.beta2 ** self.steps[i])
                p.sub_(self.lr * m_hat / (torch.sqrt(v_hat) + self.eps))

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
''',
    "demo": '''p = torch.tensor([1.0, -1.0], requires_grad=True)
opt = MyAdamW([p], lr=0.01)
p.square().sum().backward()
opt.step()
opt.zero_grad()
print(p)
''',
    "explanation": "The optimizer keeps independent state for every parameter. Parameters without gradients are untouched; active parameters receive decoupled decay followed by the bias-corrected adaptive update.",
}
