"""Lion optimizer task."""

TASK = {
    "title": "Lion Optimizer (EvoLved Sign Momentum)",
    "title_zh": "Lion 优化器（符号动量）",
    "difficulty": "Medium",
    "category": "优化器与学习率",
    "function_name": "LionOptimizer",
    "starter_code": '''import torch

class LionOptimizer:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass
''',
    "hint": "Compute the signed update from the old momentum and current gradient, step the parameter, and only then update the momentum with beta2.",
    "hint_zh": "先用旧动量和当前梯度计算符号更新并更新参数，之后再用 beta2 更新动量。",
    "description_en": "Implement Lion, a memory-efficient optimizer using a signed interpolation of momentum and gradient. Apply decoupled weight decay before the signed parameter update.",
    "description_zh": "实现 Lion：它使用动量与梯度插值的符号进行更新，内存占用较低。在符号参数更新前应用解耦权重衰减。",
    "theory_en": "Lion stores one momentum tensor. Its update direction is `sign(beta1*m + (1-beta1)*g)`, while the persistent momentum is updated with beta2 after the parameter step.",
    "theory_zh": "Lion 仅保存一个动量张量。更新方向为 `sign(beta1*m + (1-beta1)*g)`，持久动量在参数更新后使用 beta2 更新。",
    "diagram_en": "```mermaid\nflowchart LR\nM[old momentum] --> S[sign direction]\nG[gradient] --> S\nP[parameter] --> D[weight decay]\nD --> U[parameter step]\nS --> U\nG --> N[new momentum]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nM[旧动量] --> S[符号方向]\nG[梯度] --> S\nP[参数] --> D[权重衰减]\nD --> U[参数更新]\nS --> U\nG --> N[新动量]\n```",
    "tests": [
        {"name": "first step reference", "code": """import torch
p = torch.tensor([1.0, -2.0], requires_grad=True)
opt = {fn}([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.2)
p.grad = torch.tensor([3.0, -4.0])
opt.step()
expected = torch.tensor([1.0, -2.0]) * 0.98 - 0.1 * torch.tensor([1.0, -1.0])
assert p.shape == (2,)
assert torch.equal(p.detach(), expected)
"""},
        {"name": "multiple steps match reference", "code": """import torch
p = torch.tensor([0.5, -0.25], requires_grad=True)
opt = {fn}([p], lr=0.03, betas=(0.8, 0.9))
ref = p.detach().clone(); m = torch.zeros_like(ref)
for g in [torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 1.0])]:
    p.grad = g.clone()
    ref.sub_(0.03 * torch.sign(0.8 * m + 0.2 * g))
    m.mul_(0.9).add_(g, alpha=0.1)
    opt.step()
assert torch.equal(p.detach(), ref)
"""},
        {"name": "zero grad supports autograd", "code": """import torch
p = torch.randn(3, requires_grad=True)
opt = {fn}([p])
p.sum().backward()
assert p.grad is not None and p.grad.shape == p.shape
opt.zero_grad()
assert torch.equal(p.grad, torch.zeros_like(p))
"""},
        {"name": "no gradient no update", "code": """import torch
p = torch.randn(2, 3, requires_grad=True)
before = p.detach().clone()
opt = {fn}([p], lr=1.0, weight_decay=1.0)
opt.step()
assert p.shape == (2, 3)
assert torch.equal(p.detach(), before)
"""},
    ],
    "solution": '''import torch

class LionOptimizer:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                p.mul_(1.0 - self.lr * self.weight_decay)
                u = torch.sign(self.beta1 * self.m[i] + (1.0 - self.beta1) * g)
                p.sub_(self.lr * u)
                self.m[i].mul_(self.beta2).add_(g, alpha=1.0 - self.beta2)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
''',
    "demo": '''w = torch.randn(4, requires_grad=True)
opt = LionOptimizer([w])
w.square().sum().backward()
opt.step()
print(w)
''',
    "explanation": "The sign operation fixes each coordinate's update magnitude at the learning rate. The momentum buffer smooths the direction, and weight decay remains independent of the gradient.",
}
