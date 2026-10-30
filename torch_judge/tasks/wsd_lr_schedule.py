"""Warmup-Stable-Decay learning-rate task."""

TASK = {
    "title": "Warmup-Stable-Decay (WSD) LR Scheduler",
    "title_zh": "WSD（预热-稳定-衰减）学习率调度",
    "difficulty": "Easy",
    "category": "优化器与学习率",
    "function_name": "wsd_lr_schedule",
    "starter_code": '''import math

def wsd_lr_schedule(step, warmup_steps, stable_steps, decay_steps, max_lr, min_lr=0.0):
    pass
''',
    "hint": "Use three ordered boundary checks: warmup, stable plateau, and cosine decay. Return `min_lr` after the total duration.",
    "hint_zh": "依次判断预热、稳定平台和余弦衰减三个区间；超过总步数后返回 `min_lr`。",
    "description_en": "Implement a Warmup-Stable-Decay learning-rate schedule: linear warmup, a constant plateau, then cosine decay to a minimum learning rate.",
    "description_zh": "实现预热-稳定-衰减学习率调度：线性预热、恒定平台，最后余弦衰减至最小学习率。",
    "theory_en": "Warmup limits early optimization shocks, the stable phase performs most training at peak rate, and smooth cosine decay reduces the step size toward convergence.",
    "theory_zh": "预热减轻训练初期冲击，稳定阶段以峰值学习率完成主要训练，平滑余弦衰减则逐步减小步长以收敛。",
    "diagram_en": "```mermaid\nflowchart LR\nW[linear warmup] --> S[stable max LR] --> D[cosine decay] --> M[min LR]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nW[线性预热] --> S[峰值稳定期] --> D[余弦衰减] --> M[最小学习率]\n```",
    "tests": [
        {"name": "linear warmup", "code": """import torch
lr = {fn}(2, 4, 3, 5, 0.01, 0.001)
assert isinstance(lr, float)
assert abs(lr - 0.005) < 1e-12
assert torch.tensor(lr).shape == ()
"""},
        {"name": "stable boundaries", "code": """import torch
a = {fn}(4, 4, 3, 5, 0.01, 0.001)
b = {fn}(6, 4, 3, 5, 0.01, 0.001)
assert torch.tensor([a, b]).shape == (2,)
assert a == 0.01 and b == 0.01
"""},
        {"name": "cosine exact reference", "code": """import torch
import math
lr = {fn}(9, 4, 3, 5, 0.01, 0.001)
progress = (9 - 4 - 3) / 5
expected = 0.001 + 0.5 * 0.009 * (1.0 + math.cos(math.pi * progress))
assert torch.allclose(torch.tensor(lr), torch.tensor(expected), atol=0, rtol=0)
"""},
        {"name": "end and after end", "code": """import torch
end = {fn}(12, 4, 3, 5, 0.01, 0.001)
after = {fn}(99, 4, 3, 5, 0.01, 0.001)
assert torch.tensor([end, after]).shape == (2,)
assert abs(end - 0.001) < 1e-12
assert after == 0.001
"""},
    ],
    "solution": '''import math

def wsd_lr_schedule(step, warmup_steps, stable_steps, decay_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return float(max_lr * (step / warmup_steps))
    if step < warmup_steps + stable_steps:
        return float(max_lr)
    total = warmup_steps + stable_steps + decay_steps
    if step <= total:
        progress = (step - warmup_steps - stable_steps) / decay_steps
        return float(min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress)))
    return float(min_lr)
''',
    "demo": '''values = [wsd_lr_schedule(i, 2, 3, 5, 1e-3, 1e-5) for i in range(12)]
print(values)
''',
    "explanation": "The interval definitions make the warmup endpoint part of the stable phase and the decay endpoint evaluate exactly to `min_lr`.",
}
