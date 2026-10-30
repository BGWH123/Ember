"""Generalized Jensen-Shannon distillation loss task."""

TASK = {
    "title": "On-Policy Distillation: Generalized JSD (GKD) Loss",
    "title_zh": "在线策略蒸馏：广义 JS 散度（GKD）损失",
    "difficulty": "Medium",
    "category": "损失函数",
    "function_name": "gkd_jsd_loss",
    "starter_code": '''import torch

def gkd_jsd_loss(student_logits, teacher_logits, beta=0.5,
                 temperature=1.0, eps=1e-8):
    pass
''',
    "hint": "Detach teacher logits, temperature-scale both softmaxes, construct the beta mixture, compute both KL divergences along vocabulary, and average.",
    "hint_zh": "分离教师 logits，对两者进行温度缩放 softmax，构造 beta 混合分布，沿词表维计算两个 KL 散度并取均值。",
    "description_en": "Implement generalized Jensen-Shannon divergence for on-policy knowledge distillation over batched token logits. Teacher logits are fixed targets.",
    "description_zh": "针对批量 token logits 实现在线知识蒸馏的广义 Jensen-Shannon 散度。教师 logits 必须作为固定目标。",
    "theory_en": "The mixture m=beta*p_t+(1-beta)*p_s gives a finite interpolation target. Weighted forward KL terms yield generalized JSD and coincide at zero when distributions match.",
    "theory_zh": "混合分布 m=beta*p_t+(1-beta)*p_s 提供有限的插值目标。加权前向 KL 构成广义 JSD，两分布相同时为零。",
    "diagram_en": "```mermaid\nflowchart LR\nS[student logits / T] --> PS[softmax]\nT[detached teacher logits / T] --> PT[softmax]\nPS --> M[beta mixture]\nPT --> M\nM --> J[weighted KL mean]\nPS --> J\nPT --> J\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nS[学生 logits / T] --> PS[softmax]\nT[分离教师 logits / T] --> PT[softmax]\nPS --> M[beta 混合]\nPT --> M\nM --> J[加权 KL 均值]\nPS --> J\nPT --> J\n```",
    "tests": [
        {"name": "scalar nonnegative output", "code": """import torch
s = torch.randn(2, 3, 5)
t = torch.randn(2, 3, 5)
loss = {fn}(s, t)
assert loss.shape == torch.Size([])
assert torch.isfinite(loss) and loss >= -1e-7
"""},
        {"name": "exact generalized JSD", "code": """import torch
s = torch.tensor([[[1.0, -1.0, 0.5]]], dtype=torch.double)
t = torch.tensor([[[0.0, 2.0, -0.5]]], dtype=torch.double)
got = {fn}(s, t, beta=.3, temperature=.7, eps=1e-12)
ps, pt = torch.softmax(s/.7, -1), torch.softmax(t/.7, -1)
m = .3*pt + .7*ps
klt = (pt*(torch.log(pt.clamp(min=1e-12))-torch.log(m.clamp(min=1e-12)))).sum(-1)
kls = (ps*(torch.log(ps.clamp(min=1e-12))-torch.log(m.clamp(min=1e-12)))).sum(-1)
assert torch.allclose(got, (.3*klt+.7*kls).mean(), atol=1e-12, rtol=1e-12)
"""},
        {"name": "identical distributions edge", "code": """import torch
x = torch.randn(1, 2, 4)
loss = {fn}(x, x.clone(), beta=0.8, temperature=2.0)
assert loss.shape == torch.Size([])
assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)
"""},
        {"name": "teacher detached", "code": """import torch
s = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
t = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
{fn}(s, t, beta=.4).backward()
assert s.grad is not None and s.grad.shape == s.shape
assert t.grad is None and torch.isfinite(s.grad).all()
"""},
    ],
    "solution": '''import torch

def gkd_jsd_loss(student_logits, teacher_logits, beta=0.5,
                 temperature=1.0, eps=1e-8):
    p_s = torch.softmax(student_logits / temperature, dim=-1)
    p_t = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    m = beta * p_t + (1.0 - beta) * p_s
    log_m = torch.log(m.clamp(min=eps))
    kl_t = (p_t * (torch.log(p_t.clamp(min=eps)) - log_m)).sum(dim=-1)
    kl_s = (p_s * (torch.log(p_s.clamp(min=eps)) - log_m)).sum(dim=-1)
    jsd = beta * kl_t + (1.0 - beta) * kl_s
    return jsd.mean()
''',
    "demo": '''student = torch.randn(2, 4, 10)
teacher = torch.randn(2, 4, 10)
print(gkd_jsd_loss(student, teacher))
''',
    "explanation": "All divergence sums are over vocabulary and the final mean covers batch and token dimensions. Detaching before teacher softmax blocks all teacher gradients.",
}
