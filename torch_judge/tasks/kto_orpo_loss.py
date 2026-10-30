"""ORPO odds-ratio preference loss task."""

TASK = {
    "title": "ORPO (Odds Ratio Preference Optimization) Loss",
    "title_zh": "ORPO（优势比偏好优化）损失",
    "difficulty": "Medium",
    "category": "损失函数",
    "function_name": "orpo_loss",
    "starter_code": '''import torch
import torch.nn.functional as F

def orpo_loss(chosen_avg_logps, rejected_avg_logps, lambda_or=0.1, eps=1e-6):
    pass
''',
    "hint": "Convert each log-probability to log odds with logp-log(1-exp(logp)), compare chosen and rejected odds using log-sigmoid, and add the chosen SFT loss.",
    "hint_zh": "用 logp-log(1-exp(logp)) 转为对数优势，使用 log-sigmoid 比较选中和拒绝优势，再加上选中样本的 SFT 损失。",
    "description_en": "Implement ORPO using length-normalized sequence log-probabilities. Return the combined objective, chosen-response SFT term, and odds-ratio preference term.",
    "description_zh": "使用长度归一化的序列对数概率实现 ORPO，并返回总目标、选中回复的 SFT 项和优势比偏好项。",
    "theory_en": "ORPO jointly raises chosen likelihood and separates chosen from rejected responses in odds space, avoiding a separate reference model.",
    "theory_zh": "ORPO 同时提高选中回复的似然，并在优势空间中分离选中与拒绝回复，因此不需要独立参考模型。",
    "diagram_en": "```mermaid\nflowchart LR\nC[chosen log p] --> S[SFT]\nC --> OC[chosen log odds]\nR[rejected log p] --> OR[rejected log odds]\nOC --> P[preference loss]\nOR --> P\nS --> T[total]\nP --> T\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nC[选中 log p] --> S[SFT]\nC --> OC[选中对数优势]\nR[拒绝 log p] --> OR[拒绝对数优势]\nOC --> P[偏好损失]\nOR --> P\nS --> T[总损失]\nP --> T\n```",
    "tests": [
        {"name": "output shapes", "code": """import torch
c = torch.tensor([-0.2, -0.7])
r = torch.tensor([-0.9, -1.2])
out = {fn}(c, r)
assert isinstance(out, tuple) and len(out) == 3
assert all(x.shape == torch.Size([]) for x in out)
"""},
        {"name": "exact odds reference", "code": """import torch
import torch.nn.functional as F
c = torch.tensor([-0.2, -1.3], dtype=torch.double)
r = torch.tensor([-0.8, -2.0], dtype=torch.double)
total, sft, pref = {fn}(c, r, lambda_or=0.25, eps=1e-9)
odds = lambda x: x - torch.log((1.0 - torch.exp(x)).clamp(min=1e-9))
ref_sft = -c.mean()
ref_pref = -F.logsigmoid(odds(c) - odds(r)).mean()
assert torch.allclose(sft, ref_sft, atol=1e-12, rtol=1e-12)
assert torch.allclose(pref, ref_pref, atol=1e-12, rtol=1e-12) and torch.allclose(total, ref_sft + .25 * ref_pref)
"""},
        {"name": "epsilon boundary", "code": """import torch
c = torch.tensor([0.0])
r = torch.tensor([-1.0])
total, sft, pref = {fn}(c, r, eps=1e-4)
assert total.shape == torch.Size([])
assert torch.isfinite(total) and torch.isfinite(pref) and torch.equal(sft, torch.tensor(0.0))
"""},
        {"name": "gradient flow", "code": """import torch
c = torch.tensor([-0.3, -0.6], dtype=torch.double, requires_grad=True)
r = torch.tensor([-1.0, -1.4], dtype=torch.double, requires_grad=True)
total, sft, pref = {fn}(c, r)
total.backward()
assert c.grad is not None and r.grad is not None
assert c.grad.shape == c.shape and torch.isfinite(c.grad).all() and torch.isfinite(r.grad).all()
"""},
    ],
    "solution": '''import torch
import torch.nn.functional as F

def orpo_loss(chosen_avg_logps, rejected_avg_logps, lambda_or=0.1, eps=1e-6):
    sft_loss = -chosen_avg_logps.mean()
    def log_odds(logp):
        return logp - torch.log((1.0 - torch.exp(logp)).clamp(min=eps))
    log_odds_ratio = log_odds(chosen_avg_logps) - log_odds(rejected_avg_logps)
    pref_loss = -F.logsigmoid(log_odds_ratio).mean()
    total_loss = sft_loss + lambda_or * pref_loss
    return total_loss, sft_loss, pref_loss
''',
    "demo": '''chosen = torch.tensor([-0.2, -0.4])
rejected = torch.tensor([-0.8, -1.0])
print(orpo_loss(chosen, rejected))
''',
    "explanation": "Clamping only the one-minus-probability term prevents log(0). The SFT and preference components remain separately observable.",
}
