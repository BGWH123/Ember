"""Min-p sampling with repetition penalty task."""

TASK = {
    "title": "Min-P & Repetition Penalty Sampling Filter",
    "title_zh": "Min-P 采样与重复惩罚过滤",
    "difficulty": "Medium",
    "category": "采样与解码",
    "function_name": "min_p_sampling_probs",
    "starter_code": '''def min_p_sampling_probs(logits, temperature=1.0, min_p=0.1, seen_tokens=None, repetition_penalty=1.0):
    pass
''',
    "hint": "Scatter seen token IDs into a boolean `(B,V)` mask, apply the sign-aware penalty with `torch.where`, softmax temperature-scaled logits, then threshold relative to each row's maximum probability.",
    "hint_zh": "把已见 token ID scatter 到 `(B,V)` 布尔掩码，用 `torch.where` 应用符号相关惩罚；温度缩放后 softmax，再按每行最大概率的相对阈值过滤。",
    "description_en": "Return normalized sampling probabilities after a sign-aware repetition penalty and min-p filtering.",
    "description_zh": "在应用符号相关的重复惩罚与 Min-P 过滤后，返回归一化采样概率。",
    "theory_en": "Min-p adapts its cutoff to model confidence by retaining tokens whose probability is at least a fraction of the most likely token.",
    "theory_zh": "Min-P 通过保留概率至少为最高概率一定比例的 token，使截断阈值随模型置信度自适应。",
    "diagram_en": "```text\nlogits -> repetition penalty -> / temperature -> softmax -> p >= min_p*p_max -> renormalize\n```",
    "diagram_zh": "```text\nlogits -> 重复惩罚 -> / 温度 -> softmax -> p >= min_p*p_max -> 重归一化\n```",
    "tests": [
        {
            "name": "probability shape and normalization",
            "code": '''import torch
logits = torch.randn(4, 20)
probs = {fn}(logits, temperature=0.8, min_p=0.15)
assert probs.shape == logits.shape
assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-6)
assert (probs >= 0).all()
''',
        },
        {
            "name": "exact penalty and filtering reference",
            "code": '''import torch
logits = torch.tensor([[2., 1., -1., 0.5], [-2., 3., 1., -0.5]])
seen = torch.tensor([[0, 2], [0, 3]])
actual = {fn}(logits, temperature=0.7, min_p=0.2, seen_tokens=seen, repetition_penalty=2.0)
adjusted = logits.clone()
adjusted[0, 0] /= 2.0; adjusted[0, 2] *= 2.0
adjusted[1, 0] *= 2.0; adjusted[1, 3] *= 2.0
ref = torch.softmax(adjusted / 0.7, dim=-1)
ref = torch.where(ref >= 0.2 * ref.max(-1, keepdim=True).values, ref, torch.zeros_like(ref))
ref = ref / ref.sum(-1, keepdim=True)
assert torch.allclose(actual, ref, atol=1e-7)
''',
        },
        {
            "name": "duplicate seen tokens and min-p edge",
            "code": '''import torch
logits = torch.tensor([[3., 2., 1.]])
seen = torch.tensor([[0, 0, 0]])
probs = {fn}(logits, min_p=1.0, seen_tokens=seen, repetition_penalty=2.0)
adjusted = logits.clone(); adjusted[:, 0] /= 2.0
winner = torch.softmax(adjusted, -1).argmax(-1)
assert torch.equal(probs.argmax(-1), winner)
assert torch.count_nonzero(probs).item() == 1 and probs.max().item() == 1.0
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
logits = torch.tensor([[1.5, 1.0, 0.5], [0.3, 0.2, 0.1]], requires_grad=True)
seen = torch.tensor([[0], [2]])
probs = {fn}(logits, min_p=0.0, seen_tokens=seen, repetition_penalty=1.2)
probs[:, 0].sum().backward()
assert logits.grad is not None and logits.grad.shape == logits.shape
assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
''',
        },
    ],
    "solution": '''import torch

def min_p_sampling_probs(logits, temperature=1.0, min_p=0.1, seen_tokens=None, repetition_penalty=1.0):
    adjusted = logits.clone()
    if seen_tokens is not None and repetition_penalty != 1.0:
        seen_mask = torch.zeros_like(adjusted, dtype=torch.bool)
        seen_mask.scatter_(dim=1, index=seen_tokens, value=True)
        penalized = torch.where(
            adjusted > 0,
            adjusted / repetition_penalty,
            adjusted * repetition_penalty,
        )
        adjusted = torch.where(seen_mask, penalized, adjusted)
    scaled_logits = adjusted / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    probs = torch.where(probs < min_p * p_max, torch.zeros_like(probs), probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs
''',
    "demo": '''logits = torch.randn(2, 32000)
seen = torch.tensor([[10, 20, 30], [5, 6, 7]])
probs = min_p_sampling_probs(logits, min_p=0.1, seen_tokens=seen, repetition_penalty=1.1)
print(probs.shape, (probs > 0).sum(-1), probs.sum(-1))
''',
    "explanation": "A vocabulary mask ensures duplicate history entries are penalized only once. Filtering is performed on probabilities before surviving mass is normalized back to one.",
}
