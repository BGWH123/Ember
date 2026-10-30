"""DeepSeek-V3 auxiliary-loss-free MoE routing task."""

TASK = {
    "title": "DeepSeek-V3 Aux-Loss-Free MoE Router",
    "title_zh": "DeepSeek-V3 无辅助损失 MoE 路由",
    "difficulty": "Hard",
    "category": "混合专家模型",
    "function_name": "aux_free_moe_route",
    "starter_code": '''def aux_free_moe_route(x, router_weight, expert_bias, top_k, bias_lr=1e-3, eps=1e-6):
    pass
''',
    "hint": "Rank sigmoid affinities after adding the dynamic bias, but gather and normalize the original unbiased affinities. Use bincount for expert loads.",
    "hint_zh": "在 sigmoid 亲和度上加入动态偏置后排序，但应收集并归一化原始无偏亲和度；用 bincount 统计专家负载。",
    "description_en": "Implement DeepSeek-V3-style auxiliary-loss-free routing. Select experts with bias-adjusted sigmoid scores, compute gates from unbiased scores, and update the per-expert bias from load imbalance.",
    "description_zh": "实现 DeepSeek-V3 风格的无辅助损失路由：用加偏置的 sigmoid 分数选择专家，用无偏分数计算门控，并根据负载不均衡更新专家偏置。",
    "theory_en": "A non-differentiable dynamic bias steers future assignments toward underloaded experts without changing the gate values used by the current token.",
    "theory_zh": "不可微的动态偏置将后续分配引向低负载专家，同时不改变当前 token 实际使用的门控权重。",
    "diagram_en": "```text\nx @ W^T -> sigmoid -> + bias -> top-k indices\n                    \\-> gather unbiased scores -> normalize -> gates\nindices -> expert counts -> signed bias update\n```",
    "diagram_zh": "```text\nx @ W^T -> sigmoid -> + 偏置 -> top-k 索引\n                    \\-> 收集无偏分数 -> 归一化 -> 门控\n索引 -> 专家计数 -> 符号偏置更新\n```",
    "tests": [
        {
            "name": "shapes and normalized gates",
            "code": '''import torch
x = torch.randn(5, 4)
w = torch.randn(3, 4)
b = torch.zeros(3)
idx, gates, new_b = {fn}(x, w, b, top_k=2)
assert idx.shape == gates.shape == (5, 2)
assert new_b.shape == (3,)
assert torch.allclose(gates.sum(-1), torch.ones(5), atol=2e-6)
''',
        },
        {
            "name": "exact routing reference",
            "code": '''import torch
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
w = torch.tensor([[2., 0.], [0., 2.], [1., 1.]])
b = torch.tensor([0.0, 0.2, -0.1])
idx, gates, new_b = {fn}(x, w, b, 2, bias_lr=0.05)
s = torch.sigmoid(x @ w.T)
ref_idx = torch.topk(s + b.unsqueeze(0), 2, dim=-1).indices
sel = torch.gather(s, -1, ref_idx)
ref_g = sel / (sel.sum(-1, keepdim=True) + 1e-6)
counts = torch.bincount(ref_idx.reshape(-1), minlength=3).to(s.dtype)
ref_b = b - 0.05 * torch.sign(counts - 2.0)
assert torch.equal(idx, ref_idx)
assert torch.allclose(gates, ref_g, atol=1e-7)
assert torch.equal(new_b, ref_b)
''',
        },
        {
            "name": "all experts edge case",
            "code": '''import torch
torch.manual_seed(4)
x = torch.randn(2, 3)
w = torch.randn(4, 3)
b = torch.randn(4)
idx, gates, new_b = {fn}(x, w, b, top_k=4, bias_lr=0.3)
assert torch.equal(torch.sort(idx, dim=-1).values, torch.arange(4).expand(2, 4))
assert torch.equal(new_b, b)
assert torch.all(gates > 0)
''',
        },
        {
            "name": "gradient flow through gates",
            "code": '''import torch
x = torch.randn(6, 5, requires_grad=True)
w = torch.randn(4, 5, requires_grad=True)
b = torch.zeros(4)
idx, gates, new_b = {fn}(x, w, b, 2)
(gates.square().sum()).backward()
assert idx.dtype == torch.long and new_b.shape == b.shape
assert x.grad is not None and w.grad is not None
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
''',
        },
    ],
    "solution": '''import torch

def aux_free_moe_route(x, router_weight, expert_bias, top_k, bias_lr=1e-3, eps=1e-6):
    s = torch.sigmoid(x @ router_weight.T)
    s_biased = s + expert_bias.unsqueeze(0)
    topk_idx = torch.topk(s_biased, k=top_k, dim=-1).indices
    s_selected = torch.gather(s, dim=-1, index=topk_idx)
    gates = s_selected / (s_selected.sum(dim=-1, keepdim=True) + eps)
    num_experts = router_weight.shape[0]
    counts = torch.bincount(topk_idx.reshape(-1), minlength=num_experts).to(s.dtype)
    avg_load = (x.shape[0] * top_k) / num_experts
    new_bias = expert_bias - bias_lr * torch.sign(counts - avg_load)
    return topk_idx, gates, new_bias
''',
    "demo": '''x = torch.randn(8, 16)
router_weight = torch.randn(4, 16)
expert_bias = torch.zeros(4)
indices, gates, updated_bias = aux_free_moe_route(x, router_weight, expert_bias, 2)
print(indices.shape, gates.sum(-1), updated_bias)
''',
    "explanation": "Selection uses biased scores to influence load, while gate magnitudes come from unbiased affinities. Experts above average load have their bias decreased and experts below average have it increased.",
}
