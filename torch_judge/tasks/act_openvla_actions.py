"""VLA temporal action ensembling and tokenization task."""

TASK = {
    "title": "VLA Action Chunking Temporal Ensemble & OpenVLA Tokenizer",
    "title_zh": "VLA 动作分块时序集成与 OpenVLA 离散化",
    "difficulty": "Medium",
    "category": "具身智能与控制",
    "function_name": "vla_action_ensemble_and_tokenize",
    "starter_code": '''def vla_action_ensemble_and_tokenize(chunk_preds, decay_m, actions, low, high, n_bins=256):
    pass
''',
    "hint": "Use index zero as age zero for exponential temporal weights. Clamp normalized actions before converting them to integer bins, then decode bin centers.",
    "hint_zh": "令索引 0 对应年龄 0 以计算指数时序权重；将归一化动作截断后转为整数 bin，再解码 bin 中心。",
    "description_en": "Ensemble overlapping action-chunk predictions and uniformly tokenize bounded continuous actions in the OpenVLA style.",
    "description_zh": "集成重叠动作块预测，并以 OpenVLA 风格对有界连续动作进行均匀离散化。",
    "theory_en": "Exponential weighting favors newer chunk predictions while retaining older estimates. Uniform bins map continuous controls to token IDs, and bin centers provide deterministic approximate decoding.",
    "theory_zh": "指数加权偏重新动作块预测，同时保留较旧估计。均匀分箱将连续控制映射到 token ID，箱中心提供确定性的近似解码。",
    "diagram_en": "```text\nchunks -> age decay -> weighted action\nactions,bounds -> normalize -> bins -> bin centers\n```",
    "diagram_zh": "```text\n动作块 -> 年龄衰减 -> 加权动作\n动作、边界 -> 归一化 -> 分箱 -> 箱中心\n```",
    "tests": [
        {"name": "output shapes", "code": '''import torch
chunks = torch.randn(4, 3)
actions = torch.randn(2, 3)
ens, ids, recon = {fn}(chunks, .2, actions, -torch.ones(3), torch.ones(3))
assert ens.shape == (3,) and ids.shape == recon.shape == (2, 3)
'''},
        {"name": "exact reference", "code": '''import torch
chunks = torch.tensor([[1., 2.], [3., 4.]])
a = torch.tensor([[0., 1.]])
low, high = torch.tensor([-1., 0.]), torch.tensor([1., 2.])
ens, ids, recon = {fn}(chunks, torch.tensor(.5), a, low, high, n_bins=4)
w = torch.exp(-torch.tensor(.5) * torch.arange(2.)); w = w / w.sum()
ref_ids = torch.tensor([[2, 2]])
ref_recon = low + ((ref_ids.float() + .5) / 4) * (high - low)
assert torch.allclose(ens, (w[:, None] * chunks).sum(0)) and torch.equal(ids, ref_ids) and torch.allclose(recon, ref_recon)
'''},
        {"name": "bounds clipping edge case", "code": '''import torch
a = torch.tensor([[-5., 5.]])
low, high = torch.tensor([0., 0.]), torch.tensor([1., 1.])
ens, ids, recon = {fn}(torch.tensor([[2., 3.]]), 1., a, low, high, n_bins=2)
assert torch.equal(ens, torch.tensor([2., 3.])) and torch.equal(ids, torch.tensor([[0, 1]])) and torch.allclose(recon, torch.tensor([[.25, .75]]))
'''},
        {"name": "ensemble gradient flow", "code": '''import torch
chunks = torch.randn(3, 2, requires_grad=True)
decay = torch.tensor(.3, requires_grad=True)
ens, ids, recon = {fn}(chunks, decay, torch.zeros(1, 2), -torch.ones(2), torch.ones(2))
ens.sum().backward()
assert chunks.grad is not None and decay.grad is not None and ids.dtype == torch.long and recon.shape == (1, 2)
'''},
    ],
    "solution": '''import torch

def vla_action_ensemble_and_tokenize(chunk_preds, decay_m, actions, low, high, n_bins=256):
    K = chunk_preds.shape[0]
    ages = torch.arange(K, device=chunk_preds.device, dtype=chunk_preds.dtype)
    w = torch.exp(-decay_m * ages)
    w = w / w.sum()
    ensembled = (w.unsqueeze(-1) * chunk_preds).sum(dim=0)
    norm_a = torch.clamp((actions - low) / (high - low + 1e-8), 0.0, 1.0)
    bin_ids = torch.clamp((norm_a * n_bins).long(), 0, n_bins - 1)
    recon_actions = low + ((bin_ids.float() + 0.5) / n_bins) * (high - low)
    return ensembled, bin_ids, recon_actions
''',
    "demo": '''chunks = torch.randn(5, 7)
actions = torch.rand(10, 7) * 2 - 1
print(vla_action_ensemble_and_tokenize(chunks, 0.1, actions, -torch.ones(7), torch.ones(7))[0])
''',
    "explanation": "Chunk index zero is explicitly the newest estimate, so its exponential weight is one before normalization. Values at the upper bound are clamped back to the final valid bin.",
}
