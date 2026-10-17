"""Noisy Top-K Gating (MoE) task."""

TASK = {
    "title": "Noisy Top-K Gating (MoE)",
    "title_zh": "噪声 Top-K 门控（MoE）",
    "difficulty": "Hard",
    "category": "混合专家模型",
    "description_en": "Implement Noisy Top-K Gating for Mixture-of-Experts. Given input x and n_experts, compute: logits = x @ W_g + noise * (x @ W_noise), where noise = StandardNormal * softplus(x @ W_noise). Select top-k experts, apply softmax to their logits. Return (weights, expert_indices) where weights sum to 1 for each token.",
    "description_zh": "实现混合专家模型（MoE）的噪声 Top-K 门控。给定输入 x 和专家数量 n_experts，计算：logits = x @ W_g + noise * (x @ W_noise)，其中 noise = StandardNormal * softplus(x @ W_noise)。选择 top-k 专家，对它们的 logits 应用 softmax。返回 (weights, expert_indices)，其中每个 token 的权重和为1。",
    "function_name": "NoisyTopKGating",
    "hint": "Generate noise using torch.randn. Use softplus for noise scaling. Top-k with softmax on selected logits only. Expose the clean-logit and noise projections as self.W_g and self.W_noise, matching the starter class.",
    "hint_zh": "使用 torch.randn 生成噪声。用 softplus 做噪声缩放。仅对选中的 top-k logits 做 softmax。 按起始模板将干净 logits 和噪声投影分别命名为 self.W_g 与 self.W_noise。",
    "theory_en": "Noisy Top-K gating is the routing mechanism in Switch Transformer and other MoE models. Noise helps explore different expert assignments during training, improving load balancing.",
    "theory_zh": "噪声 Top-K 门控是 Switch Transformer 等 MoE 模型的路由机制。噪声帮助训练时探索不同的专家分配，改善负载均衡。",
    "tests": [
    {
        "name": "Deterministic expert ranking without noise",
        "code": "layer = {fn}(3, 4, top_k=2)\nwith torch.no_grad():\n    layer.W_g.weight.copy_(torch.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [-1., 0., 0.]]))\n    layer.W_g.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))\nx = torch.tensor([[2., 1., -1.], [-1., 2., 1.]])\nlogits = F.linear(x, layer.W_g.weight, layer.W_g.bias)\nvalues, indices = logits.topk(2, dim=-1)\nweights, actual_indices = layer(x, training=False)\nassert torch.equal(actual_indices, indices)\nassert torch.allclose(weights, values.softmax(-1), atol=1e-6)"
    },
    {
        "name": "Top-one and all experts",
        "code": "for k in [1, 5]:\n    torch.manual_seed(123)\n    layer = {fn}(3, 5, top_k=k)\n    weights, indices = layer(torch.ones(4, 3), training=False)\n    assert weights.shape == indices.shape == (4, k)\n    assert indices.dtype == torch.long\n    assert ((indices >= 0) & (indices < 5)).all()\n    assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-6)\n    for row in indices:\n        assert row.unique().numel() == k"
    },
    {
        "name": "Training noise and differentiable weights",
        "code": "layer = {fn}(3, 4, top_k=2)\nwith torch.no_grad():\n    layer.W_g.weight.zero_(); layer.W_g.bias.zero_()\n    layer.W_noise.weight.zero_(); layer.W_noise.bias.zero_()\nx = torch.ones(32, 3)\ntorch.manual_seed(99)\nw1, i1 = layer(x, training=True)\ntorch.manual_seed(99)\nw2, i2 = layer(x, training=True)\nassert torch.equal(i1, i2) and torch.allclose(w1, w2)\nassert i1.unique().numel() > 2\nassert torch.allclose(w1.sum(-1), torch.ones(32), atol=1e-6)\n(w1[:, 0].sum()).backward()\nassert layer.W_noise.bias.grad is not None\nassert layer.W_noise.bias.grad.abs().sum() > 0"
    }
],
    "solution": '''

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyTopKGating(nn.Module):
    """
    Noisy Top-K Gating for Mixture-of-Experts.
    Routes each token to k experts with load-balancing noise.
    """
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 门控投影：将输入映射到专家数量维度
        # Gate projection: map input to expert count dimension
        self.W_g = nn.Linear(input_dim, num_experts)
        # 噪声投影：学习每个输入的噪声强度
        # Noise projection: learn noise magnitude per input
        self.W_noise = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, training: bool = True):
        # 干净的路由 logits
        # Clean routing logits
        clean_logits = self.W_g(x)  # (B, num_experts)

        if training:
            # 噪声强度：softplus 保证非负
            # Noise magnitude: softplus ensures non-negative
            noise_std = F.softplus(self.W_noise(x))
            # 采样高斯噪声
            # Sample Gaussian noise
            noise = torch.randn_like(clean_logits) * noise_std
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits

        # Top-K 选择：找出每个 token 最匹配的 k 个专家
        # Top-K selection: find k best experts per token
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)

        # 对选中的专家做 softmax，得到路由权重
        # Softmax over selected experts to get routing weights
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices

    
    ''',
    "demo": "",
}
