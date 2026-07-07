"""KL Divergence task."""

TASK = {
    "title": "KL Divergence",
    "title_zh": "KL 散度",
    "difficulty": "Medium",
    "category": "损失函数",
    "description_en": "Implement KL divergence between two probability distributions p (target) and q (predicted): KL(p||q) = Σ p_i * (log(p_i) - log(q_i)). Both p and q are probability distributions (sum to 1). Add a small epsilon (1e-10) to avoid log(0).",
    "description_zh": "实现两个概率分布 p（目标）和 q（预测）之间的 KL 散度：KL(p||q) = Σ p_i * (log(p_i) - log(q_i))。p 和 q 都是概率分布（和为1）。添加小 epsilon（1e-10）避免 log(0)。",
    "function_name": "kl_divergence",
    "hint": "Clip q with epsilon before log. Use `(p * (torch.log(p + eps) - torch.log(q + eps))).sum()`.",
    "hint_zh": "对 q 做 epsilon 裁剪后再取 log。使用 `(p * (torch.log(p + eps) - torch.log(q + eps))).sum()`。",
    "theory_en": "KL divergence measures how much one probability distribution q diverges from a reference distribution p. It is non-negative and zero only when p = q. Widely used in VAE, diffusion models, and knowledge distillation.",
    "theory_zh": "KL 散度衡量概率分布 q 相对于参考分布 p 的差异。非负，且仅当 p=q 时为0。广泛用于 VAE、扩散模型和知识蒸馏。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
        {
            "name": "basic",
            "code": "",
        },
    ],
    "solution": '''

import torch

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    KL divergence: KL(p||q) = sum(p * (log(p) - log(q)))
    p: target distribution, q: predicted distribution.
    Both should sum to 1 (probability distributions).
    """
    # 添加 epsilon 避免 log(0)，然后逐元素计算
    # Add epsilon to avoid log(0), compute element-wise
    p_safe = p + eps
    q_safe = q + eps
    kl = (p_safe * (torch.log(p_safe) - torch.log(q_safe))).sum()
    return kl

    
    ''',
    "demo": "",
}
