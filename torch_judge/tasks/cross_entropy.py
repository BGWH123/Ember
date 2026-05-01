"""Cross-Entropy Loss task."""

TASK = {
    "title": "Cross-Entropy Loss",
    "title_zh": "交叉熵损失",
    "difficulty": "Easy",
    "category": "损失函数",
    "description_en": (
        "Implement cross-entropy loss for classification.\n\nCross-entropy measures the difference between predicted logits and true class labels. It is the standard loss for classification tasks.\n\n**Signature:** `cross_entropy_loss(logits, targets) -> Tensor`\n\n**Parameters:**\n- `logits` — raw scores (B, C) where C is the number of classes\n- `targets` — ground-truth class indices (B,)\n\n**Returns:** scalar mean loss\n\n**Constraints:**\n- Must be numerically stable (handle large logits)\n- Use log-sum-exp trick for stability"
    ),
    "description_zh": (
        "实现分类交叉熵损失。\n\n交叉熵衡量预测 logits 与真实类别标签之间的差异，是分类任务的标准损失函数。\n\n**签名:** `cross_entropy_loss(logits, targets) -> Tensor`\n\n**参数:**\n- `logits` — 原始分数 (B, C)，C 为类别数\n- `targets` — 真实类别索引 (B,)\n\n**返回:** 标量平均损失\n\n**约束:**\n- 必须数值稳定（处理大 logits）\n- 使用 log-sum-exp 技巧保证稳定性"
    ),
    "function_name": "cross_entropy_loss",
    "hint": (
        "1. `log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)`\n2. `return -log_probs[arange(B), targets].mean()`"
    ),
    "hint_zh": (
        "1. `log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)`\n2. `return -log_probs[arange(B), targets].mean()`"
    ),
    "theory_en": (
        "Cross-Entropy Loss measures the difference between predicted probability distribution and true labels.\n\n**Formula:**\n$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{c=1}^C y_{i,c} \\log(\\hat{y}_{i,c})$$\n\nFor hard labels (one-hot $y$), this simplifies to:\n$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\log(\\hat{y}_{i, t_i})$$\nwhere $t_i$ is the true class index.\n\n**Log-Sum-Exp Trick:**\nTo compute $\\log \\sum_j e^{x_j}$ stably:\n$$\\text{lse}(x) = \\max(x) + \\log \\sum_j e^{x_j - \\max(x)}$$\nThis prevents overflow when logits are large."
    ),
    "theory_zh": (
        "交叉熵损失衡量预测概率分布与真实标签之间的差异。\n\n**公式：**\n$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{c=1}^C y_{i,c} \\log(\\hat{y}_{i,c})$$\n\n对于硬标签（one-hot $y$），简化为：\n$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\log(\\hat{y}_{i, t_i})$$\n其中 $t_i$ 是真实类别索引。\n\n**Log-Sum-Exp 技巧：**\n为稳定计算 $\\log \\sum_j e^{x_j}$：\n$$\\text{lse}(x) = \\max(x) + \\log \\sum_j e^{x_j - \\max(x)}$$\n这防止了大 logits 时的溢出问题。"
    ),
    "tests": [
        {
            "name": "Matches F.cross_entropy",
            "code": """









import torch
torch.manual_seed(0)
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
out = {fn}(logits, targets)
ref = torch.nn.functional.cross_entropy(logits, targets)
assert torch.allclose(out, ref, atol=1e-5), f'Mismatch: {out.item():.4f} vs {ref.item():.4f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Numerical stability",
            "code": """









import torch
logits = torch.tensor([[1000., 0., 0.], [0., 1000., 0.]])
targets = torch.tensor([0, 1])
out = {fn}(logits, targets)
assert not torch.isnan(out), 'NaN with large logits'
assert not torch.isinf(out), 'Inf with large logits'
assert out.item() < 0.01, 'Should be ~0 for confident correct predictions'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Scalar output",
            "code": """









import torch
out = {fn}(torch.randn(8, 5), torch.randint(0, 5, (8,)))
assert out.dim() == 0, 'Loss must be a scalar'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
logits = torch.randn(8, 5, requires_grad=True)
targets = torch.randint(0, 5, (8,))
{fn}(logits, targets).backward()
assert logits.grad is not None, 'logits.grad is None'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''





def cross_entropy_loss(logits, targets):
    """
    Cross-Entropy Loss implementation.
    衡量预测概率分布与真实标签之间的差异。对于 one-hot 标签，等价于负对数似然。
    """
    # logits: (N, C) 未归一化的分数，targets: (N,) 类别索引
    N = logits.shape[0]                                  # 样本数

    # Step 1: Log-Sum-Exp 技巧计算 log(Σ_j exp(logit_j))
    # 先减去最大值防止溢出: log(Σ exp(x_j)) = max(x) + log(Σ exp(x_j - max(x)))
    log_sum_exp = logits.logsumexp(dim=-1)               # (N,)，每个样本的归一化常数

    # Step 2: 提取正样本（真实标签）对应的 logits
    # logits[range(N), targets] 取出每个样本真实类别的未归一化分数
    positive_logits = logits[torch.arange(N), targets]   # (N,)

    # Step 3: 计算交叉熵: -log(p_i) = -(logit_i - log_sum_exp) = log_sum_exp - logit_i
    # 平均到每个样本，得到最终损失
    return (log_sum_exp - positive_logits).mean()        # 标量，平均交叉熵损失

    
    
    
    
    
    ''',
    "demo": '''








logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())
    
    
    
    
    
    
    
    
    ''',
}
