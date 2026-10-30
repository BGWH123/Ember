"""KL Divergence task."""

TASK = {
    'title': 'KL Divergence',
    'title_zh': 'KL 散度',
    'difficulty': 'Medium',
    'category': '损失函数',
    'function_name': 'kl_divergence',
    'starter_code': """def kl_divergence(p, q, eps=1e-10):
    pass""",
    'hint': 'Clip q with epsilon before log. Use `(p * (torch.log(p + eps) - torch.log(q + eps))).sum()`.',
    'hint_zh': '对 q 做 epsilon 裁剪后再取 log。使用 `(p * (torch.log(p + eps) - torch.log(q + eps))).sum()`。',
    'description_en': """Implement KL divergence between two probability distributions p (target) and q (predicted): KL(p||q) = Σ p_i * (log(p_i) - log(q_i)). Both p and q are probability distributions (sum to 1). Add a small epsilon (1e-10) to avoid log(0).""",
    'description_zh': """实现两个概率分布 p（目标）和 q（预测）之间的 KL 散度：KL(p||q) = Σ p_i * (log(p_i) - log(q_i))。p 和 q 都是概率分布（和为1）。添加小 epsilon（1e-10）避免 log(0)。""",
    'theory_en': """KL divergence measures how much one probability distribution q diverges from a reference distribution p. It is non-negative and zero only when p = q. Widely used in VAE, diffusion models, and knowledge distillation.""",
    'theory_zh': 'KL 散度衡量概率分布 q 相对于参考分布 p 的差异。非负，且仅当 p=q 时为0。广泛用于 VAE、扩散模型和知识蒸馏。',
    "tests": [
        {
            "name": 'Asymmetric KL direction',
            "code": """p = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
q = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
expected = (p * (p.log() - q.log())).sum()
assert torch.allclose({fn}(p, q), expected, atol=1e-8)
assert abs({fn}(p, q).item() - {fn}(q, p).item()) > 0.01""",
        },
        {
            "name": 'Zeros and identical distributions',
            "code": """p = torch.tensor([0., 0.4, 0.6], dtype=torch.float64)
assert abs({fn}(p, p).item()) < 1e-10
q = torch.tensor([0.3, 0., 0.7], dtype=torch.float64)
eps = 1e-5
expected = ((p + eps) * ((p + eps).log() - (q + eps).log())).sum()
assert torch.isfinite({fn}(p, q, eps))
assert torch.allclose({fn}(p, q, eps), expected, atol=1e-10)""",
        },
        {
            "name": 'Gradient with respect to prediction',
            "code": """p = torch.tensor([0.2, 0.8], dtype=torch.float64)
q = torch.tensor([0.6, 0.4], dtype=torch.float64, requires_grad=True)
eps = 1e-6
{fn}(p, q, eps).backward()
assert torch.allclose(q.grad, -(p + eps) / (q.detach() + eps), atol=1e-9)""",
        },
    ],
    'solution': '''def kl_divergence(p, q, eps=1e-10):
    """计算离散分布的 KL 散度 (KL Divergence)。"""
    # Step 1: 为两个分布加入对数稳定项
    p_safe = p + eps
    q_safe = q + eps

    # Step 2: 计算逐元素对数概率差
    log_ratio = torch.log(p_safe) - torch.log(q_safe)

    # Step 3: 按目标分布加权并求和
    return (p_safe * log_ratio).sum()''',
    'demo': """torch.manual_seed(42)
p = torch.softmax(torch.randn(5), dim=0)
q = torch.softmax(torch.randn(5), dim=0)
loss = kl_divergence(p, q)
print("Output shape:", loss.shape)
print("KL value:", loss.item())
print("Reverse KL:", kl_divergence(q, p).item())""",
    'explanation': r"""### 核心思路
按照 $KL(p\|q)=\sum_i(p_i+\epsilon)[\log(p_i+\epsilon)-\log(q_i+\epsilon)]$ 计算有方向的分布差异。

### 代码步骤拆解
- **Step 1（稳定概率）**：`p_safe` 与 `q_safe` 在零概率位置也保持正值。
- **Step 2（对数比）**：`log_ratio` 与输入分布形状相同。
- **Step 3（加权归约）**：以 `p_safe` 加权后对所有元素求和得到标量。

### 重点与易错点分析
- **注意点 1（方向性）**：$KL(p\|q)$ 通常不等于 $KL(q\|p)$，两者参数不能交换。
- **注意点 2（eps 语义）**：测试要求同时给 `p`、`q` 加 `eps`，而不是只对 `q` 做截断。
- **注意点 3（梯度）**：对预测分布 `q` 的梯度为 $-(p+\epsilon)/(q+\epsilon)$。

### 复杂度分析
- **时间复杂度**：$O(M)$，其中 $M$ 为分布元素数。
- **空间复杂度**：$O(M)$，保存安全概率与对数比。""",
}
