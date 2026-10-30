"""QK-Norm Scaled Dot-Product Attention task."""

TASK = {
    'title': 'QK-Norm Scaled Dot-Product Attention',
    'title_zh': 'QK-Norm 注意力机制',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'qk_norm_attention',
    'starter_code': """def qk_norm_attention(q, k, v, gamma_q, gamma_k, eps=1e-06):
    pass""",
    'hint': """RMS-normalize q and k over D, apply their scale vectors, then use ordinary scaled dot-product attention.""",
    'hint_zh': '沿 D 维对 q 和 k 做 RMS 归一化并乘各自缩放向量，然后执行标准缩放点积注意力。',
    'description_en': """Implement multi-head scaled dot-product attention after applying independently scaled RMSNorm to queries and keys.""",
    'description_zh': '先对查询和键应用带独立缩放参数的 RMSNorm，再实现多头缩放点积注意力。',
    'theory_en': """QK-Norm controls query/key magnitudes before their dot product. This can prevent attention logits from growing excessively while preserving learned per-feature scales.""",
    'theory_zh': 'QK-Norm 在点积前控制查询和键的幅值，可避免注意力 logits 过大，同时保留逐特征可学习缩放。',
    'diagram_en': r"""```text
q -> RMSNorm(gamma_q) --\
                         QK^T/sqrt(D) -> softmax -> @v
k -> RMSNorm(gamma_k) --/
```""",
    'diagram_zh': r"""```text
q -> RMSNorm(gamma_q) --\
                         QK^T/sqrt(D) -> softmax -> @v
k -> RMSNorm(gamma_k) --/
```""",
    "tests": [
        {
            "name": 'output shape',
            "code": """import torch
q = torch.randn(2, 3, 5, 4)
k = torch.randn(2, 3, 7, 4)
v = torch.randn(2, 3, 7, 6)
out = {fn}(q, k, v, torch.ones(4), torch.ones(4))
assert out.shape == (2, 3, 5, 6)""",
        },
        {
            "name": 'numerical reference',
            "code": """import torch
torch.manual_seed(2)
q, k, v = [torch.randn(1, 2, 4, 8) for _ in range(3)]
gq, gk = torch.randn(8), torch.randn(8)
out = {fn}(q, k, v, gq, gk)
qn = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * gq
kn = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6) * gk
ref = torch.softmax(qn @ kn.transpose(-2, -1) / (8 ** 0.5), -1) @ v
assert torch.allclose(out, ref, atol=1e-6)""",
        },
        {
            "name": 'zero q edge case',
            "code": """import torch
q = torch.zeros(1, 1, 2, 3)
k = torch.randn(1, 1, 4, 3)
v = torch.randn(1, 1, 4, 5)
out = {fn}(q, k, v, torch.ones(3), torch.ones(3))
ref = v.mean(dim=-2, keepdim=True).expand(1, 1, 2, 5)
assert torch.allclose(out, ref, atol=1e-6) and torch.isfinite(out).all()""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
q = torch.randn(1, 2, 3, 4, requires_grad=True)
k = torch.randn(1, 2, 3, 4, requires_grad=True)
v = torch.randn(1, 2, 3, 4, requires_grad=True)
gq = torch.ones(4, requires_grad=True)
gk = torch.ones(4, requires_grad=True)
{fn}(q, k, v, gq, gk).square().sum().backward()
assert all(t.grad is not None for t in (q, k, v, gq, gk))""",
        },
    ],
    'solution': '''def qk_norm_attention(q, k, v, gamma_q, gamma_k, eps=1e-06):
    """QK-Norm 注意力机制 (QK-Norm Scaled Dot-Product Attention)。"""
    # Step 1: 沿头维分别计算 Q、K 的均方根并施加独立可学习缩放
    D = q.size(-1)
    q_norm = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + eps) * gamma_q
    k_norm = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + eps) * gamma_k
    # Step 2: 由归一化后的 QK 点积构造缩放注意力 logits 并做 softmax
    weights = torch.softmax(q_norm @ k_norm.transpose(-2, -1) / D ** 0.5, dim=-1)
    # Step 3: 用每个查询位置的注意力分布对 V 加权求和
    return weights @ v''',
    'demo': """torch.manual_seed(42)
q = torch.randn(1, 2, 3, 8)
k = torch.randn(1, 2, 5, 8)
v = torch.randn(1, 2, 5, 6)
gamma_q, gamma_k = torch.ones(8), torch.linspace(0.5, 1.5, 8)
out = qk_norm_attention(q, k, v, gamma_q, gamma_k)
print('Q/K/V -> output shapes:', q.shape, k.shape, v.shape, out.shape)
print('Output RMS per query:', out.square().mean(dim=-1).sqrt())
print('Finite output:', torch.isfinite(out).all().item())""",
    'explanation': r"""### 核心思路
先对每个注意力头、每个位置的 query/key 向量做 RMSNorm：$\hat q=q/\sqrt{\operatorname{mean}(q^2)+\epsilon}\odot\gamma_q$，$\hat k=k/\sqrt{\operatorname{mean}(k^2)+\epsilon}\odot\gamma_k$。再计算 $Y=\operatorname{softmax}(\hat Q\hat K^\top/\sqrt D)V$。归一化控制 Q/K 模长，独立的 $\gamma_q,\gamma_k$ 保留逐特征可学习尺度。

### 代码步骤拆解
- **Step 1（QK 独立 RMSNorm）**：对形状 `(B,H,S_q,D)` 与 `(B,H,S_k,D)` 的最后一维求平方均值，`gamma_q/gamma_k` 的 `(D,)` 形状广播到所有批次、头和位置。
- **Step 2（形成注意力分布）**：`q_norm @ k_norm.transpose(-2,-1)` 得到 `(B,H,S_q,S_k)`，除以 $\sqrt D$ 后在键序列轴做 softmax。
- **Step 3（聚合 Value）**：权重矩阵乘形状 `(B,H,S_k,D_v)` 的 `v`，输出 `(B,H,S_q,D_v)`。

### 重点与易错点分析
- **注意点 1（不是 L2 单位归一化）**：RMS 使用 $\sqrt{\sum x_i^2/D}$，因此未缩放向量的 L2 范数约为 $\sqrt D$；若误用 `torch.norm`，logit 尺度会额外缩小。
- **注意点 2（Q 与 K 参数独立）**：`gamma_q` 和 `gamma_k` 不能共享或错配；它们共同决定双线性点积中的逐特征权重，`v` 不参与 QK-Norm。
- **注意点 3（零向量行为）**：`eps` 使全零 Q 的归一化结果仍为零，此时所有 key logits 相等，输出正好是 V 在键轴上的均值。

### 复杂度分析
- **时间复杂度**：归一化为 $O(BH(S_q+S_k)D)$，两次注意力矩阵乘法为 $O(BHS_qS_k(D+D_v))$，后者主导。
- **空间复杂度**：显式注意力权重占 $O(BHS_qS_k)$，归一化后的 Q/K 占 $O(BH(S_q+S_k)D)$。""",
}
