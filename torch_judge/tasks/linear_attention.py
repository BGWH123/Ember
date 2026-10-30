"""Linear Self-Attention task."""

TASK = {
    'title': 'Linear Self-Attention',
    'title_zh': '线性自注意力',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'linear_attention',
    'starter_code': """def linear_attention(Q, K, V):
    pass""",
    'hint': """`phi(x) = elu(x) + 1`. Let `KV = phi(K).transpose(-2, -1) @ V` and `Z = phi(K).sum(dim=-2, keepdim=True)`. Output = `(phi(Q) @ KV) / (phi(Q) @ Z.transpose(-2, -1) + 1e-6)`.""",
    'hint_zh': """`phi(x) = elu(x) + 1`。令 `KV = phi(K).transpose(-2, -1) @ V`，`Z = phi(K).sum(dim=-2, keepdim=True)`。输出 = `(phi(Q) @ KV) / (phi(Q) @ Z.transpose(-2, -1) + 1e-6)`。""",
    'description_en': """Implement linear self-attention with kernel feature maps.

Linear attention replaces softmax with a feature map phi, enabling O(S*D^2) complexity instead of O(S^2*D) by reordering the computation.

**Signature:** `linear_attention(Q, K, V) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S, D_k)
- `K` — key tensor (B, S, D_k)
- `V` — value tensor (B, S, D_v)

**Returns:** attention output (B, S, D_v)

**Constraints:**
- Feature map: `phi(x) = elu(x) + 1`
- Compute `phi(Q) @ (phi(K)^T @ V)` not `softmax(Q @ K^T) @ V`
- Normalize by `phi(Q) @ sum(phi(K))` (add `eps=1e-6` for numerical stability)""",
    'description_zh': """实现基于核特征映射的线性自注意力。

线性注意力用特征映射 phi 替代 softmax，通过重排计算顺序将复杂度从 O(S^2*D) 降至 O(S*D^2)。

**签名:** `linear_attention(Q, K, V) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S, D_k)
- `K` — 键张量 (B, S, D_k)
- `V` — 值张量 (B, S, D_v)

**返回:** 注意力输出 (B, S, D_v)

**约束:**
- 特征映射：`phi(x) = elu(x) + 1`
- 计算 `phi(Q) @ (phi(K)^T @ V)` 而非 `softmax(Q @ K^T) @ V`
- 通过 `phi(Q) @ sum(phi(K))` 归一化（加 `eps=1e-6` 保证数值稳定）""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": 'No NaN or Inf',
            "code": """import torch
torch.manual_seed(0)
out = {fn}(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8))
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
Q = torch.randn(1, 4, 8, requires_grad=True)
K = torch.randn(1, 4, 8, requires_grad=True)
V = torch.randn(1, 4, 8, requires_grad=True)
{fn}(Q, K, V).sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'""",
        },
        {
            "name": 'Runs fast on long sequences (linear complexity)',
            "code": """import torch, time
torch.manual_seed(0)
Q = torch.randn(1, 2048, 64)
K = torch.randn(1, 2048, 64)
V = torch.randn(1, 2048, 64)
t0 = time.perf_counter()
for _ in range(10):
    {fn}(Q, K, V)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s — should be O(S*D^2) not O(S^2*D)'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch
torch.manual_seed(7)
B, S, D = 1, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
# Reference: phi(x) = elu(x) + 1, elu(x) = x if x>0 else exp(x)-1
def elu_plus_1(x):
    return torch.where(x > 0, x + 1, x.exp())
phi_Q = elu_plus_1(Q)
phi_K = elu_plus_1(K)
KV = phi_K.transpose(-2, -1) @ V          # (B, D, D_v)
Z = phi_K.sum(dim=1, keepdim=True)        # (B, 1, D)
num = phi_Q @ KV                           # (B, S, D_v)
den = phi_Q @ Z.transpose(-2, -1)         # (B, S, 1)
expected = num / (den + 1e-6)
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff {(out - expected).abs().max()}'""",
        },
    ],
    'solution': '''def linear_attention(Q, K, V):
    """实现核特征映射线性注意力 (Linear Attention)。"""
    # Step 1: 使用 ELU 加一得到非负查询与键特征
    q_features = torch.where(Q > 0, Q + 1.0, torch.exp(Q))
    k_features = torch.where(K > 0, K + 1.0, torch.exp(K))

    # Step 2: 先沿序列维聚合键值与键总量
    kv_summary = k_features.transpose(1, 2) @ V
    key_sum = k_features.sum(dim=1, keepdim=True)

    # Step 3: 计算分子与归一化分母
    numerator = q_features @ kv_summary
    denominator = q_features @ key_sum.transpose(1, 2)
    out = numerator / (denominator + 1e-6)
    return out''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, 32)
out = linear_attention(Q, K, V)
print("Output shape:", out.shape)
print("Finite values:", torch.isfinite(out).all().item())
print("Sample values:", out[0, 0, :4])""",
    'explanation': r"""### 核心思路
线性注意力用 $\phi(x)=\operatorname{ELU}(x)+1$ 替代 softmax 核，并借助结合律先计算 `k_features.T @ V`。这样避免显式构造 `(S, S)` 注意力矩阵，输出为 `(B, S, D_v)`。

### 代码步骤拆解
- **Step 1（核映射）**：把 `Q`、`K` 映射为严格非负的 `q_features`、`k_features`。
- **Step 2（序列聚合）**：生成 `(B, D_k, D_v)` 的 `kv_summary` 和 `(B, 1, D_k)` 的 `key_sum`。
- **Step 3（归一化）**：用查询乘聚合量，并以对应核权重和加 `1e-6` 归一化。

### 重点与易错点分析
- **注意点 1（运算顺序）**：若先计算 `Q @ K.T` 就重新产生 $O(S^2)$ 时间与显存，失去线性注意力优势。
- **注意点 2（分母稳定性）**：非负核使分母通常为正，仍需加 `eps` 防止极端下溢导致除零。
- **注意点 3（非 softmax 等价）**：该核注意力只是替代形式，并不等于标准 softmax 注意力。

### 复杂度分析
- **时间复杂度**：$O(BS D_kD_v)$，主要瓶颈是键值摘要和查询投影。
- **空间复杂度**：$O(BD_kD_v + BSD_v)$，不保存平方级注意力矩阵。""",
}
