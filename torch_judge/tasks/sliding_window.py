"""Sliding Window Attention task."""

TASK = {
    'title': 'Sliding Window Attention',
    'title_zh': '滑动窗口注意力',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'sliding_window_attention',
    'starter_code': """def sliding_window_attention(Q, K, V, window_size):
    pass""",
    'hint': """Standard attention + window mask. Build `|i-j|` matrix → mask positions where `|i-j| > window_size` with `-inf` → softmax → `@ V`.""",
    'hint_zh': '标准注意力 + 窗口遮蔽。构造 `|i-j|` 矩阵 → 将 `|i-j| > window_size` 的位置填 `-inf` → softmax → `@ V`。',
    'description_en': """Implement sliding window attention.

Sliding window attention restricts each position to attend only within a fixed window, reducing complexity for long sequences while maintaining local context.

**Signature:** `sliding_window_attention(Q, K, V, window_size) -> Tensor`

**Parameters:**
- `Q`, `K`, `V` — input tensors (B, S, D)
- `window_size` — each position attends to positions within |i-j| <= window_size

**Returns:** attention output (B, S, D)

**Constraints:**
- Mask positions where `|i - j| > window_size` with `-inf`
- `window_size=0` means each position only sees itself
- Large window equals full attention""",
    'description_zh': """实现滑动窗口注意力。

滑动窗口注意力限制每个位置只关注固定窗口内的位置，在保持局部上下文的同时降低长序列的复杂度。

**签名:** `sliding_window_attention(Q, K, V, window_size) -> Tensor`

**参数:**
- `Q`, `K`, `V` — 输入张量 (B, S, D)
- `window_size` — 每个位置关注 |i-j| <= window_size 范围内的位置

**返回:** 注意力输出 (B, S, D)

**约束:**
- 用 `-inf` 掩盖 `|i - j| > window_size` 的位置
- `window_size=0` 表示每个位置只能看到自身
- 大窗口等同于全注意力""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 16), window_size=2)
assert out.shape == (2, 8, 16), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": 'window_size=0 — only sees itself',
            "code": """import torch
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V, window_size=0)
assert torch.allclose(out, V, atol=1e-5), 'window=0: each position should output V[i]'""",
        },
        {
            "name": 'Large window equals full attention',
            "code": """import torch, math
torch.manual_seed(0)
B, S, D = 2, 6, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_win = {fn}(Q, K, V, window_size=S)
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
assert torch.allclose(out_win, ref, atol=1e-5), 'Large window should equal full attention'""",
        },
        {
            "name": "Distant tokens don't affect output",
            "code": """import torch
torch.manual_seed(0)
B, S, D = 1, 10, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V, window_size=1)
K2, V2 = K.clone(), V.clone()
K2[:, 5:] = torch.randn(B, 5, D)
V2[:, 5:] = torch.randn(B, 5, D)
out2 = {fn}(Q, K2, V2, window_size=1)
assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Distant tokens should not affect output'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
{fn}(Q, K, V, window_size=1).sum().backward()
assert Q.grad is not None, 'Q.grad is None'""",
        },
    ],
    'solution': '''def sliding_window_attention(Q, K, V, window_size):
    """实现滑动窗口注意力（Sliding Window Attention）。"""
    # Step 1: 计算所有查询与键之间的缩放点积分数
    B, S, D = Q.shape
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    # Step 2: 按位置距离屏蔽窗口外键并仅在局部邻域归一化
    positions = torch.arange(S, device=Q.device)
    mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    # Step 3: 用局部注意力权重加权值向量得到每个查询位置的表示
    return torch.bmm(weights, V)''',
    'demo': """torch.manual_seed(42)
q = torch.randn(1, 6, 8)
k = torch.randn(1, 6, 8)
v = torch.randn(1, 6, 8)
out = sliding_window_attention(q, k, v, window_size=1)
self_only = sliding_window_attention(q, k, v, window_size=0)
print("Window-1 output shape:", out.shape)
print("First local output:", out[0, 0, :3])
print("Window-0 equals V:", torch.allclose(self_only, v, atol=1e-6))""",
    'explanation': r"""### 核心思路
对位置 $i$，只允许键位置 $j$ 满足 $|i-j|\le w$。代码先计算 $s_{ij}=q_i^\top k_j/\sqrt{d_k}$，再把窗口外分数置为 $-\infty$，因此 softmax 后这些位置概率严格为 0，最终 $o_i=\sum_{|i-j|\le w}\alpha_{ij}v_j$。

### 代码步骤拆解
- **Step 1（缩放点积分数）**：`Q @ K.transpose(1, 2)` 从 $(B,S,D)$ 生成 $(B,S,S)$ 的 `scores`，除以 $\sqrt{d_k}$ 防止维度增大导致 softmax 过度饱和。
- **Step 2（施加双向局部窗口）**：`positions` 的两次扩维构成 $(S,S)$ 距离矩阵；距离大于 `window_size` 的格子填 $-\infty$，再沿键轴归一化。
- **Step 3（聚合局部值向量）**：`weights` 与 $(B,S,D)$ 的 `V` 做批矩阵乘，返回每个查询位置的 $(B,S,D)$ 局部上下文。

### 重点与易错点分析
- **注意点 1（mask 时机与方向）**：必须在 softmax 前屏蔽，并沿最后的键维归一化；`window_size=0` 只保留对角线，所以输出精确等于 `V`。
- **注意点 2（序列边界）**：首尾位置的可见 token 少于 $2w+1$，但 softmax 会在实际保留项上重新归一化；大于等于 $S-1$ 的窗口则等价于全注意力。
- **注意点 3（算法与实现复杂度差异）**：滑窗理论上可用带状 kernel 降到 $O(BSwD)$，但本题显式构造完整 `scores` 后再 mask，仍执行平方级矩阵乘与存储。

### 复杂度分析
- **时间复杂度**：当前稠密实现为 $O(BS^2D)$，完整的 `QK^\top` 和 `weights @ V` 是瓶颈。
- **空间复杂度**：$O(BS^2+BSD)$，平方级 `scores`/`weights` 主导长序列显存，布尔 mask 另占 $O(S^2)$。""",
}
