"""Grouped Query Attention task."""

TASK = {
    'title': 'Grouped Query Attention',
    'title_zh': '分组查询注意力（GQA）',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'GroupQueryAttention',
    'starter_code': """class GroupQueryAttention(nn.Module):
    pass""",
    'hint': """1. `W_q` → `(B,H,S,d_k)`, `W_k`/`W_v` → `(B,KV,S,d_k)`
2. `K = K.repeat_interleave(H//KV, dim=1)` to expand KV heads
3. Scaled dot-product attn → reshape to `(B,S,d_model)`""",
    'hint_zh': """1. `W_q` → `(B,H,S,d_k)`，`W_k`/`W_v` → `(B,KV,S,d_k)`
2. `K = K.repeat_interleave(H//KV, dim=1)` 扩展 KV 头
3. 缩放点积注意力 → reshape 为 `(B,S,d_model)`""",
    'description_en': """Implement Grouped Query Attention (GQA).

GQA uses fewer KV heads than query heads, sharing each KV head across a group of query heads. This reduces KV cache size while preserving quality.

**Signature:** `GroupQueryAttention(d_model, num_heads, num_kv_heads)`

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, S, d_model)

**Returns:** attention output (B, S, d_model)

**Constraints:**
- W_k/W_v project to `num_kv_heads * d_k` dimensions
- Expand KV heads with `repeat_interleave` to match query heads
- Degenerates to standard MHA when `num_kv_heads == num_heads`""",
    'description_zh': """实现分组查询注意力（GQA）。

GQA 使用比查询头更少的 KV 头，每个 KV 头在一组查询头之间共享，在保持质量的同时减少 KV 缓存大小。

**签名:** `GroupQueryAttention(d_model, num_heads, num_kv_heads)`

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, S, d_model)

**返回:** 注意力输出 (B, S, d_model)

**约束:**
- W_k/W_v 投影到 `num_kv_heads * d_k` 维
- 使用 `repeat_interleave` 扩展 KV 头以匹配查询头数
- 当 `num_kv_heads == num_heads` 时退化为标准 MHA""",
    'theory_en': r"""Grouped Query Attention reduces KV cache memory by sharing KV heads across groups of query heads.

**Motivation:**
In standard MHA with $H$ heads, KV cache stores $2 \times B \times S \times H \times d_k$ values.
GQA with $K$ KV heads ($K < H$) stores only $2 \times B \times S \times K \times d_k$.

**Memory Reduction:**
$$\text{Compression ratio} = \frac{K}{H}$$

**Trade-off:**
- $K=1$: Maximum Memory Multi-Query Attention (MQA) — fastest, slightly lower quality
- $1 < K < H$: GQA — balanced
- $K=H$: Standard MHA — no compression

KV heads are repeated $\frac{H}{K}$ times via `repeat_interleave` before attention computation.""",
    'theory_zh': r"""分组查询注意力通过在查询头组之间共享 KV 头来减少 KV 缓存内存。

**动机：**
标准 MHA 有 $H$ 个头时，KV 缓存存储 $2 \times B \times S \times H \times d_k$ 个值。
GQA 使用 $K$ 个 KV 头（$K < H$）只存储 $2 \times B \times S \times K \times d_k$。

**内存压缩比：**
$$\text{压缩比} = \frac{K}{H}$$

**权衡：**
- $K=1$：最大内存节省的多查询注意力（MQA）——最快，质量略降
- $1 < K < H$：GQA —— 平衡
- $K=H$：标准 MHA —— 无压缩

KV 头通过 `repeat_interleave` 重复 $\frac{H}{K}$ 次后参与注意力计算。""",
    'diagram_en': """```mermaid
flowchart LR
    Q[Q] -->|W_q| QH[Query heads<br/>H heads]
    K[K] -->|W_k| KH[KV heads<br/>K heads]
    V[V] -->|W_v| VH[KV heads<br/>K heads]
    KH -->|repeat_interleave<br/>H/K times| KHE[Expanded K heads]
    VH -->|repeat_interleave<br/>H/K times| VHE[Expanded V heads]
    QH -->|attention| OUT[Output]
    KHE --> OUT
    VHE --> OUT
```""",
    'diagram_zh': """```mermaid
flowchart LR
    Q[Q] -->|W_q| QH[查询头<br/>H 个头]
    K[K] -->|W_k| KH[KV 头<br/>K 个头]
    V[V] -->|W_v| VH[KV 头<br/>K 个头]
    KH -->|repeat_interleave<br/>H/K 次| KHE[扩展 K 头]
    VH -->|repeat_interleave<br/>H/K 次| VHE[扩展 V 头]
    QH -->|注意力| OUT[输出]
    KHE --> OUT
    VHE --> OUT
```""",
    "tests": [
        {
            "name": 'Is nn.Module',
            "code": """import torch, torch.nn as nn
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=2)
assert isinstance(gqa, nn.Module), 'GroupQueryAttention should inherit from nn.Module'""",
        },
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
out = gqa.forward(torch.randn(2, 6, 32))
assert out.shape == (2, 6, 32), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": 'nn.Linear with correct shapes',
            "code": """import torch, torch.nn as nn
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
d_k = 32 // 8
assert isinstance((((((((gqa.W_q if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q), nn.Linear) and (((((((gqa.W_q if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q).weight.shape == (32, 32), f'W_q wrong'
assert isinstance((((((((gqa.W_k if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k), nn.Linear) and (((((((gqa.W_k if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k).weight.shape == (2 * d_k, 32), f"W_k shape: {(((((((gqa.W_k if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k).weight.shape}"
assert isinstance((((((((gqa.W_v if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v), nn.Linear) and (((((((gqa.W_v if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v).weight.shape == (2 * d_k, 32), f"W_v shape: {(((((((gqa.W_v if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v) if hasattr(gqa, 'W_v') else gqa.w_v).weight.shape}"
assert isinstance((((((((gqa.W_o if hasattr(gqa, 'W_o') else gqa.w_o) if hasattr(gqa, 'W_o') else gqa.w_o) if hasattr(gqa, 'W_o') else gqa.w_o) if hasattr(gqa, 'W_o') else gqa.w_o) if hasattr(gqa, 'W_o') else gqa.w_o) if hasattr(gqa, 'W_o') else gqa.w_o) if hasattr(gqa, 'W_o') else gqa.w_o), nn.Linear), 'W_o should be nn.Linear'""",
        },
        {
            "name": 'Degenerates to MHA when kv_heads == heads',
            "code": """import torch
torch.manual_seed(42)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=4)
out = gqa.forward(torch.randn(1, 4, 16))
assert out.shape == (1, 4, 16)
assert (((((((gqa.W_k if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k).weight.shape == (16, 16), 'Full KV when kv_heads == heads'""",
        },
        {
            "name": 'KV heads are shared correctly',
            "code": """import torch
torch.manual_seed(0)
D, H, KV = 16, 4, 2
d_k = D // H
gqa = {fn}(d_model=D, num_heads=H, num_kv_heads=KV)
x = torch.randn(1, 4, D)
k = (((((((gqa.W_k if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k)(x).view(1, 4, KV, d_k).transpose(1, 2)
k_exp = k.repeat_interleave(H // KV, dim=1)
assert torch.equal(k_exp[:, 0], k_exp[:, 1]), 'Heads 0,1 should share same K'
assert not torch.equal(k_exp[:, 0], k_exp[:, 2]), 'Different groups need different K'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(0)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
gqa.forward(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
assert (((((((gqa.W_q if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q) if hasattr(gqa, 'W_q') else gqa.w_q).weight.grad is not None and (((((((gqa.W_k if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k) if hasattr(gqa, 'W_k') else gqa.w_k).weight.grad is not None, 'Missing weight gradients'""",
        },
    ],
    'solution': '''class GroupQueryAttention(nn.Module):
    """实现分组查询注意力 (Grouped Query Attention)。"""

    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        # Step 1: 构造查询、共享键值和输出投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape
        # Step 2: 划分查询头并扩展共享 KV 头
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        # Step 3: 计算缩放点积注意力并合并多头
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(context)''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 6, 32
attention = GroupQueryAttention(D, 8, 2)
out = attention(torch.randn(B, S, D))
print("Output shape:", out.shape)
print("KV heads:", attention.num_kv_heads)
print("Sample values:", out[0, 0, :4])""",
    'explanation': """### 核心思路
GQA 为 $H$ 个查询头仅生成 $H_{kv}$ 组键值头，再重复每个 KV 头 $H/H_{kv}$ 次后执行标准缩放点积注意力。

### 代码步骤拆解
- **Step 1（投影构造）**：查询输出维为 `D`，键和值仅输出 `num_kv_heads * d_k`。
- **Step 2（头部共享）**：张量变为 `(B, H, S, d_k)`，`repeat_interleave` 将 KV 头扩展到查询头数。
- **Step 3（注意力合并）**：上下文转回 `(B, S, D)` 后通过 `W_o`。

### 重点与易错点分析
- **注意点 1（整除关系）**：`d_model` 应被 `num_heads` 整除，`num_heads` 也应被 `num_kv_heads` 整除。
- **注意点 2（重复顺序）**：`repeat_interleave` 让相邻查询头共享同一 KV 头，不能用普通 `repeat` 替代。
- **注意点 3（连续布局）**：转置后先 `.contiguous()` 再 `view`，避免非连续内存导致错误。

### 复杂度分析
- **时间复杂度**：$O(BHS^2d_k+BSD^2)$，注意力得分计算占主导。
- **空间复杂度**：$O(BHS^2+BHSd_k)$，显式扩展后的 KV 与权重占空间。""",
}
