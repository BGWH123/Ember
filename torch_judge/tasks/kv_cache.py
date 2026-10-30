"""KV Cache Attention task."""

TASK = {
    'title': 'KV Cache Attention',
    'title_zh': 'KV Cache 注意力',
    'difficulty': 'Hard',
    'category': '推理优化',
    'function_name': 'KVCacheAttention',
    'starter_code': """class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        pass

    def forward(self, x, cache=None):
        pass""",
    'hint': """1. Project Q/K/V → reshape to `(B, num_heads, S, d_k)`
2. If cache: K = cat([cache_K, K], dim=2),  V = cat([cache_V, V], dim=2)
3. Apply causal mask when S_new > 1 (prefill)
4. Scaled dot-product attention → output projection
5. Return (output, (K_all, V_all))""",
    'hint_zh': """1. 投影 Q/K/V → reshape 为 `(B, num_heads, S, d_k)`
2. 有缓存时：K = cat([cache_K, K], dim=2),  V = cat([cache_V, V], dim=2)
3. S_new > 1（预填充）时应用因果掩码
4. 缩放点积注意力 → 输出投影
5. 返回 (output, (K_all, V_all))""",
    'description_en': """Implement multi-head attention with KV cache for efficient autoregressive decoding.

KV caching stores previously computed keys and values, so each decode step only processes the new token instead of the full sequence.

**Signature:** `KVCacheAttention(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x, cache=None) -> (Tensor, tuple)`
- `x` — input tensor (B, S_new, d_model)
- `cache` — optional (K, V) tuple from previous steps

**Returns:** (output, (K_all, V_all)) where cache grows each step

**Constraints:**
- Concatenate new K/V with cached along sequence dim
- Apply causal mask during prefill (S_new > 1)
- Incremental decode must match full forward numerically""",
    'description_zh': """实现带 KV 缓存的多头注意力，用于高效自回归解码。

KV 缓存存储之前计算的键和值，使每个解码步骤只需处理新 token 而非完整序列。

**签名:** `KVCacheAttention(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x, cache=None) -> (Tensor, tuple)`
- `x` — 输入张量 (B, S_new, d_model)
- `cache` — 可选的前序步骤 (K, V) 元组

**返回:** (输出, (K_all, V_all))，缓存逐步增长

**约束:**
- 将新 K/V 与缓存沿序列维度拼接
- 预填充时（S_new > 1）应用因果掩码
- 增量解码必须与完整前向传播数值一致""",
    "tests": [
        {
            "name": 'Output shape (no cache)',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
attn = {fn}(d_model=64, num_heads=4)
assert isinstance(attn, nn.Module), 'Must inherit from nn.Module'
x = torch.randn(2, 8, 64)
out, cache = attn(x)
assert out.shape == (2, 8, 64), f'Output shape: {out.shape}'""",
        },
        {
            "name": 'Cache structure',
            "code": """import torch
torch.manual_seed(0)
attn = {fn}(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)
out, cache = attn(x)
assert isinstance(cache, tuple) and len(cache) == 2, 'Cache must be a (K, V) tuple'
k_cache, v_cache = cache
assert k_cache.shape == (2, 4, 8, 16), f'K cache shape: {k_cache.shape}, expected (2, 4, 8, 16)'
assert v_cache.shape == (2, 4, 8, 16), f'V cache shape: {v_cache.shape}, expected (2, 4, 8, 16)'""",
        },
        {
            "name": 'Decode step appends to cache',
            "code": """import torch
torch.manual_seed(0)
attn = {fn}(d_model=32, num_heads=2)
x = torch.randn(1, 4, 32)
_, cache = attn(x)
new_token = torch.randn(1, 1, 32)
out, new_cache = attn(new_token, cache=cache)
assert out.shape == (1, 1, 32), f'Decode output shape: {out.shape}'
k_cache, v_cache = new_cache
assert k_cache.shape[2] == 5, f'Cache should grow: K has {k_cache.shape[2]} positions, expected 5'
assert v_cache.shape[2] == 5, f'Cache should grow: V has {v_cache.shape[2]} positions, expected 5'""",
        },
        {
            "name": 'Incremental decode matches full forward',
            "code": """import torch
torch.manual_seed(42)
attn = {fn}(d_model=32, num_heads=2)
x = torch.randn(1, 6, 32)
full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)
assert torch.allclose(full_out, inc_out, atol=1e-5), 'Incremental decode must match full forward'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(0)
attn = {fn}(d_model=32, num_heads=2)
x = torch.randn(1, 4, 32, requires_grad=True)
out, cache = attn(x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in attn.parameters())
n_grad = sum(1 for p in attn.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'""",
        },
    ],
    'solution': '''class KVCacheAttention(nn.Module):
    """实现带键值缓存的多头注意力 (KV Cache Attention)。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None):
        B, S_new, _ = x.shape

        # Step 1: 投影并拆分查询、键和值到多个注意力头
        q = self.W_q(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)

        # Step 2: 沿序列维拼接历史缓存，并构造预填充因果掩码
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = (k, v)
        S_total = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if S_new > 1:
            S_past = S_total - S_new
            causal_mask = torch.triu(
                torch.ones(S_new, S_total, dtype=torch.bool, device=x.device),
                diagonal=S_past + 1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Step 3: 聚合值向量、合并多头并完成输出投影
        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(B, S_new, -1)
        out = self.W_o(context)
        return out, new_cache''',
    'demo': """torch.manual_seed(42)
attention = KVCacheAttention(d_model=32, num_heads=4)
x = torch.randn(1, 6, 32)
full_out, _ = attention(x)
prefix_out, cache = attention(x[:, :4])
tail_out, cache = attention(x[:, 4:], cache=cache)
incremental_out = torch.cat((prefix_out, tail_out), dim=1)
print("Output shape:", full_out.shape)
print("Cache shape:", cache[0].shape)
print("Incremental match:", torch.allclose(full_out, incremental_out, atol=1e-5))""",
    'explanation': r"""### 核心思路
KV Cache 只为新 token 计算投影，并把新 `k`、`v` 与历史缓存沿序列维拼接；注意力仍按 $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ 计算。输出保持 `(B, S_new, d_model)`，缓存增长为 `(B, H, S_total, d_k)`。

### 代码步骤拆解
- **Step 1（多头投影）**：将 `x` 投影并重排为 `(B, num_heads, S_new, d_k)` 的 `q`、`k`、`v`。
- **Step 2（缓存与掩码）**：将历史缓存接到新键值之前，并依据 `S_past` 构造 `(S_new, S_total)` 因果掩码。
- **Step 3（注意力输出）**：对最后一维做 softmax，聚合 `v` 后合并多头并通过 `W_o`。

### 重点与易错点分析
- **注意点 1（掩码偏移）**：带缓存的多 token 输入不能使用普通主对角线；对角线必须偏移 `S_past + 1` 才能允许查询看到全部历史。
- **注意点 2（单步解码）**：`S_new == 1` 时唯一查询位于序列末尾，可直接看到全部缓存，无需额外掩码。
- **注意点 3（缓存梯度与显存）**：训练时缓存会保留计算图；真实推理通常在无梯度环境使用，以免历史激活持续占用显存。

### 复杂度分析
- **时间复杂度**：$O(BHS_{new}S_{total}d_k)$，瓶颈是查询与全部缓存键的点积。
- **空间复杂度**：$O(BHS_{total}d_k + BHS_{new}S_{total})$，分别对应 KV 缓存和注意力矩阵。""",
}
