"""Multi-Head Latent Attention (MLA) task."""

TASK = {
    'title': 'Multi-Head Latent Attention (MLA)',
    'title_zh': '多头潜在注意力（MLA）',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'mla_attention',
    'starter_code': """def mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads):
    pass""",
    'hint': """1. `c_kv = X @ W_dkv`  → `(B,S,kv_rank)`
2. `K = c_kv @ W_uk`, `V = c_kv @ W_uv`, `Q = X @ W_q`
3. `.view(B,S,num_heads,D_h).transpose(1,2)` for each
4. Scaled dot-product attn → `.transpose(1,2).reshape(B,S,-1)`""",
    'hint_zh': """1. `c_kv = X @ W_dkv`  → `(B,S,kv_rank)`
2. `K = c_kv @ W_uk`，`V = c_kv @ W_uv`，`Q = X @ W_q`
3. 各自 `.view(B,S,num_heads,D_h).transpose(1,2)`
4. 缩放点积注意力 → `.transpose(1,2).reshape(B,S,-1)`""",
    'description_en': """Implement Multi-Head Latent Attention (MLA) from DeepSeek V2/V3.

MLA's key innovation: instead of caching full K and V tensors, compress them into a low-rank latent vector `c_kv`, then decompress on the fly. This dramatically reduces KV cache memory during inference.

**Signature:** `mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads) -> Tensor`

**Parameters:**
- `X` — input tensor (B, S, D)
- `W_dkv` — KV compression matrix (D, kv_rank)
- `W_uk` — K decompression matrix (kv_rank, num_heads * D_h)
- `W_uv` — V decompression matrix (kv_rank, num_heads * D_h)
- `W_q` — Q projection matrix (D, num_heads * D_h)
- `num_heads` — number of attention heads

**Returns:** output tensor (B, S, num_heads * D_h)

**Steps:**
1. Compress: `c_kv = X @ W_dkv` → (B, S, kv_rank)
2. Decompress: `K = c_kv @ W_uk`, `V = c_kv @ W_uv`
3. Project queries: `Q = X @ W_q`
4. Reshape all to (B, num_heads, S, D_h)
5. Scaled dot-product attention
6. Reshape output to (B, S, num_heads * D_h)""",
    'description_zh': """实现 DeepSeek V2/V3 中的多头潜在注意力（MLA）。

MLA 的核心创新：不缓存完整的 K 和 V 张量，而是将其压缩为低秩潜在向量 `c_kv`，推理时再即时解压。这大幅降低了推理时的 KV 缓存内存占用。

**签名:** `mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads) -> Tensor`

**参数:**
- `X` — 输入张量 (B, S, D)
- `W_dkv` — KV 压缩矩阵 (D, kv_rank)
- `W_uk` — K 解压矩阵 (kv_rank, num_heads * D_h)
- `W_uv` — V 解压矩阵 (kv_rank, num_heads * D_h)
- `W_q` — Q 投影矩阵 (D, num_heads * D_h)
- `num_heads` — 注意力头数

**返回:** 输出张量 (B, S, num_heads * D_h)

**步骤:**
1. 压缩：`c_kv = X @ W_dkv` → (B, S, kv_rank)
2. 解压：`K = c_kv @ W_uk`，`V = c_kv @ W_uv`
3. 投影查询：`Q = X @ W_q`
4. 重塑为 (B, num_heads, S, D_h)
5. 缩放点积注意力
6. 重塑输出为 (B, S, num_heads * D_h)""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, S, D, kv_rank, num_heads, D_h = 2, 6, 64, 16, 4, 8
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
assert out.shape == (B, S, num_heads * D_h), f'Expected ({B}, {S}, {num_heads * D_h}), got {out.shape}'""",
        },
        {
            "name": 'KV latent has correct compressed shape',
            "code": """import torch
torch.manual_seed(1)
B, S, D, kv_rank, num_heads, D_h = 1, 8, 32, 6, 2, 4
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
# Conceptual check: verify the compression dimensions are correct.
# This independently computes c_kv to validate the shape of the latent
# representation. The student's actual code path is authoritatively
# validated by the 'Numerical correctness' test (test 5).
c_kv = X @ W_dkv
assert c_kv.shape == (B, S, kv_rank), f'Latent shape should be ({B}, {S}, {kv_rank}), got {c_kv.shape}'
assert kv_rank < D, 'kv_rank should be smaller than D for compression'
assert out.shape == (B, S, num_heads * D_h), f'Output shape mismatch: {out.shape}'""",
        },
        {
            "name": 'Gradient flows through X and weights',
            "code": """import torch
torch.manual_seed(2)
B, S, D, kv_rank, num_heads, D_h = 1, 4, 16, 4, 2, 4
W_dkv = torch.randn(D, kv_rank, requires_grad=True)
W_uk  = torch.randn(kv_rank, num_heads * D_h, requires_grad=True)
W_uv  = torch.randn(kv_rank, num_heads * D_h, requires_grad=True)
W_q   = torch.randn(D, num_heads * D_h, requires_grad=True)
X = torch.randn(B, S, D, requires_grad=True)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
out.sum().backward()
assert X.grad is not None, 'Missing gradient for X'
assert W_dkv.grad is not None, 'Missing gradient for W_dkv'
assert W_uk.grad is not None, 'Missing gradient for W_uk'
assert W_uv.grad is not None, 'Missing gradient for W_uv'
assert W_q.grad is not None, 'Missing gradient for W_q'""",
        },
        {
            "name": 'kv_rank compression reduces memory',
            "code": """import torch
torch.manual_seed(3)
B, S, D, kv_rank, num_heads, D_h = 2, 10, 128, 8, 4, 16
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
# c_kv should be much smaller than full KV
c_kv = X @ W_dkv
full_kv_size = B * S * num_heads * D_h * 2  # K and V
compressed_size = B * S * kv_rank
assert c_kv.shape == (B, S, kv_rank), f'Wrong latent shape: {c_kv.shape}'
assert compressed_size < full_kv_size, 'Compressed KV should be smaller than full KV'
assert out.shape == (B, S, num_heads * D_h), f'Output shape mismatch: {out.shape}'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch
torch.manual_seed(9)
B, S, D, kv_rank, num_heads, D_h = 2, 5, 32, 8, 4, 8
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
# Reference computation
c_kv = X @ W_dkv                          # (B, S, kv_rank)
K = c_kv @ W_uk                            # (B, S, num_heads*D_h)
V = c_kv @ W_uv                            # (B, S, num_heads*D_h)
Q = X @ W_q                                # (B, S, num_heads*D_h)
def split_heads(t):
    return t.view(B, S, num_heads, D_h).transpose(1, 2)
Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
scale = D_h ** -0.5
attn = torch.softmax(Qh @ Kh.transpose(-2, -1) * scale, dim=-1)
expected = (attn @ Vh).transpose(1, 2).reshape(B, S, num_heads * D_h)
assert torch.allclose(out, expected, atol=1e-5), f'MLA numerical mismatch: max diff {(out - expected).abs().max()}'""",
        },
    ],
    'solution': '''def mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads):
    """实现多头潜在注意力 (Multi-Head Latent Attention)。"""
    B, S, _ = X.shape
    d_h = W_q.shape[1] // num_heads

    # Step 1: 压缩 KV 并分别恢复键、值，同时投影查询
    c_kv = X @ W_dkv
    k_projected = c_kv @ W_uk
    v_projected = c_kv @ W_uv
    q_projected = X @ W_q

    # Step 2: 将投影结果拆分为多头布局
    def split_heads(tensor):
        return tensor.view(B, S, num_heads, d_h).transpose(1, 2)

    q = split_heads(q_projected)
    k = split_heads(k_projected)
    v = split_heads(v_projected)

    # Step 3: 计算注意力并合并多头输出
    scale = d_h ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    context = attn_weights @ v
    out = context.transpose(1, 2).reshape(B, S, num_heads * d_h)
    return out''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 6, 32
num_heads, d_h, d_c = 4, 8, 8
X = torch.randn(B, S, D)
W_dkv = torch.randn(D, d_c)
W_uk = torch.randn(d_c, num_heads * d_h)
W_uv = torch.randn(d_c, num_heads * d_h)
W_q = torch.randn(D, num_heads * d_h)
out = mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads)
print("Output shape:", out.shape)
print("Latent shape:", (X @ W_dkv).shape)
print("Sample values:", out[0, 0, :4])""",
    'explanation': """### 核心思路
MLA 先把输入 KV 压缩到低秩 `c_kv`，再即时上投影为多头键值；查询独立投影后执行标准缩放点积注意力。缓存潜变量 `(B, S, d_c)` 可小于完整 KV。

### 代码步骤拆解
- **Step 1（低秩投影）**：由 `X @ W_dkv` 得到潜变量并恢复 `k_projected`、`v_projected`。
- **Step 2（多头拆分）**：把三个投影重排为 `(B, num_heads, S, d_h)`。
- **Step 3（注意力聚合）**：softmax 分数乘 `v`，再合并为 `(B, S, num_heads * d_h)`。

### 重点与易错点分析
- **注意点 1（维度推导）**：`d_h` 应由 `W_q.shape[1] // num_heads` 得到，投影输出维必须可整除。
- **注意点 2（缓存收益）**：只有推理缓存 `c_kv` 才体现显存优势；本实现为计算注意力仍即时展开 K/V。
- **注意点 3（缩放维度）**：缩放因子使用头维 $d_h$，不是输入模型维或压缩秩。

### 复杂度分析
- **时间复杂度**：$O(BSDd_c+BSd_cHd_h+BHS^2d_h)$，平方注意力通常占主导。
- **空间复杂度**：$O(BSd_c+BHS^2)$，分别为潜表示和注意力矩阵。""",
}
