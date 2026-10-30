"""Flash Attention (Tiled) task."""

TASK = {
    'title': 'Flash Attention (Tiled)',
    'title_zh': 'Flash Attention（分块）',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'flash_attention',
    'starter_code': """def flash_attention(Q, K, V, block_size=32):
    pass""",
    'hint': """Process Q in tiles. For each Q-block, iterate over K/V blocks:
1. `block_max = scores.max(dim=-1)` → `new_max = max(row_max, block_max)`
2. `correction = exp(row_max - new_max)` → rescale `acc` and `row_sum`
3. `acc += exp(scores - new_max) @ V_block`
4. `output = acc / row_sum`""",
    'hint_zh': """分块处理 Q。对每个 Q 块，遍历 K/V 块：
1. `block_max = scores.max(dim=-1)` → `new_max = max(row_max, block_max)`
2. `correction = exp(row_max - new_max)` → 重新缩放 `acc` 和 `row_sum`
3. `acc += exp(scores - new_max) @ V_block`
4. `output = acc / row_sum`""",
    'description_en': """Implement tiled (flash) attention with online softmax.

Flash attention processes Q/K/V in blocks to reduce memory usage, using the online softmax trick to maintain numerical correctness across tiles.

**Signature:** `flash_attention(Q, K, V, block_size=32) -> Tensor`

**Parameters:**
- `Q`, `K`, `V` — input tensors (B, S, D)
- `block_size` — tile size for blocking

**Returns:** attention output (B, S, D), identical to standard attention

**Constraints:**
- Must match standard softmax attention numerically
- Handle non-aligned sequence lengths (S not divisible by block_size)
- Result must be invariant to block_size choice""",
    'description_zh': """实现分块（Flash）注意力与在线 softmax。

Flash 注意力将 Q/K/V 分块处理以减少内存使用，利用在线 softmax 技巧在分块间保持数值正确性。

**签名:** `flash_attention(Q, K, V, block_size=32) -> Tensor`

**参数:**
- `Q`, `K`, `V` — 输入张量 (B, S, D)
- `block_size` — 分块大小

**返回:** 注意力输出 (B, S, D)，与标准注意力数值一致

**约束:**
- 必须与标准 softmax 注意力数值匹配
- 处理非对齐序列长度（S 不能被 block_size 整除）
- 结果不受 block_size 选择影响""",
    "tests": [
        {
            "name": 'Matches standard attention',
            "code": """import torch, math
torch.manual_seed(0)
B, S, D = 2, 16, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V, block_size=4)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
assert torch.allclose(out, ref, atol=1e-4), f'Max diff: {(out-ref).abs().max():.6f}'""",
        },
        {
            "name": 'Non-aligned block size',
            "code": """import torch, math
torch.manual_seed(42)
B, S, D = 1, 7, 4
Q, K, V = torch.randn(B,S,D), torch.randn(B,S,D), torch.randn(B,S,D)
out = {fn}(Q, K, V, block_size=3)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
assert torch.allclose(out, ref, atol=1e-4), 'Mismatch with non-aligned block size'""",
        },
        {
            "name": 'Block size invariant',
            "code": """import torch
torch.manual_seed(0)
Q, K, V = torch.randn(1,12,8), torch.randn(1,12,8), torch.randn(1,12,8)
out4 = {fn}(Q, K, V, block_size=4)
out6 = {fn}(Q, K, V, block_size=6)
assert torch.allclose(out4, out6, atol=1e-4), 'Different block sizes should give same result'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
Q = torch.randn(1, 8, 4, requires_grad=True)
K = torch.randn(1, 8, 4, requires_grad=True)
V = torch.randn(1, 8, 4, requires_grad=True)
{fn}(Q, K, V, block_size=4).sum().backward()
assert Q.grad is not None, 'Q.grad is None'""",
        },
    ],
    'solution': '''def flash_attention(Q, K, V, block_size=32):
    """实现分块在线注意力 (Tiled FlashAttention)。"""
    B, S, D = Q.shape
    scale = D ** -0.5
    output_blocks = []

    # Step 1: 按查询序列切块，分别维护在线 softmax 状态
    for q_start in range(0, S, block_size):
        query = Q[:, q_start:q_start + block_size]
        S_q = query.shape[1]
        row_max = torch.full((B, S_q, 1), -torch.inf, device=Q.device, dtype=Q.dtype)
        row_sum = torch.zeros((B, S_q, 1), device=Q.device, dtype=Q.dtype)
        accumulator = torch.zeros((B, S_q, D), device=Q.device, dtype=Q.dtype)

        # Step 2: 遍历键值块，并在新最大值基准下合并 softmax 统计量
        for kv_start in range(0, S, block_size):
            key = K[:, kv_start:kv_start + block_size]
            value = V[:, kv_start:kv_start + block_size]
            scores = torch.bmm(query, key.transpose(1, 2)) * scale
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, block_max)
            correction = torch.exp(row_max - new_max)
            exp_scores = torch.exp(scores - new_max)
            accumulator = accumulator * correction + torch.bmm(exp_scores, value)
            row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
            row_max = new_max

        # Step 3: 归一化当前查询块并拼接完整输出
        output_blocks.append(accumulator / row_sum)
    return torch.cat(output_blocks, dim=1)''',
    'demo': """torch.manual_seed(42)
B, S, D = 1, 16, 8
Q, K, V = [torch.randn(B, S, D) for _ in range(3)]
out = flash_attention(Q, K, V, block_size=4)
reference = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(D), dim=-1) @ V
print("Output shape:", out.shape)
print("Max difference:", (out - reference).abs().max().item())""",
    'explanation': r"""### 核心思路
FlashAttention 将 $QK^\top$ 分块计算，并为每个查询行在线维护最大值 `row_max`、指数和 `row_sum` 与加权值 `accumulator`，无需物化完整注意力矩阵。

### 代码步骤拆解
- **Step 1（查询分块）**：每个 `(B, S_q, D)` 查询块初始化在线 softmax 的三组状态。
- **Step 2（键值流式合并）**：每遇到新块都用 `new_max` 重标定旧统计量，再累加当前块指数权重和价值结果。
- **Step 3（块归一化）**：所有键值块处理完后计算 `accumulator / row_sum`，并沿序列维拼接。

### 重点与易错点分析
- **注意点 1（数值稳定性）**：最大值变化时必须用 `exp(row_max - new_max)` 同时缩放旧分子和旧分母，否则不同块无法正确合并。
- **注意点 2（边界块）**：切片自然处理不能被 `block_size` 整除的尾块，状态形状应使用实际 `S_q`。
- **注意点 3（显存折中）**：算法把注意力显存从 $O(BS^2)$ 降到块级规模，但 Python 循环版本主要用于展示原理而非替代融合内核。

### 复杂度分析
- **时间复杂度**：$O(BS^2D)$，与标准注意力相同，瓶颈仍是所有查询—键点积。
- **空间复杂度**：$O(BSD+Bb^2)$，其中 $b$ 为块大小，不保存完整 $S\times S$ 注意力图。""",
}
