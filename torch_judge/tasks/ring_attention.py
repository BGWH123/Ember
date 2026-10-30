"""Ring Attention task."""

TASK = {
    'title': 'Ring Attention',
    'title_zh': '环形注意力',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'ring_attention',
    'starter_code': """def ring_attention(Q, K, V, num_devices):
    pass""",
    'hint': """Split Q/K/V into num_devices chunks. For each Q chunk i, loop over all K/V chunks (ring order).
Online softmax per step:
  m_new = max(m, scores.max)
  l_new = l·exp(m-m_new) + Σexp(scores-m_new)
  o = (o·l·exp(m-m_new) + exp(scores-m_new)@V) / l_new""",
    'hint_zh': """将 Q/K/V 分成 num_devices 块。对每个 Q 块 i，遍历所有 K/V 块（环形顺序）。
每步 online softmax：
  m_new = max(m, scores.max)
  l_new = l·exp(m-m_new) + Σexp(scores-m_new)
  o = (o·l·exp(m-m_new) + exp(scores-m_new)@V) / l_new""",
    'description_en': """Implement ring attention (single-device simulation).

Ring attention enables sequence parallelism by partitioning Q/K/V across devices in a ring. Each device holds a Q chunk and rotates K/V chunks around the ring, accumulating attention outputs using online softmax.

**Signature:** `ring_attention(Q, K, V, num_devices) -> Tensor`

**Parameters:**
- `Q, K, V` — tensors of shape `(B, S, D)`, S divisible by num_devices
- `num_devices` — number of virtual ring participants

**Returns:** output tensor `(B, S, D)`, numerically equivalent to standard attention

**Algorithm:**
1. Split Q, K, V into `num_devices` chunks along S
2. For each Q chunk (device i), iterate over all K/V chunks in ring order
3. Accumulate using online softmax (track running max and sum)
4. Reassemble output chunks""",
    'description_zh': """实现环形注意力（单机模拟）。

环形注意力通过将 Q/K/V 分布在环形拓扑的设备上实现序列并行。每个设备持有一个 Q 分块，K/V 分块在环上轮转，用 online softmax 累加注意力输出。

**签名:** `ring_attention(Q, K, V, num_devices) -> Tensor`

**参数:**
- `Q, K, V` — 形状 `(B, S, D)` 的张量，S 可被 num_devices 整除
- `num_devices` — 虚拟环参与者数量

**返回:** 输出张量 `(B, S, D)`，数值上等价于标准注意力

**算法:**
1. 沿 S 维将 Q, K, V 分成 `num_devices` 个分块
2. 对每个 Q 分块（设备 i），遍历所有 K/V 分块（环形顺序）
3. 用 online softmax 累加（跟踪运行最大值和归一化因子）
4. 重新拼接输出分块""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, S, D = 2, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V, num_devices=2)
assert out.shape == (B, S, D), f'Expected ({B},{S},{D}), got {out.shape}'""",
        },
        {
            "name": 'Matches standard attention (2 devices)',
            "code": """import torch
torch.manual_seed(1)
B, S, D = 2, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_ring = {fn}(Q, K, V, num_devices=2)
# Standard attention
scale = D ** -0.5
scores = (Q @ K.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
out_std = attn @ V
assert torch.allclose(out_ring, out_std, atol=1e-5), f'Ring attention should match standard attention'""",
        },
        {
            "name": 'Matches standard attention (4 devices)',
            "code": """import torch
torch.manual_seed(2)
B, S, D = 1, 16, 32
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_ring = {fn}(Q, K, V, num_devices=4)
scale = D ** -0.5
scores = (Q @ K.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
out_std = attn @ V
assert torch.allclose(out_ring, out_std, atol=1e-4), 'Ring attention (4 devices) should match standard'""",
        },
        {
            "name": 'Gradient flows',
            "code": """import torch
B, S, D = 1, 4, 8
Q = torch.randn(B, S, D, requires_grad=True)
K = torch.randn(B, S, D, requires_grad=True)
V = torch.randn(B, S, D, requires_grad=True)
out = {fn}(Q, K, V, num_devices=2)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None""",
        },
    ],
    'solution': '''def ring_attention(Q, K, V, num_devices):
    """环形注意力 (Ring Attention)。"""
    # Step 1: 沿序列轴切分每个虚拟设备持有的 Q、K、V 分块
    B, S, D = Q.shape
    chunk_size = S // num_devices
    scale = D ** (-0.5)
    q_chunks = Q.split(chunk_size, dim=1)
    k_chunks = K.split(chunk_size, dim=1)
    v_chunks = V.split(chunk_size, dim=1)
    output_chunks = []
    # Step 2: 对每个 Q 块按环形次序轮转全部 KV 块
    for query_index, q_chunk in enumerate(q_chunks):
        row_max = torch.full((B, chunk_size, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
        row_sum = torch.zeros(B, chunk_size, 1, device=Q.device, dtype=Q.dtype)
        out_chunk = torch.zeros(B, chunk_size, D, device=Q.device, dtype=Q.dtype)
        for ring_step in range(num_devices):
            k_chunk = k_chunks[(query_index + ring_step) % num_devices]
            v_chunk = v_chunks[(query_index + ring_step) % num_devices]
            scores = q_chunk @ k_chunk.transpose(-2, -1) * scale
            new_row_max = torch.maximum(row_max, scores.max(dim=-1, keepdim=True).values)
            exp_scores = torch.exp(scores - new_row_max)
            # Step 3: 用新的行最大值重标定旧统计量，在线合并 softmax 分母与加权 V
            new_row_sum = row_sum * torch.exp(row_max - new_row_max) + exp_scores.sum(dim=-1, keepdim=True)
            out_chunk = out_chunk * (row_sum * torch.exp(row_max - new_row_max)) / new_row_sum + exp_scores @ v_chunk / new_row_sum
            row_max, row_sum = (new_row_max, new_row_sum)
        output_chunks.append(out_chunk)
    return torch.cat(output_chunks, dim=1)''',
    'demo': """torch.manual_seed(42)
B, S, D = (2, 8, 16)
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
scale = D ** (-0.5)
scores_ref = Q @ K.transpose(-2, -1) * scale
ref_out = torch.softmax(scores_ref, dim=-1) @ V
for num_devices in [2, 4]:
    ring_out = ring_attention(Q, K, V, num_devices=num_devices)
    max_diff = (ring_out - ref_out).abs().max().item()
    match = torch.allclose(ring_out, ref_out, atol=1e-05)
    print(f'num_devices={num_devices}  shape={tuple(ring_out.shape)}  max_diff={max_diff:.2e}  match={match}')""",
    'explanation': r"""### 核心思路
将长度 $S$ 的序列切成 $P$ 块，每个虚拟设备固定持有一个 Q 块，并让所有 KV 块沿环轮转。对某个 Q 块，无需同时保存完整 $S$ 列 logits：维护已见键上的行最大值 $m$、指数和 $\ell$ 与归一化输出 $o$，每收到新块就在线合并，最终仍精确得到 $\operatorname{softmax}(QK^\top/\sqrt D)V$。

### 代码步骤拆解
- **Step 1（序列分片）**：`Q/K/V.split(chunk_size,dim=1)` 得到 $P$ 个 `(B,S/P,D)` 块，缩放系数为 `D**-0.5`。
- **Step 2（环形 KV 轮转）**：第 `query_index` 个 Q 块在 `ring_step` 使用索引 `(query_index+ring_step)%P`，经过 $P$ 步恰好访问每个 KV 块一次。
- **Step 3（在线 softmax 合并）**：令新最大值 `new_row_max=max(row_max,max(scores))`，先以 $\exp(m-m')$ 重缩放旧分母和旧输出分子，再加入当前块的 `exp_scores` 与 `exp_scores @ v_chunk`。

### 重点与易错点分析
- **注意点 1（旧统计量必须重标定）**：最大值从 $m$ 更新为 $m'$ 后，旧指数和不是原值，而要乘 $\exp(m-m')$；漏掉该因子会使不同 KV 块的权重不在同一尺度。
- **注意点 2（保存的是归一化输出）**：代码的 `out_chunk` 已除以旧 `row_sum`，合并时需先乘回重标定后的旧分母再除以 `new_row_sum`；这与直接累计未归一化分子是等价的两种写法。
- **注意点 3（序列并行收益）**：单机模拟总 FLOPs 不变，优势在真实多设备上是每卡仅持有局部 Q/K/V 和 `(S/P)\times(S/P)` 分数块；代价是 $P$ 轮点对点 KV 通信，且要求 `S` 可被 `P` 整除。

### 复杂度分析
- **时间复杂度**：所有 Q 块与所有 K 块两两相乘，总计算量为 $O(BS^2D)$，与标准全注意力相同。
- **空间复杂度**：每次分数块为 $O(BS^2/P^2)$；单个 Q 块的在线状态为 $O(BSD/P)$。本单机实现还保留完整输入输出，而分布式每卡激活目标约为局部序列规模。""",
}
