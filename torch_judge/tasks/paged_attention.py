"""Paged Attention task."""

TASK = {
    'title': 'Paged Attention',
    'title_zh': '分页注意力',
    'difficulty': 'Hard',
    'category': '推理优化',
    'function_name': 'paged_attention',
    'starter_code': """def paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size):
    pass""",
    'hint': """for b in range(B):
  K = k_pages[block_table[b]].reshape(-1, D)[:context_len]  # gather + slice
  V = v_pages[block_table[b]].reshape(-1, D)[:context_len]
  scores = Q[b] @ K.T / √D → softmax → @ V""",
    'hint_zh': """for b in range(B):
  K = k_pages[block_table[b]].reshape(-1, D)[:context_len]  # gather + 截取
  V = v_pages[block_table[b]].reshape(-1, D)[:context_len]
  scores = Q[b] @ K.T / √D → softmax → @ V""",
    'description_en': """Implement paged attention (vLLM-style).

Instead of a contiguous KV cache, keys and values are stored in fixed-size pages (blocks). A block table maps logical block indices to physical page indices, enabling non-contiguous memory allocation.

**Signature:** `paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size) -> Tensor`

**Parameters:**
- `Q` — query tensor `(B, 1, D)` (single decode step)
- `k_pages` — key pages `(num_pages, block_size, D)`
- `v_pages` — value pages `(num_pages, block_size, D)`
- `block_table` — integer tensor `(B, max_blocks)` mapping logical → physical page
- `context_len` — number of valid KV tokens per sequence (scalar)
- `block_size` — tokens per page

**Returns:** output tensor `(B, 1, D)`

**Steps:** Use block_table to gather K/V pages, slice to context_len, then compute standard scaled dot-product attention.""",
    'description_zh': """实现分页注意力（vLLM 风格）。

KV 缓存不再连续存储，而是存在固定大小的页（block）中。block table 将逻辑块索引映射到物理页索引，实现非连续内存分配。

**签名:** `paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size) -> Tensor`

**参数:**
- `Q` — 查询张量 `(B, 1, D)`（单步解码）
- `k_pages` — 键页 `(num_pages, block_size, D)`
- `v_pages` — 值页 `(num_pages, block_size, D)`
- `block_table` — 整数张量 `(B, max_blocks)`，逻辑块 → 物理页
- `context_len` — 每条序列有效 KV token 数（标量）
- `block_size` — 每页 token 数

**返回:** 输出张量 `(B, 1, D)`

**步骤:** 用 block_table gather K/V 页，截取到 context_len，再做标准缩放点积注意力。""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, D, block_size, num_pages = 2, 16, 4, 8
context_len = 6
max_blocks = (context_len + block_size - 1) // block_size
Q = torch.randn(B, 1, D)
k_pages = torch.randn(num_pages, block_size, D)
v_pages = torch.randn(num_pages, block_size, D)
block_table = torch.zeros(B, max_blocks, dtype=torch.long)
for b in range(B):
    for i in range(max_blocks):
        block_table[b, i] = b * max_blocks + i
out = {fn}(Q, k_pages, v_pages, block_table, context_len, block_size)
assert out.shape == (B, 1, D), f'Expected ({B}, 1, {D}), got {out.shape}'""",
        },
        {
            "name": 'Matches standard attention with contiguous layout',
            "code": """import torch
torch.manual_seed(1)
B, S, D, block_size = 1, 8, 16, 4
num_blocks = S // block_size
# Build contiguous K/V as pages
K_full = torch.randn(B, S, D)
V_full = torch.randn(B, S, D)
Q = torch.randn(B, 1, D)
# Pages: (num_blocks, block_size, D)
k_pages = K_full[0].view(num_blocks, block_size, D)
v_pages = V_full[0].view(num_blocks, block_size, D)
# Identity block table: logical block i -> physical page i
block_table = torch.arange(num_blocks).unsqueeze(0)  # (1, num_blocks)
out_paged = {fn}(Q, k_pages, v_pages, block_table, S, block_size)
# Standard attention
scale = D ** -0.5
scores = (Q @ K_full.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
out_std = attn @ V_full
assert torch.allclose(out_paged, out_std, atol=1e-5), f'Paged attention should match standard attention'""",
        },
        {
            "name": 'Works with non-contiguous block table',
            "code": """import torch
torch.manual_seed(2)
B, S, D, block_size = 1, 8, 16, 4
num_blocks = S // block_size  # 2 blocks
K_full = torch.randn(B, S, D)
V_full = torch.randn(B, S, D)
Q = torch.randn(B, 1, D)
# Allocate 4 physical pages, use pages 3 and 1 (non-contiguous)
num_phys_pages = 4
k_pages = torch.zeros(num_phys_pages, block_size, D)
v_pages = torch.zeros(num_phys_pages, block_size, D)
phys_indices = [3, 1]  # logical block 0 -> page 3, logical block 1 -> page 1
for logical, phys in enumerate(phys_indices):
    k_pages[phys] = K_full[0, logical*block_size:(logical+1)*block_size]
    v_pages[phys] = V_full[0, logical*block_size:(logical+1)*block_size]
block_table = torch.tensor(phys_indices).unsqueeze(0)  # (1, 2)
out_paged = {fn}(Q, k_pages, v_pages, block_table, S, block_size)
# Standard attention
scale = D ** -0.5
scores = (Q @ K_full.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
out_std = attn @ V_full
assert torch.allclose(out_paged, out_std, atol=1e-5), 'Non-contiguous block table should give same result'""",
        },
        {
            "name": 'Gradient flows',
            "code": """import torch
B, D, block_size = 1, 8, 4
Q = torch.randn(B, 1, D, requires_grad=True)
k_pages = torch.randn(4, block_size, D, requires_grad=True)
v_pages = torch.randn(4, block_size, D, requires_grad=True)
block_table = torch.arange(4).unsqueeze(0)
out = {fn}(Q, k_pages, v_pages, block_table, 16, block_size)
out.sum().backward()
assert Q.grad is not None and k_pages.grad is not None""",
        },
    ],
    'solution': '''def paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size):
    """分页注意力 (Paged Attention)。"""
    # Step 1: 按 block_table 将每条序列的逻辑 KV 块映射到物理页
    B, _, D = Q.shape
    scale = D ** (-0.5)
    output_chunks = []
    for batch_index in range(B):
        physical_blocks = block_table[batch_index]
        gathered_keys = k_pages[physical_blocks].reshape(-1, D)
        gathered_values = v_pages[physical_blocks].reshape(-1, D)
        # Step 2: 丢弃末页填充，只保留 context_len 个有效历史 token
        context_keys = gathered_keys[:context_len].unsqueeze(0)
        context_values = gathered_values[:context_len].unsqueeze(0)
        # Step 3: 对重建的 KV 上下文计算缩放点积注意力并拼回批次
        scores = Q[batch_index:batch_index + 1] @ context_keys.transpose(-2, -1) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output_chunks.append(attn_weights @ context_values)
    return torch.cat(output_chunks, dim=0)''',
    'demo': """torch.manual_seed(42)
B, S, D = (2, 8, 16)
block_size = 4
num_blocks = S // block_size
K_full = torch.randn(B, S, D)
V_full = torch.randn(B, S, D)
Q = torch.randn(B, 1, D)
scale = D ** (-0.5)
scores_ref = Q @ K_full.transpose(-2, -1) * scale
ref_out = torch.softmax(scores_ref, dim=-1) @ V_full
total_pages = B * num_blocks
k_pages = torch.zeros(total_pages, block_size, D)
v_pages = torch.zeros(total_pages, block_size, D)
block_table = []
for b in range(B):
    page_ids = []
    for blk in range(num_blocks):
        pid = b * num_blocks + blk
        k_pages[pid] = K_full[b, blk * block_size:(blk + 1) * block_size]
        v_pages[pid] = V_full[b, blk * block_size:(blk + 1) * block_size]
        page_ids.append(pid)
    block_table.append(page_ids)
paged_out = paged_attention(Q, k_pages, v_pages, block_table, context_len=S, block_size=block_size)
print('Shape:', paged_out.shape)
print('Max diff vs reference:', (paged_out - ref_out).abs().max().item())
print('Match:', torch.allclose(paged_out, ref_out, atol=1e-05))""",
    'explanation': r"""### 核心思路
分页注意力改变的是 KV cache 的存储方式而非注意力公式。`block_table[b, j]` 把序列 `b` 的第 $j$ 个逻辑块映射到任意物理页；按表取页并展平后，前 `context_len` 行恰好恢复按 token 顺序排列的 $K_b,V_b$。随后计算 $Y_b=\operatorname{softmax}(Q_bK_b^\top/\sqrt D)V_b$，因此结果与连续 KV cache 完全一致。

### 代码步骤拆解
- **Step 1（逻辑块寻址）**：对每个 `batch_index` 读取 `physical_blocks=block_table[batch_index]`，用高级索引从形状 `(num_pages, block_size, D)` 的页池中按逻辑顺序取页并展平成 `(-1, D)`。
- **Step 2（裁掉末页空槽）**：`gathered_keys[:context_len]` 与 `gathered_values[:context_len]` 只保留真实历史 token；这一步尤其处理 `context_len` 不能整除 `block_size` 的情况。
- **Step 3（单步解码注意力）**：`Q[b:b+1]` 的形状为 `(1,1,D)`，与 `(1,L,D)` 的上下文计算 `(1,1,L)` 权重，再乘 `context_values` 得到 `(1,1,D)`，最后沿批次拼接。

### 重点与易错点分析
- **注意点 1（页表顺序）**：不能先排序 `physical_blocks`；物理页编号不连续甚至逆序都合法，token 的语义顺序完全由 `block_table` 的列顺序决定。
- **注意点 2（末页有效长度）**：若直接使用所有已分配页，末页未写入的槽位也会进入 softmax，改变归一化分母；必须在展平后截断到 `context_len`。
- **注意点 3（训练与推理代价）**：页索引是离散寻址、对 `block_table` 不可微，但被 gather 的 `k_pages/v_pages` 仍能接收梯度；真实 vLLM 内核会融合 gather 与 attention，避免显式复制连续 KV。

### 复杂度分析
- **时间复杂度**：设上下文长度为 $L$，页收集与单查询注意力均为 $O(BLD)$；点积和加权求和是主要算术开销。
- **空间复杂度**：当前直观实现为每个批次元素暂存重建的 K/V，共峰值 $O(LD)$，输出列表为 $O(BD)$；融合内核可省去连续 K/V 临时副本。""",
}
