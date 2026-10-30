"""Chunkwise Parallel Causal Linear Attention task."""

TASK = {
    'title': 'Chunkwise Parallel Causal Linear Attention',
    'title_zh': '分块并行因果线性注意力（Chunkwise Linear Attention）',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'chunkwise_linear_attention',
    'starter_code': """def chunkwise_linear_attention(Q, K, V, chunk_size=4, gamma=0.95):
    pass""",
    'hint': """Within each chunk, form all q-k dot products and multiply by a lower-triangular decay matrix. Add the decayed incoming-state read, then update the state with decayed outer products.""",
    'hint_zh': '在每个块内计算全部 q-k 点积并乘以下三角衰减矩阵，加上衰减后的输入状态读取，再用衰减外积更新状态。',
    'description_en': """Implement chunk-parallel causal linear attention whose outputs and final matrix state match the token-by-token decayed recurrence.""",
    'description_zh': '实现分块并行因果线性注意力，使输出和最终矩阵状态与逐 token 衰减递推一致。',
    'theory_en': """The recurrent state summarizes all previous key-value outer products. Inside a chunk, associativity exposes a causal matrix multiplication, while powers of gamma preserve each token's exact recurrence age.""",
    'theory_zh': '递归状态汇总此前所有键值外积。在块内，结合律将计算转化为因果矩阵乘法，而 gamma 的幂精确保留每个 token 的递推时距。',
    'diagram_en': r"""```text
incoming S --gamma^(i+1) reads--\
chunk Q,K -> causal decayed QK -> V -> outputs
chunk K,V -> terminal decay weights -> next S
```""",
    'diagram_zh': r"""```text
输入 S --gamma^(i+1) 读取--\
块 Q,K -> 因果衰减 QK -> V -> 输出
块 K,V -> 末端衰减权重 -> 下一状态 S
```""",
    "tests": [
        {
            "name": 'sequence and state shapes',
            "code": """import torch
Q,K,V = [torch.randn(2,8,3) for _ in range(3)]
out, state = {fn}(Q,K,V,chunk_size=4)
assert out.shape == (2,8,3) and state.shape == (2,3,3)""",
        },
        {
            "name": 'exact step recurrence',
            "code": """import torch
torch.manual_seed(11)
Q,K,V = [torch.randn(2,6,3,dtype=torch.double) for _ in range(3)]
out, state = {fn}(Q,K,V,chunk_size=2,gamma=0.8)
s = torch.zeros(2,3,3,dtype=torch.double); ys = []
for t in range(6):
    s = 0.8*s + K[:,t,:,None]*V[:,t,None,:]
    ys.append(torch.einsum('bd,bde->be',Q[:,t],s))
ref = torch.stack(ys,1)
assert torch.allclose(out,ref,atol=1e-12,rtol=1e-12)
assert torch.allclose(state,s,atol=1e-12,rtol=1e-12)""",
        },
        {
            "name": 'unit decay and single-token chunks',
            "code": """import torch
Q = torch.ones(1,4,2); K = torch.ones(1,4,2); V = torch.arange(8.).reshape(1,4,2)
out, state = {fn}(Q,K,V,chunk_size=1,gamma=1.0)
ref = 2*V.cumsum(dim=1)
assert torch.equal(out,ref)
assert torch.equal(state,torch.einsum('bsd,bse->bde',K,V))""",
        },
        {
            "name": 'gradient flow through chunks',
            "code": """import torch
Q,K,V = [torch.randn(1,8,4,requires_grad=True) for _ in range(3)]
out, state = {fn}(Q,K,V,chunk_size=4,gamma=0.9)
(out.square().sum()+state.square().sum()).backward()
assert out.requires_grad and state.requires_grad
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (Q,K,V))""",
        },
    ],
    'solution': '''def chunkwise_linear_attention(Q, K, V, chunk_size=4, gamma=0.95):
    """分块并行因果线性注意力 (Chunkwise Linear Attention)。"""
    B, S, D = Q.shape
    num_chunks = S // chunk_size
    state = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
    outputs = []
    decay = torch.full((), gamma, device=Q.device, dtype=Q.dtype)

    # Step 1: 预计算块内因果衰减与状态读写权重
    positions = torch.arange(chunk_size, device=Q.device)
    causal = positions.unsqueeze(1) >= positions.unsqueeze(0)
    intra_decay = decay ** (
        positions.unsqueeze(1) - positions.unsqueeze(0)
    )
    intra_decay = intra_decay * causal.to(Q.dtype)
    read_decay = decay ** (positions + 1)
    write_decay = decay ** (chunk_size - 1 - positions)

    # Step 2: 并行计算每个块的历史状态读取与块内注意力
    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size
        q_chunk = Q[:, start:start + chunk_size]
        k_chunk = K[:, start:start + chunk_size]
        v_chunk = V[:, start:start + chunk_size]
        inter_chunk = torch.einsum(
            "bcd,bde->bce", q_chunk, state
        ) * read_decay.view(1, chunk_size, 1)
        qk_scores = torch.einsum("bid,bjd->bij", q_chunk, k_chunk)
        intra_chunk = torch.einsum(
            "bij,bje->bie", qk_scores * intra_decay, v_chunk
        )
        outputs.append(inter_chunk + intra_chunk)

        # Step 3: 将当前块写入递归状态
        chunk_update = torch.einsum(
            "c,bcd,bce->bde", write_decay, k_chunk, v_chunk
        )
        state = (decay ** chunk_size) * state + chunk_update
    return torch.cat(outputs, dim=1), state''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 16, 8
Q, K, V = [torch.randn(B, S, D) for _ in range(3)]
out, state = chunkwise_linear_attention(Q, K, V, chunk_size=4)
print("Output shape:", out.shape)
print("State shape:", state.shape)
print("Sample values:", out[0, 0, :3])""",
    'explanation': r"""### 核心思路
递归状态累计衰减的键值外积；块内利用结合律并行计算因果贡献，同时用 $\gamma$ 的幂保持与逐 token 递推一致。

### 代码步骤拆解
- **Step 1（衰减模板）**：构造块内下三角 `intra_decay` 及状态读写的时间权重。
- **Step 2（块内并行）**：`inter_chunk` 读取历史状态，`intra_chunk` 聚合当前块内较早 token。
- **Step 3（状态更新）**：旧 `state` 衰减 `chunk_size` 次，再加入当前块的加权外积。

### 重点与易错点分析
- **注意点 1（时距指数）**：读取第 $i$ 个位置前旧状态应衰减 $i+1$ 次，块内写入时距为 $i-j$。
- **注意点 2（因果方向）**：下三角条件必须对应查询位置不早于键位置。
- **注意点 3（分块边界）**：本实现要求 `S` 能被 `chunk_size` 整除，否则尾部 token 不会被处理。

### 复杂度分析
- **时间复杂度**：$O(BSD^2+BSCD)$，其中 $C$ 为块大小。
- **空间复杂度**：$O(BD^2+BSC)$，保存递归状态和块内注意力。""",
}
