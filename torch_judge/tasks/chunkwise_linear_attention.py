"""Chunkwise parallel causal linear-attention task."""

TASK = {
    "title": "Chunkwise Parallel Causal Linear Attention",
    "title_zh": "分块并行因果线性注意力（Chunkwise Linear Attention）",
    "difficulty": "Hard",
    "category": "注意力机制",
    "function_name": "chunkwise_linear_attention",
    "starter_code": '''import torch

def chunkwise_linear_attention(Q, K, V, chunk_size=4, gamma=0.95):
    pass
''',
    "hint": "Within each chunk, form all q-k dot products and multiply by a lower-triangular decay matrix. Add the decayed incoming-state read, then update the state with decayed outer products.",
    "hint_zh": "在每个块内计算全部 q-k 点积并乘以下三角衰减矩阵，加上衰减后的输入状态读取，再用衰减外积更新状态。",
    "description_en": "Implement chunk-parallel causal linear attention whose outputs and final matrix state match the token-by-token decayed recurrence.",
    "description_zh": "实现分块并行因果线性注意力，使输出和最终矩阵状态与逐 token 衰减递推一致。",
    "theory_en": "The recurrent state summarizes all previous key-value outer products. Inside a chunk, associativity exposes a causal matrix multiplication, while powers of gamma preserve each token's exact recurrence age.",
    "theory_zh": "递归状态汇总此前所有键值外积。在块内，结合律将计算转化为因果矩阵乘法，而 gamma 的幂精确保留每个 token 的递推时距。",
    "diagram_en": "```text\nincoming S --gamma^(i+1) reads--\\\nchunk Q,K -> causal decayed QK -> V -> outputs\nchunk K,V -> terminal decay weights -> next S\n```",
    "diagram_zh": "```text\n输入 S --gamma^(i+1) 读取--\\\n块 Q,K -> 因果衰减 QK -> V -> 输出\n块 K,V -> 末端衰减权重 -> 下一状态 S\n```",
    "tests": [
        {
            "name": "sequence and state shapes",
            "code": '''import torch
Q,K,V = [torch.randn(2,8,3) for _ in range(3)]
out, state = {fn}(Q,K,V,chunk_size=4)
assert out.shape == (2,8,3) and state.shape == (2,3,3)
''',
        },
        {
            "name": "exact step recurrence",
            "code": '''import torch
torch.manual_seed(11)
Q,K,V = [torch.randn(2,6,3,dtype=torch.double) for _ in range(3)]
out, state = {fn}(Q,K,V,chunk_size=2,gamma=0.8)
s = torch.zeros(2,3,3,dtype=torch.double); ys = []
for t in range(6):
    s = 0.8*s + K[:,t,:,None]*V[:,t,None,:]
    ys.append(torch.einsum('bd,bde->be',Q[:,t],s))
ref = torch.stack(ys,1)
assert torch.allclose(out,ref,atol=1e-12,rtol=1e-12)
assert torch.allclose(state,s,atol=1e-12,rtol=1e-12)
''',
        },
        {
            "name": "unit decay and single-token chunks",
            "code": '''import torch
Q = torch.ones(1,4,2); K = torch.ones(1,4,2); V = torch.arange(8.).reshape(1,4,2)
out, state = {fn}(Q,K,V,chunk_size=1,gamma=1.0)
ref = 2*V.cumsum(dim=1)
assert torch.equal(out,ref)
assert torch.equal(state,torch.einsum('bsd,bse->bde',K,V))
''',
        },
        {
            "name": "gradient flow through chunks",
            "code": '''import torch
Q,K,V = [torch.randn(1,8,4,requires_grad=True) for _ in range(3)]
out, state = {fn}(Q,K,V,chunk_size=4,gamma=0.9)
(out.square().sum()+state.square().sum()).backward()
assert out.requires_grad and state.requires_grad
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (Q,K,V))
''',
        },
    ],
    "solution": '''import torch

def chunkwise_linear_attention(Q, K, V, chunk_size=4, gamma=0.95):
    B, S, D = Q.shape
    C = chunk_size
    num_chunks = S // C
    state = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
    outputs = []
    decay = torch.full((), gamma, device=Q.device, dtype=Q.dtype)
    local = torch.arange(C, device=Q.device)
    causal = local.unsqueeze(1) >= local.unsqueeze(0)
    intra_decay = decay ** (local.unsqueeze(1) - local.unsqueeze(0))
    intra_decay = intra_decay * causal.to(Q.dtype)
    read_decay = decay ** (local + 1)
    write_decay = decay ** (C - 1 - local)
    for c in range(num_chunks):
        start = c * C
        q_c = Q[:, start:start + C]
        k_c = K[:, start:start + C]
        v_c = V[:, start:start + C]
        inter = torch.einsum('bcd,bde->bce', q_c, state) * read_decay.view(1, C, 1)
        qk = torch.einsum('bid,bjd->bij', q_c, k_c)
        intra = torch.einsum('bij,bje->bie', qk * intra_decay, v_c)
        outputs.append(inter + intra)
        chunk_update = torch.einsum('c,bcd,bce->bde', write_decay, k_c, v_c)
        state = (decay ** C) * state + chunk_update
    return torch.cat(outputs, dim=1), state
''',
    "demo": '''Q,K,V = [torch.randn(2,128,32) for _ in range(3)]
out, state = chunkwise_linear_attention(Q,K,V,chunk_size=16,gamma=0.99)
print(out.shape, state.shape)
''',
    "explanation": "Inter-chunk reads age the incoming state by i+1 steps. The lower-triangular intra-chunk kernel assigns age i-j to each local write, and the terminal update ages writes to the chunk boundary.",
}
