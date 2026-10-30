"""Mixture of Experts (MoE) task."""

TASK = {
    'title': 'Mixture of Experts (MoE)',
    'title_zh': '混合专家（MoE）',
    'difficulty': 'Hard',
    'category': '混合专家模型',
    'function_name': 'MixtureOfExperts',
    'starter_code': """class MixtureOfExperts(nn.Module):
    pass""",
    'hint': """`router`: `Linear(d, num_experts)` → `topk` → `softmax` weights. Each expert: `Linear → ReLU → Linear`. Weighted sum of top-k expert outputs per token.""",
    'hint_zh': """`router`：`Linear(d, num_experts)` → `topk` → `softmax` 权重。每个专家：`Linear → ReLU → Linear`。对每个 token 的 top-k 专家输出加权求和。""",
    'description_en': """Implement a Mixture of Experts (MoE) layer as an nn.Module.

MoE routes each token to the top-k experts via a learned router, combining their outputs with softmax-normalized weights for conditional computation.

**Signature:** `MixtureOfExperts(d_model, d_ff, num_experts, top_k=2)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, S, d_model)

**Returns:** output tensor (B, S, d_model)

**Constraints:**
- Router: `nn.Linear(d_model, num_experts)` -> topk -> softmax
- Each expert: Linear -> ReLU -> Linear
- Store experts in `self.experts` (nn.ModuleList)""",
    'description_zh': """实现混合专家（MoE）层（nn.Module）。

MoE 通过学习的路由器将每个 token 路由到 top-k 个专家，用 softmax 归一化的权重组合它们的输出，实现条件计算。

**签名:** `MixtureOfExperts(d_model, d_ff, num_experts, top_k=2)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, S, d_model)

**返回:** 输出张量 (B, S, d_model)

**约束:**
- 路由器：`nn.Linear(d_model, num_experts)` -> topk -> softmax
- 每个专家：Linear -> ReLU -> Linear
- 专家存储在 `self.experts`（nn.ModuleList）中""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch, torch.nn as nn
moe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2)
assert isinstance(moe, nn.Module)
out = moe(torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape: {out.shape}'""",
        },
        {
            "name": 'Has router and experts',
            "code": """import torch, torch.nn as nn
moe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2)
assert hasattr(moe, 'router'), 'Need self.router'
assert hasattr(moe, 'experts'), 'Need self.experts'
assert len(moe.experts) == 4, f'Expected 4 experts, got {len(moe.experts)}'""",
        },
        {
            "name": 'Router logits shape',
            "code": """import torch
moe = {fn}(d_model=16, d_ff=32, num_experts=8, top_k=2)
logits = moe.router(torch.randn(4, 16))
assert logits.shape == (4, 8), f'Router output: {logits.shape}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
moe = {fn}(d_model=16, d_ff=32, num_experts=4, top_k=2)
x = torch.randn(1, 4, 16, requires_grad=True)
moe(x).sum().backward()
assert x.grad is not None, 'x.grad is None'""",
        },
        {
            "name": 'Routing to single expert matches manual computation',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
d_model, d_ff, num_experts, top_k = 8, 16, 4, 1
moe = {fn}(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k)
# Force router to always select expert 0 via large bias
with torch.no_grad():
    moe.router.weight.zero_()
    if moe.router.bias is not None:
        moe.router.bias.zero_()
        moe.router.bias[0] = 1e6  # expert 0 always wins
    else:
        # No bias: set weight row 0 to a large constant vector
        moe.router.weight[0] = 1e6
torch.manual_seed(1)
x = torch.randn(2, 3, d_model)
out = moe(x)
# With top_k=1 and expert 0 always winning, softmax of single value = 1.0
# Expected: expert 0 applied to every token with weight 1.0
x_flat = x.reshape(-1, d_model)
expert0 = moe.experts[0]
expected = expert0(x_flat).reshape(2, 3, d_model)
assert torch.allclose(out, expected, atol=1e-5), f'Max diff: {(out - expected).abs().max().item()}'""",
        },
    ],
    'solution': '''class MixtureOfExperts(nn.Module):
    """混合专家（MoE） (Mixture of Experts (MoE))。"""
    # Step 1: 构造 token 路由器与 E 个独立的两层前馈专家

    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_experts)])

    def forward(self, x):
        original_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        # Step 2: 为每个 token 选出 top-k 专家并仅在入选 logits 上归一化门控权重
        router_logits = self.router(x_flat)
        top_logits, top_indices = router_logits.topk(self.top_k, dim=-1)
        router_weights = torch.softmax(top_logits, dim=-1)
        out_flat = torch.zeros_like(x_flat)
        # Step 3: 按专家收集 token，计算前馈输出后乘路由权重并散加回原位置
        for route_index in range(self.top_k):
            for expert_index, expert in enumerate(self.experts):
                token_mask = top_indices[:, route_index] == expert_index
                if token_mask.any():
                    expert_out = expert(x_flat[token_mask])
                    route_weight = router_weights[token_mask, route_index].unsqueeze(-1)
                    out_flat[token_mask] += route_weight * expert_out
        return out_flat.reshape(original_shape)''',
    'demo': """torch.manual_seed(42)
B, S, D = (2, 8, 32)
moe = MixtureOfExperts(D, 64, num_experts=4, top_k=2)
x = torch.randn(B, S, D)
out = moe(x)
with torch.no_grad():
    chosen = moe.router(x).topk(2, dim=-1).indices
print('输入/输出形状:', tuple(x.shape), tuple(out.shape))
print('首个 token 的专家编号:', chosen[0, 0].tolist())
print('输出均方值:', f'{out.square().mean().item():.4f}')""",
    'explanation': r"""### 核心思路
MoE 将稠密前馈计算改为逐 token 的条件计算。路由器把展平后的 $x_t\in\mathbb R^D$ 映射为 $E$ 个分数，只保留集合 $\mathcal K_t=\operatorname{TopK}(r_t)$，再在该集合内计算 $p_{t,e}=\operatorname{softmax}(r_{t,\mathcal K_t})_e$。最终输出为 $y_t=\sum_{e\in\mathcal K_t}p_{t,e}E_e(x_t)$；每个专家都是 $D\to d_{ff}\to D$ 的 ReLU 前馈网络。

### 代码步骤拆解
- **Step 1（建立路由与专家池）**：`self.router` 产生每个 token 的 `num_experts` 个 logits，`self.experts` 保存互不共享参数的两层 MLP。
- **Step 2（稀疏选路）**：把 `(B,S,D)` 展平为 `(BS,D)`，`topk` 得到 `(BS,K)` 的 `top_indices`，并对对应的 `top_logits` 做 softmax，使每个 token 的 K 个权重之和为 1。
- **Step 3（分组执行与散加）**：`token_mask` 将同一路由槽中发往同一专家的 token 批量送入该专家，专家输出乘 `(n_e,1)` 的门控权重后累加到 `out_flat`，最后恢复 `(B,S,D)`。

### 重点与易错点分析
- **注意点 1（归一化范围）**：softmax 必须只作用于入选的 K 个 logits；若先对全部 E 个专家归一化再截断，保留权重之和会小于 1，输出尺度随被丢弃概率改变。
- **注意点 2（稀疏路由的梯度）**：`top_indices` 是离散选择，未入选专家本轮没有梯度；入选分支中的 `router_weights`、专家参数和输入仍保持可微。
- **注意点 3（重复累加与容量）**：一个 token 出现在 K 个路由槽中，因此必须用 `+=` 汇总。此实现没有专家容量上限，语义精确但负载不均时单个专家可能处理大量 token。

### 复杂度分析
- **时间复杂度**：路由投影为 $O(BSDE)$，top-k 选择约为 $O(BSE)$，实际专家计算为 $O(BSKDd_{ff})$；瓶颈通常是路由投影和 K 路稀疏 MLP。
- **空间复杂度**：路由 logits 占 $O(BSE)$，索引与权重占 $O(BSK)$，输出及分批专家激活占 $O(BSD+BSKd_{ff})$。""",
}
