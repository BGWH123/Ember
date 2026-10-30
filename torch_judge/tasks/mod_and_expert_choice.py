"""Mixture-of-Depths (MoD) & Expert-Choice Routing task."""

TASK = {
    'title': 'Mixture-of-Depths (MoD) & Expert-Choice Routing',
    'title_zh': '混合深度路由（MoD）与专家选择路由（Expert-Choice）',
    'difficulty': 'Hard',
    'category': '混合专家模型',
    'function_name': 'mod_and_expert_choice',
    'starter_code': """def mod_and_expert_choice(x, mod_logits, ffn_out, router_logits, capacity_k):
    pass""",
    'hint': """Scatter each batch row's MoD top-k indices into a boolean mask. Separately softmax router logits over experts, transpose, and let every expert run its own top-k.""",
    'hint_zh': '将每个批次行的 MoD top-k 索引散射成布尔掩码；另对路由 logit 按专家做 softmax、转置，并让每位专家独立执行 top-k。',
    'description_en': """Combine token-level depth skipping with expert-choice routing, returning gated residual outputs and each expert's selected tokens and scores.""",
    'description_zh': '结合 token 级深度跳过与专家选择路由，返回门控残差输出以及每位专家选择的 token 和分数。',
    'theory_en': """MoD limits expensive block computation by selecting tokens, while preserving an identity residual for bypassed positions. Expert-choice reverses token-choice routing: experts claim their highest-probability tokens under fixed capacity.""",
    'theory_zh': 'MoD 通过选择 token 限制昂贵模块的计算，并让跳过位置保持恒等残差。专家选择路由反转 token 选择：每位专家在固定容量下挑选概率最高的 token。',
    'diagram_en': """```text
mod logits -> sigmoid -> per-batch top-k -> gated FFN residual
router logits -> softmax over experts -> transpose -> per-expert top-k tokens
```""",
    'diagram_zh': """```text
深度 logit -> sigmoid -> 逐批 top-k -> 门控 FFN 残差
专家 logit -> 按专家 softmax -> 转置 -> 每位专家 top-k token
```""",
    "tests": [
        {
            "name": 'routing output shapes',
            "code": """import torch
B,S,D,E,K = 2,6,4,3,2
out, tok, val = {fn}(torch.randn(B,S,D),torch.randn(B,S),torch.randn(B,S,D),torch.randn(S,E),K)
assert out.shape == (B,S,D)
assert tok.shape == val.shape == (E,K) and tok.dtype == torch.long""",
        },
        {
            "name": 'exact MoD and expert reference',
            "code": """import torch
x = torch.arange(24.,dtype=torch.double).reshape(2,4,3)
f = torch.ones_like(x)*2
m = torch.tensor([[0.,3.,-2.,1.],[4.,-1.,2.,0.]],dtype=torch.double)
r = torch.tensor([[3.,0.],[0.,2.],[1.,1.],[2.,-1.]],dtype=torch.double)
out, tok, val = {fn}(x,m,f,r,2)
w = torch.sigmoid(m); idx = torch.topk(w,2,-1).indices
mask = torch.zeros_like(w,dtype=torch.bool).scatter(1,idx,True)
ref = x + mask.unsqueeze(-1)*w.unsqueeze(-1)*f
p = torch.softmax(r,-1); rv,rt = torch.topk(p.T,2,-1)
assert torch.equal(out,ref) and torch.equal(tok,rt) and torch.equal(val,rv)""",
        },
        {
            "name": 'full-capacity edge case',
            "code": """import torch
B,S,D,E = 1,3,2,2
x = torch.randn(B,S,D); f = torch.randn(B,S,D); m = torch.randn(B,S); r = torch.randn(S,E)
out, tok, val = {fn}(x,m,f,r,S)
assert torch.allclose(out,x+torch.sigmoid(m).unsqueeze(-1)*f)
assert torch.equal(torch.sort(tok,-1).values,torch.arange(S).expand(E,S))""",
        },
        {
            "name": 'gradient through selected routes',
            "code": """import torch
B,S,D,E = 2,5,3,4
x = torch.randn(B,S,D,requires_grad=True); f = torch.randn(B,S,D,requires_grad=True)
m = torch.randn(B,S,requires_grad=True); r = torch.randn(S,E,requires_grad=True)
out, tok, val = {fn}(x,m,f,r,3)
(out.square().sum()+val.square().sum()).backward()
assert tok.shape == (E,3) and out.requires_grad
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x,f,m,r))""",
        },
    ],
    'solution': '''def mod_and_expert_choice(x, mod_logits, ffn_out, router_logits, capacity_k):
    """联合实现深度与专家选择路由 (MoD and Expert-Choice Routing)。"""
    # Step 1: 为每个批次选择 MoD 权重最高的 token
    mod_weights = torch.sigmoid(mod_logits)
    selected_tokens = torch.topk(mod_weights, k=capacity_k, dim=-1).indices
    mod_mask = torch.zeros_like(mod_weights, dtype=torch.bool)
    mod_mask.scatter_(dim=-1, index=selected_tokens, value=True)

    # Step 2: 仅在选中位置加入门控 FFN 残差
    gated_ffn = mod_mask.unsqueeze(-1) * mod_weights.unsqueeze(-1) * ffn_out
    mod_out = x + gated_ffn

    # Step 3: 每个专家独立选择概率最高的 capacity_k 个 token
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_scores, expert_tokens = torch.topk(
        router_probs.T, k=capacity_k, dim=-1
    )
    return mod_out, expert_tokens, expert_scores''',
    'demo': """torch.manual_seed(42)
B, S, D, E = 2, 8, 16, 4
x = torch.randn(B, S, D)
mod_logits = torch.randn(B, S)
ffn_out = torch.randn(B, S, D)
router_logits = torch.randn(S, E)
out, tokens, scores = mod_and_expert_choice(x, mod_logits, ffn_out, router_logits, capacity_k=3)
print("Output shape:", out.shape)
print("Expert token shape:", tokens.shape)
print("Expert scores:", scores[:, 0])""",
    'explanation': r"""### 核心思路
MoD 在每个批次内选择 `capacity_k` 个 token 执行门控 FFN，同时保留所有 token 的恒等残差；Expert-Choice 则让每个专家从全序列独立挑选最高路由概率 token。

### 代码步骤拆解
- **Step 1（深度选择）**：sigmoid 后逐批 top-k，并 scatter 成 `(B, S)` mask。
- **Step 2（门控残差）**：只在选中位置加入 `mod_weights * ffn_out`。
- **Step 3（专家选择）**：对专家维 softmax 后转置为 `(E, S)`，逐专家 top-k。

### 重点与易错点分析
- **注意点 1（两种归一化）**：MoD 使用独立 sigmoid，专家路由则按每个 token 的专家维 softmax。
- **注意点 2（容量边界）**：`capacity_k` 不能超过序列长度，否则 `topk` 会报错。
- **注意点 3（离散路由梯度）**：top-k 索引本身不可导，但被选中分数和门控权重仍可向 logits 传播梯度。

### 复杂度分析
- **时间复杂度**：$O(BS\log K+SE+ES\log K)$，top-k 与路由 softmax 占主导。
- **空间复杂度**：$O(BS+SE+EK)$，保存掩码、概率及专家选择结果。""",
}
