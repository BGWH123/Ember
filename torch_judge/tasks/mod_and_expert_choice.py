"""Mixture-of-Depths and expert-choice routing task."""

TASK = {
    "title": "Mixture-of-Depths (MoD) & Expert-Choice Routing",
    "title_zh": "混合深度路由（MoD）与专家选择路由（Expert-Choice）",
    "difficulty": "Hard",
    "category": "混合专家模型",
    "function_name": "mod_and_expert_choice",
    "starter_code": '''import torch

def mod_and_expert_choice(x, mod_logits, ffn_out, router_logits, capacity_k):
    pass
''',
    "hint": "Scatter each batch row's MoD top-k indices into a boolean mask. Separately softmax router logits over experts, transpose, and let every expert run its own top-k.",
    "hint_zh": "将每个批次行的 MoD top-k 索引散射成布尔掩码；另对路由 logit 按专家做 softmax、转置，并让每位专家独立执行 top-k。",
    "description_en": "Combine token-level depth skipping with expert-choice routing, returning gated residual outputs and each expert's selected tokens and scores.",
    "description_zh": "结合 token 级深度跳过与专家选择路由，返回门控残差输出以及每位专家选择的 token 和分数。",
    "theory_en": "MoD limits expensive block computation by selecting tokens, while preserving an identity residual for bypassed positions. Expert-choice reverses token-choice routing: experts claim their highest-probability tokens under fixed capacity.",
    "theory_zh": "MoD 通过选择 token 限制昂贵模块的计算，并让跳过位置保持恒等残差。专家选择路由反转 token 选择：每位专家在固定容量下挑选概率最高的 token。",
    "diagram_en": "```text\nmod logits -> sigmoid -> per-batch top-k -> gated FFN residual\nrouter logits -> softmax over experts -> transpose -> per-expert top-k tokens\n```",
    "diagram_zh": "```text\n深度 logit -> sigmoid -> 逐批 top-k -> 门控 FFN 残差\n专家 logit -> 按专家 softmax -> 转置 -> 每位专家 top-k token\n```",
    "tests": [
        {
            "name": "routing output shapes",
            "code": '''import torch
B,S,D,E,K = 2,6,4,3,2
out, tok, val = {fn}(torch.randn(B,S,D),torch.randn(B,S),torch.randn(B,S,D),torch.randn(S,E),K)
assert out.shape == (B,S,D)
assert tok.shape == val.shape == (E,K) and tok.dtype == torch.long
''',
        },
        {
            "name": "exact MoD and expert reference",
            "code": '''import torch
x = torch.arange(24.,dtype=torch.double).reshape(2,4,3)
f = torch.ones_like(x)*2
m = torch.tensor([[0.,3.,-2.,1.],[4.,-1.,2.,0.]],dtype=torch.double)
r = torch.tensor([[3.,0.],[0.,2.],[1.,1.],[2.,-1.]],dtype=torch.double)
out, tok, val = {fn}(x,m,f,r,2)
w = torch.sigmoid(m); idx = torch.topk(w,2,-1).indices
mask = torch.zeros_like(w,dtype=torch.bool).scatter(1,idx,True)
ref = x + mask.unsqueeze(-1)*w.unsqueeze(-1)*f
p = torch.softmax(r,-1); rv,rt = torch.topk(p.T,2,-1)
assert torch.equal(out,ref) and torch.equal(tok,rt) and torch.equal(val,rv)
''',
        },
        {
            "name": "full-capacity edge case",
            "code": '''import torch
B,S,D,E = 1,3,2,2
x = torch.randn(B,S,D); f = torch.randn(B,S,D); m = torch.randn(B,S); r = torch.randn(S,E)
out, tok, val = {fn}(x,m,f,r,S)
assert torch.allclose(out,x+torch.sigmoid(m).unsqueeze(-1)*f)
assert torch.equal(torch.sort(tok,-1).values,torch.arange(S).expand(E,S))
''',
        },
        {
            "name": "gradient through selected routes",
            "code": '''import torch
B,S,D,E = 2,5,3,4
x = torch.randn(B,S,D,requires_grad=True); f = torch.randn(B,S,D,requires_grad=True)
m = torch.randn(B,S,requires_grad=True); r = torch.randn(S,E,requires_grad=True)
out, tok, val = {fn}(x,m,f,r,3)
(out.square().sum()+val.square().sum()).backward()
assert tok.shape == (E,3) and out.requires_grad
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x,f,m,r))
''',
        },
    ],
    "solution": '''import torch

def mod_and_expert_choice(x, mod_logits, ffn_out, router_logits, capacity_k):
    mod_weights = torch.sigmoid(mod_logits)
    selected = torch.topk(mod_weights, k=capacity_k, dim=-1).indices
    mask = torch.zeros_like(mod_weights, dtype=torch.bool)
    mask = mask.scatter(dim=-1, index=selected, value=True)
    mod_out = x + mask.unsqueeze(-1) * mod_weights.unsqueeze(-1) * ffn_out
    probabilities = torch.softmax(router_logits, dim=-1)
    top_vals, top_tokens = torch.topk(probabilities.T, k=capacity_k, dim=-1)
    return mod_out, top_tokens, top_vals
''',
    "demo": '''B,S,D,E = 2,16,32,4
out, tokens, scores = mod_and_expert_choice(torch.randn(B,S,D),torch.randn(B,S),torch.randn(B,S,D),torch.randn(S,E),4)
print(out.shape, tokens.shape, scores.shape)
''',
    "explanation": "The MoD mask changes only selected positions, while every token retains its residual x. Expert-choice probabilities are normalized token-wise before each expert independently fills its capacity.",
}
