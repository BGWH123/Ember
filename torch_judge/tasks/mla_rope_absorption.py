"""MLA decoupled RoPE and weight-absorption task."""

TASK = {
    "title": "MLA Decoupled RoPE & Weight Matrix Absorption",
    "title_zh": "MLA 解耦 RoPE 与矩阵吸收推理",
    "difficulty": "Hard",
    "category": "注意力机制",
    "function_name": "mla_absorbed_attention",
    "starter_code": '''import torch
import math

def mla_absorbed_attention(q_nope, q_rope, c_kv, k_rope, W_uk, W_uv):
    pass
''',
    "hint": "Move the per-head key up-projection onto the content query, score the compressed cache directly, aggregate in latent space, and apply the value up-projection only after attention.",
    "hint_zh": "把逐头键上投影吸收到内容查询中，直接对压缩缓存打分，在潜空间聚合后再应用值上投影。",
    "description_en": "Implement MLA inference with decoupled positional keys and matrix absorption, avoiding materialization of full per-head K and V caches.",
    "description_zh": "实现带解耦位置键和矩阵吸收的 MLA 推理，避免物化完整的逐头 K、V 缓存。",
    "theory_en": "Associativity lets q(W_uk)^T be evaluated before interaction with the compressed latent. Likewise, attention can aggregate compressed values first and apply W_uv afterward. The separate RoPE branch preserves positional information.",
    "theory_zh": "利用结合律，可在压缩潜变量交互前先计算 q(W_uk)^T；同理可先聚合压缩值，再应用 W_uv。独立 RoPE 分支保留位置信息。",
    "diagram_en": "```text\nq_nope -> absorb W_uk -> score c_kv --\\\nq_rope -----------------> score k_rope -> softmax -> aggregate c_kv -> W_uv -> out\n```",
    "diagram_zh": "```text\nq_nope -> 吸收 W_uk -> 对 c_kv 打分 --\\\nq_rope -----------------> 对 k_rope 打分 -> softmax -> 聚合 c_kv -> W_uv -> 输出\n```",
    "tests": [
        {
            "name": "absorbed tensor and output shapes",
            "code": '''import torch
B,H,Q,K,C,D,R,V = 2,3,4,5,6,7,2,8
out, qa = {fn}(torch.randn(B,H,Q,D),torch.randn(B,H,Q,R),torch.randn(B,K,C),torch.randn(B,K,R),torch.randn(H,C,D),torch.randn(H,C,V))
assert qa.shape == (B,H,Q,C) and out.shape == (B,H,Q,V)
''',
        },
        {
            "name": "exact absorbed reference",
            "code": '''import torch
import math
torch.manual_seed(4)
B,H,Q,K,C,D,R,V = 1,2,3,4,3,2,2,3
qn = torch.randn(B,H,Q,D,dtype=torch.double); qr = torch.randn(B,H,Q,R,dtype=torch.double)
c = torch.randn(B,K,C,dtype=torch.double); kr = torch.randn(B,K,R,dtype=torch.double)
wk = torch.randn(H,C,D,dtype=torch.double); wv = torch.randn(H,C,V,dtype=torch.double)
out, qa = {fn}(qn,qr,c,kr,wk,wv)
rqa = torch.einsum('bhqd,hcd->bhqc',qn,wk)
scores = (torch.einsum('bhqc,bkc->bhqk',rqa,c)+torch.einsum('bhqr,bkr->bhqk',qr,kr))/math.sqrt(D+R)
a = torch.softmax(scores,-1); ro = torch.einsum('bhqc,hcv->bhqv',torch.einsum('bhqk,bkc->bhqc',a,c),wv)
assert torch.allclose(qa,rqa,atol=1e-12,rtol=1e-12)
assert torch.allclose(out,ro,atol=1e-12,rtol=1e-12)
''',
        },
        {
            "name": "single-key edge case",
            "code": '''import torch
B,H,Q,K,C,D,R,V = 2,2,3,1,4,3,2,5
c = torch.randn(B,K,C); wv = torch.randn(H,C,V)
out, qa = {fn}(torch.randn(B,H,Q,D),torch.randn(B,H,Q,R),c,torch.randn(B,K,R),torch.randn(H,C,D),wv)
ctx = c[:,None,:,:].expand(B,H,Q,C)
ref = torch.einsum('bhqc,hcv->bhqv',ctx,wv)
assert qa.shape == (B,H,Q,C) and torch.allclose(out,ref,atol=1e-6)
''',
        },
        {
            "name": "gradient flow through absorbed path",
            "code": '''import torch
B,H,Q,K,C,D,R,V = 1,2,2,3,3,4,2,3
xs = [torch.randn(B,H,Q,D,requires_grad=True),torch.randn(B,H,Q,R,requires_grad=True),torch.randn(B,K,C,requires_grad=True),torch.randn(B,K,R,requires_grad=True),torch.randn(H,C,D,requires_grad=True),torch.randn(H,C,V,requires_grad=True)]
out, qa = {fn}(*xs)
(out.square().sum()+qa.square().sum()).backward()
assert out.requires_grad and qa.requires_grad
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in xs)
''',
        },
    ],
    "solution": '''import torch
import math

def mla_absorbed_attention(q_nope, q_rope, c_kv, k_rope, W_uk, W_uv):
    d_h = q_nope.shape[-1]
    d_r = q_rope.shape[-1]
    q_absorbed = torch.einsum('bhqd,hcd->bhqc', q_nope, W_uk)
    score_nope = torch.einsum('bhqc,bkc->bhqk', q_absorbed, c_kv)
    score_rope = torch.einsum('bhqr,bkr->bhqk', q_rope, k_rope)
    scores = (score_nope + score_rope) / math.sqrt(d_h + d_r)
    attn = torch.softmax(scores, dim=-1)
    latent_ctx = torch.einsum('bhqk,bkc->bhqc', attn, c_kv)
    out = torch.einsum('bhqc,hcv->bhqv', latent_ctx, W_uv)
    return out, q_absorbed
''',
    "demo": '''B,H,Lq,Lk,dc,dh,dr,dv = 2,8,4,128,32,16,8,16
out, absorbed = mla_absorbed_attention(torch.randn(B,H,Lq,dh),torch.randn(B,H,Lq,dr),torch.randn(B,Lk,dc),torch.randn(B,Lk,dr),torch.randn(H,dc,dh),torch.randn(H,dc,dv))
print(out.shape, absorbed.shape)
''',
    "explanation": "Only the compressed cache and shared RoPE key grow with sequence length. Per-head expansion is moved to the short query side and to the post-aggregation output side.",
}
