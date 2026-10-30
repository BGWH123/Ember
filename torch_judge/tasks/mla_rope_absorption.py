"""MLA Decoupled RoPE & Weight Matrix Absorption task."""

TASK = {
    'title': 'MLA Decoupled RoPE & Weight Matrix Absorption',
    'title_zh': 'MLA 解耦 RoPE 与矩阵吸收推理',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'mla_absorbed_attention',
    'starter_code': """def mla_absorbed_attention(q_nope, q_rope, c_kv, k_rope, W_uk, W_uv):
    pass""",
    'hint': """Move the per-head key up-projection onto the content query, score the compressed cache directly, aggregate in latent space, and apply the value up-projection only after attention.""",
    'hint_zh': '把逐头键上投影吸收到内容查询中，直接对压缩缓存打分，在潜空间聚合后再应用值上投影。',
    'description_en': """Implement MLA inference with decoupled positional keys and matrix absorption, avoiding materialization of full per-head K and V caches.""",
    'description_zh': '实现带解耦位置键和矩阵吸收的 MLA 推理，避免物化完整的逐头 K、V 缓存。',
    'theory_en': """Associativity lets q(W_uk)^T be evaluated before interaction with the compressed latent. Likewise, attention can aggregate compressed values first and apply W_uv afterward. The separate RoPE branch preserves positional information.""",
    'theory_zh': '利用结合律，可在压缩潜变量交互前先计算 q(W_uk)^T；同理可先聚合压缩值，再应用 W_uv。独立 RoPE 分支保留位置信息。',
    'diagram_en': r"""```text
q_nope -> absorb W_uk -> score c_kv --\
q_rope -----------------> score k_rope -> softmax -> aggregate c_kv -> W_uv -> out
```""",
    'diagram_zh': r"""```text
q_nope -> 吸收 W_uk -> 对 c_kv 打分 --\
q_rope -----------------> 对 k_rope 打分 -> softmax -> 聚合 c_kv -> W_uv -> 输出
```""",
    "tests": [
        {
            "name": 'absorbed tensor and output shapes',
            "code": """import torch
B,H,Q,K,C,D,R,V = 2,3,4,5,6,7,2,8
out, qa = {fn}(torch.randn(B,H,Q,D),torch.randn(B,H,Q,R),torch.randn(B,K,C),torch.randn(B,K,R),torch.randn(H,C,D),torch.randn(H,C,V))
assert qa.shape == (B,H,Q,C) and out.shape == (B,H,Q,V)""",
        },
        {
            "name": 'exact absorbed reference',
            "code": """import torch
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
assert torch.allclose(out,ro,atol=1e-12,rtol=1e-12)""",
        },
        {
            "name": 'single-key edge case',
            "code": """import torch
B,H,Q,K,C,D,R,V = 2,2,3,1,4,3,2,5
c = torch.randn(B,K,C); wv = torch.randn(H,C,V)
out, qa = {fn}(torch.randn(B,H,Q,D),torch.randn(B,H,Q,R),c,torch.randn(B,K,R),torch.randn(H,C,D),wv)
ctx = c[:,None,:,:].expand(B,H,Q,C)
ref = torch.einsum('bhqc,hcv->bhqv',ctx,wv)
assert qa.shape == (B,H,Q,C) and torch.allclose(out,ref,atol=1e-6)""",
        },
        {
            "name": 'gradient flow through absorbed path',
            "code": """import torch
B,H,Q,K,C,D,R,V = 1,2,2,3,3,4,2,3
xs = [torch.randn(B,H,Q,D,requires_grad=True),torch.randn(B,H,Q,R,requires_grad=True),torch.randn(B,K,C,requires_grad=True),torch.randn(B,K,R,requires_grad=True),torch.randn(H,C,D,requires_grad=True),torch.randn(H,C,V,requires_grad=True)]
out, qa = {fn}(*xs)
(out.square().sum()+qa.square().sum()).backward()
assert out.requires_grad and qa.requires_grad
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in xs)""",
        },
    ],
    'solution': '''def mla_absorbed_attention(q_nope, q_rope, c_kv, k_rope, W_uk, W_uv):
    """实现 MLA 解耦 RoPE 与矩阵吸收推理 (MLA Absorbed Attention)。"""
    d_h = q_nope.shape[-1]
    d_r = q_rope.shape[-1]

    # Step 1: 将键上投影矩阵吸收到内容查询侧
    q_absorbed = torch.einsum("bhqd,hcd->bhqc", q_nope, W_uk)

    # Step 2: 在压缩内容与独立 RoPE 分支上联合打分
    content_scores = torch.einsum("bhqc,bkc->bhqk", q_absorbed, c_kv)
    rope_scores = torch.einsum("bhqr,bkr->bhqk", q_rope, k_rope)
    scores = (content_scores + rope_scores) / math.sqrt(d_h + d_r)
    attn_weights = torch.softmax(scores, dim=-1)

    # Step 3: 先在潜空间聚合，再应用值上投影
    latent_context = torch.einsum("bhqk,bkc->bhqc", attn_weights, c_kv)
    out = torch.einsum("bhqc,hcv->bhqv", latent_context, W_uv)
    return out, q_absorbed''',
    'demo': """torch.manual_seed(42)
B, H, S_q, S_k = 2, 4, 3, 8
d_c, d_h, d_r, d_v = 6, 8, 4, 8
out, absorbed = mla_absorbed_attention(torch.randn(B, H, S_q, d_h), torch.randn(B, H, S_q, d_r), torch.randn(B, S_k, d_c), torch.randn(B, S_k, d_r), torch.randn(H, d_c, d_h), torch.randn(H, d_c, d_v))
print("Output shape:", out.shape)
print("Absorbed query shape:", absorbed.shape)
print("Sample values:", out[0, 0, 0, :4])""",
    'explanation': r"""### 核心思路
矩阵结合律把键上投影吸收到查询侧，使内容分支直接与压缩缓存 `c_kv` 打分；值分支则先在潜空间聚合，再应用 `W_uv`。RoPE 分支独立保留位置信息。

### 代码步骤拆解
- **Step 1（查询吸收）**：用 `W_uk` 将 `q_nope` 转为 `(B, H, S_q, d_c)`。
- **Step 2（联合打分）**：内容分数与 RoPE 分数相加后按 $\sqrt{d_h+d_r}$ 缩放。
- **Step 3（延迟上投影）**：权重先聚合 `c_kv`，再逐头映射到值维。

### 重点与易错点分析
- **注意点 1（einsum 轴）**：`W_uk` 布局为 `(H, d_c, d_h)`，收缩的是 `d_h`，不能交换后两轴。
- **注意点 2（缩放维度）**：两条点积分支联合后使用内容头维与 RoPE 维之和。
- **注意点 3（缓存显存）**：序列增长时只存共享压缩内容和 RoPE 键，避免物化逐头完整 K/V。

### 复杂度分析
- **时间复杂度**：$O(BHS_qS_k(d_c+d_r)+BHS_qd_cd_v)$，联合打分是瓶颈。
- **空间复杂度**：$O(BHS_qS_k+BS_k(d_c+d_r))$，主要为分数矩阵和压缩缓存。""",
}
