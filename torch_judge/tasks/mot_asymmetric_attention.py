"""Mixture-of-Transformers asymmetric attention task."""

TASK = {
    "title": "Mixture-of-Transformers (MoT) Asymmetric Cross-Modal Attention",
    "title_zh": "Mixture-of-Transformers (MoT) 非对称跨模态注意力",
    "difficulty": "Hard",
    "category": "多模态",
    "function_name": "mot_asymmetric_attention",
    "starter_code": '''def mot_asymmetric_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm):
    pass
''',
    "hint": "Concatenate both streams, then combine a causal lower-triangular AR query block with an all-true diffusion query block. Use the boolean mask before softmax.",
    "hint_zh": "先拼接两条流，再将 AR 查询的因果下三角块与全真的扩散查询块组合，在 softmax 前应用布尔掩码。",
    "description_en": "Compute single-head scaled dot-product attention over concatenated autoregressive and diffusion streams. AR queries are causal and cannot read diffusion tokens; diffusion queries can read every token. Return both split outputs and the boolean mask.",
    "description_zh": "在拼接的自回归流和扩散流上计算单头缩放点积注意力。AR 查询具有因果性且不能读取扩散 token；扩散查询可读取所有 token。返回拆分输出和布尔掩码。",
    "theory_en": "The asymmetric mask protects autoregressive generation from future and diffusion information while exposing the complete reasoner context to the bidirectional diffusion generator.",
    "theory_zh": "非对称掩码保护自回归生成不接触未来及扩散信息，同时向双向扩散生成器开放完整推理器上下文。",
    "diagram_en": "```text\nkeys:     AR          DM\nAR q:   causal      blocked\nDM q:   all visible all visible\n```",
    "diagram_zh": "```text\n键：       AR          DM\nAR 查询： 因果可见      禁止\nDM 查询： 全部可见      全部可见\n```",
    "tests": [
        {
            "name": "shapes and mask structure",
            "code": '''import torch
a=[torch.randn(2,3,4) for _ in range(3)]; d=[torch.randn(2,5,4) for _ in range(3)]
oa,od,m={fn}(*(a+d))
ref=torch.zeros(8,8,dtype=torch.bool); ref[:3,:3]=torch.tril(torch.ones(3,3,dtype=torch.bool)); ref[3:]=True
assert oa.shape==(2,3,4) and od.shape==(2,5,4) and m.dtype==torch.bool
assert torch.equal(m.cpu(),ref)
''',
        },
        {
            "name": "exact masked attention",
            "code": '''import torch
torch.manual_seed(3)
qa,ka,va=[torch.randn(1,2,3) for _ in range(3)]; qd,kd,vd=[torch.randn(1,2,3) for _ in range(3)]
oa,od,m={fn}(qa,ka,va,qd,kd,vd)
Q=torch.cat((qa,qd),1); K=torch.cat((ka,kd),1); V=torch.cat((va,vd),1)
s=(Q@K.transpose(-2,-1))/(3**.5); ref=torch.softmax(s.masked_fill(~m,float('-inf')),-1)@V
assert torch.equal(oa,ref[:,:2]) and torch.equal(od,ref[:,2:])
''',
        },
        {
            "name": "AR output ignores diffusion edge",
            "code": '''import torch
qa=torch.zeros(1,2,2); ka=torch.zeros_like(qa); va=torch.tensor([[[1.,2.],[3.,4.]]])
qd=torch.zeros(1,1,2); kd=torch.zeros_like(qd); vd=torch.tensor([[[100.,200.]]])
oa,od,m={fn}(qa,ka,va,qd,kd,vd)
assert torch.equal(oa[:,0],va[:,0]) and torch.equal(oa[:,1],va.mean(1))
assert torch.allclose(od,torch.tensor([[[104/3,206/3]]]))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
a=[torch.randn(1,3,4,requires_grad=True) for _ in range(3)]
d=[torch.randn(1,2,4,requires_grad=True) for _ in range(3)]
oa,od,m={fn}(*(a+d)); (oa.square().sum()+od.square().sum()).backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in a+d)
''',
        },
    ],
    "solution": '''import torch
import math

def mot_asymmetric_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm):
    length_ar = q_ar.size(1)
    Q = torch.cat((q_ar, q_dm), dim=1)
    K = torch.cat((k_ar, k_dm), dim=1)
    V = torch.cat((v_ar, v_dm), dim=1)
    total_length = Q.size(1)
    mask = torch.ones(
        total_length, total_length, dtype=torch.bool, device=Q.device
    )
    mask[:length_ar] = False
    mask[:length_ar, :length_ar] = torch.tril(torch.ones(
        length_ar, length_ar, dtype=torch.bool, device=Q.device
    ))
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    scores = scores.masked_fill(~mask, float('-inf'))
    output = torch.softmax(scores, dim=-1) @ V
    return output[:, :length_ar], output[:, length_ar:], mask
''',
    "demo": '''ar = [torch.randn(2, 8, 64) for _ in range(3)]
dm = [torch.randn(2, 32, 64) for _ in range(3)]
out_ar, out_dm, mask = mot_asymmetric_attention(*(ar + dm))
print(out_ar.shape, out_dm.shape, mask.shape)
''',
    "explanation": "Rows encode queries and columns encode keys. AR rows contain only a lower triangle in the AR columns, whereas every entry of every diffusion row is visible.",
}
