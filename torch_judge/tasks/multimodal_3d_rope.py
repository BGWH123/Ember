"""Multimodal three-axis rotary position embedding task."""

TASK = {
    "title": "Multimodal 3D Rotary Position Embedding (M-RoPE)",
    "title_zh": "多模态三维旋转位置编码（M-RoPE）",
    "difficulty": "Hard",
    "category": "多模态",
    "function_name": "apply_mrope_3d",
    "starter_code": '''def apply_mrope_3d(q, k, pos_3d, section_dims=(4, 6, 6), base=10000.0):
    pass
''',
    "hint": "Split each embedding into temporal, height, and width sections. For each section, build half as many frequencies and rotate adjacent even/odd coordinate pairs before concatenating.",
    "hint_zh": "将嵌入拆分为时间、高度和宽度三段。每段构造一半数量的频率，旋转相邻偶数/奇数坐标对后再拼接。",
    "description_en": "Apply axis-specific rotary embeddings to q and k of shape (B,S,D), using per-token temporal, height, and width coordinates in pos_3d and even section dimensions summing to D.",
    "description_zh": "对形状为 (B,S,D) 的 q 和 k 应用分轴旋转位置编码，使用 pos_3d 中逐 token 的时间、高度、宽度坐标；三个偶数段维度之和为 D。",
    "theory_en": "RoPE represents position by orthogonal rotations of adjacent feature pairs. M-RoPE allocates disjoint channel sections to independent temporal and spatial coordinates, retaining norm while encoding three-dimensional relative phase.",
    "theory_zh": "RoPE 通过相邻特征对的正交旋转表示位置。M-RoPE 将互不重叠的通道段分配给时间和空间坐标，在保持范数的同时编码三维相对相位。",
    "diagram_en": "```text\nfeatures -> [temporal | height | width]\npositions -> [t rotation | h rotation | w rotation] -> concatenate\n```",
    "diagram_zh": "```text\n特征 -> [时间段 | 高度段 | 宽度段]\n坐标 -> [t 旋转 | h 旋转 | w 旋转] -> 拼接\n```",
    "tests": [
        {
            "name": "shape and norm preservation",
            "code": '''import torch
q=torch.randn(2,5,16); k=torch.randn(2,5,16); p=torch.randint(0,8,(2,5,3))
qr,kr={fn}(q,k,p)
assert qr.shape==q.shape and kr.shape==k.shape
assert torch.allclose(qr.norm(dim=-1),q.norm(dim=-1),atol=1e-5)
assert torch.allclose(kr.norm(dim=-1),k.norm(dim=-1),atol=1e-5)
''',
        },
        {
            "name": "exact axis reference",
            "code": '''import torch
q=torch.arange(1.,13.).reshape(1,1,12); k=-q; p=torch.tensor([[[1.,2.,3.]]]); dims=(2,4,6); base=100.
qr,kr={fn}(q,k,p,dims,base)
def rot(x,d,pos):
 f=1/(base**(torch.arange(0,d,2).float()/d)); a=pos*f
 e=x[...,0::2]; o=x[...,1::2]
 return torch.stack((e*torch.cos(a)-o*torch.sin(a),e*torch.sin(a)+o*torch.cos(a)),-1).flatten(-2)
parts=torch.split(q,dims,-1); ref=torch.cat([rot(parts[i],dims[i],p[...,i:i+1]) for i in range(3)],-1)
assert torch.equal(qr,ref) and torch.equal(kr,-ref)
''',
        },
        {
            "name": "zero position identity edge",
            "code": '''import torch
q=torch.randn(1,3,8); k=torch.randn(1,3,8); p=torch.zeros(1,3,3)
qr,kr={fn}(q,k,p,(2,2,4))
assert torch.equal(qr,q) and torch.equal(kr,k)
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
q=torch.randn(2,3,12,requires_grad=True); k=torch.randn(2,3,12,requires_grad=True)
p=torch.randn(2,3,3,requires_grad=True); qr,kr={fn}(q,k,p,(4,4,4))
(qr.square().sum()+kr.sum()).backward()
assert q.grad is not None and k.grad is not None and p.grad is not None
assert torch.isfinite(p.grad).all()
''',
        },
    ],
    "solution": '''import torch

def apply_mrope_3d(q, k, pos_3d, section_dims=(4, 6, 6), base=10000.0):
    q_sections = torch.split(q, section_dims, dim=-1)
    k_sections = torch.split(k, section_dims, dim=-1)
    q_rotated = []
    k_rotated = []
    for axis in range(3):
        dim = section_dims[axis]
        freqs = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=q.device).float() / dim)
        )
        angles = pos_3d[..., axis].float().unsqueeze(-1) * freqs
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        q_even, q_odd = q_sections[axis][..., 0::2], q_sections[axis][..., 1::2]
        k_even, k_odd = k_sections[axis][..., 0::2], k_sections[axis][..., 1::2]
        q_rotated.append(torch.stack(
            (q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1
        ).flatten(-2))
        k_rotated.append(torch.stack(
            (k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1
        ).flatten(-2))
    return torch.cat(q_rotated, dim=-1), torch.cat(k_rotated, dim=-1)
''',
    "demo": '''q = torch.randn(2, 32, 16)
k = torch.randn_like(q)
positions = torch.randint(0, 16, (2, 32, 3))
q_rot, k_rot = apply_mrope_3d(q, k, positions)
print(q_rot.shape, k_rot.shape)
''',
    "explanation": "Each axis uses its own coordinates and frequency ladder. Stacking the transformed even and odd values before flattening restores the original adjacent-pair channel order.",
}
