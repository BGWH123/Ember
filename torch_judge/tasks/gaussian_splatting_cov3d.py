"""3D Gaussian splatting covariance task."""

TASK = {
    "title": "3D Gaussian Splatting Covariance & Quaternion Rotation",
    "title_zh": "3D Gaussian Splatting 协方差矩阵与四元数旋转",
    "difficulty": "Hard",
    "category": "三维与具身感知",
    "function_name": "gaussian_splat_covariance",
    "starter_code": '''def gaussian_splat_covariance(quaternions, scales, eps=1e-8):
    pass
''',
    "hint": "Normalize each (w,x,y,z) quaternion, construct the three rotation rows with stack operations, then compute R @ diag_embed(scales**2) @ R.T.",
    "hint_zh": "归一化每个 (w,x,y,z) 四元数，用 stack 构造三行旋转矩阵，再计算 R @ diag_embed(scales**2) @ R.T。",
    "description_en": "Normalize (N,4) quaternions in (w,x,y,z) order, convert them to (N,3,3) rotations, and rotate diagonal squared scales into 3D covariance matrices. Return (R,Sigma).",
    "description_zh": "归一化按 (w,x,y,z) 排列的 (N,4) 四元数，将其转换为 (N,3,3) 旋转矩阵，并把对角尺度平方旋转成三维协方差矩阵。返回 (R,Sigma)。",
    "theory_en": "A quaternion compactly represents 3D rotation. Independent principal-axis standard deviations form diag(s^2); changing basis by R produces covariance R diag(s^2) R^T.",
    "theory_zh": "四元数紧凑表示三维旋转。独立主轴标准差构成 diag(s^2)；通过 R 变换基底后得到协方差 R diag(s^2) R^T。",
    "diagram_en": "```text\nquaternion -> normalize -> R --\\\nscales -> square -> diagonal ----> R diag(s^2) R^T -> Sigma\n```",
    "diagram_zh": "```text\n四元数 -> 归一化 -> R --\\\n尺度 -> 平方 -> 对角矩阵 ----> R diag(s^2) R^T -> Sigma\n```",
    "tests": [
        {
            "name": "output shapes and symmetry",
            "code": '''import torch
q=torch.randn(5,4); s=torch.rand(5,3)+.1
R,S={fn}(q,s)
assert R.shape==(5,3,3) and S.shape==(5,3,3)
assert torch.allclose(S,S.transpose(-2,-1),atol=1e-6)
''',
        },
        {
            "name": "exact quaternion reference",
            "code": '''import torch
q=torch.tensor([[1.,2.,3.,4.]])
s=torch.tensor([[2.,3.,4.]])
R,S={fn}(q,s,eps=0.0)
w,x,y,z=(q/q.norm(dim=-1,keepdim=True)).unbind(-1)
ref=torch.stack((torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),-1),torch.stack((2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),-1),torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)),-2)
assert torch.equal(R,ref) and torch.equal(S,ref@torch.diag_embed(s.square())@ref.transpose(-2,-1))
''',
        },
        {
            "name": "identity quaternion edge case",
            "code": '''import torch
q=torch.tensor([[1.,0.,0.,0.]]); s=torch.tensor([[1.,2.,3.]])
R,S={fn}(q,s,eps=0.0)
assert torch.equal(R,torch.eye(3).unsqueeze(0))
assert torch.equal(S,torch.diag(torch.tensor([1.,4.,9.])).unsqueeze(0))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
q=torch.randn(3,4,requires_grad=True); s=(torch.rand(3,3)+.2).requires_grad_()
R,S={fn}(q,s); (R.sum()+S.square().sum()).backward()
assert q.grad is not None and s.grad is not None
assert torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()
''',
        },
    ],
    "solution": '''import torch

def gaussian_splat_covariance(quaternions, scales, eps=1e-8):
    q = quaternions / (quaternions.norm(dim=-1, keepdim=True) + eps)
    w, x, y, z = q.unbind(dim=-1)
    row0 = torch.stack((
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
    ), dim=-1)
    row1 = torch.stack((
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
    ), dim=-1)
    row2 = torch.stack((
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ), dim=-1)
    R = torch.stack((row0, row1, row2), dim=-2)
    Sigma = R @ torch.diag_embed(scales.pow(2)) @ R.transpose(-2, -1)
    return R, Sigma
''',
    "demo": '''q = torch.randn(10, 4)
scales = torch.rand(10, 3) + 0.01
R, covariance = gaussian_splat_covariance(q, scales)
print(R.shape, covariance.shape)
''',
    "explanation": "Stacking matrix entries keeps construction differentiable. Squared scales are variances along the Gaussian's local principal axes.",
}
