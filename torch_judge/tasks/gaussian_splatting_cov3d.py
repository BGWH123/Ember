"""3D Gaussian Splatting Covariance & Quaternion Rotation task."""

TASK = {
    'title': '3D Gaussian Splatting Covariance & Quaternion Rotation',
    'title_zh': '3D Gaussian Splatting 协方差矩阵与四元数旋转',
    'difficulty': 'Hard',
    'category': '三维与具身感知',
    'function_name': 'gaussian_splat_covariance',
    'starter_code': """def gaussian_splat_covariance(quaternions, scales, eps=1e-8):
    pass""",
    'hint': """Normalize each (w,x,y,z) quaternion, construct the three rotation rows with stack operations, then compute R @ diag_embed(scales**2) @ R.T.""",
    'hint_zh': '归一化每个 (w,x,y,z) 四元数，用 stack 构造三行旋转矩阵，再计算 R @ diag_embed(scales**2) @ R.T。',
    'description_en': """Normalize (N,4) quaternions in (w,x,y,z) order, convert them to (N,3,3) rotations, and rotate diagonal squared scales into 3D covariance matrices. Return (R,Sigma).""",
    'description_zh': '归一化按 (w,x,y,z) 排列的 (N,4) 四元数，将其转换为 (N,3,3) 旋转矩阵，并把对角尺度平方旋转成三维协方差矩阵。返回 (R,Sigma)。',
    'theory_en': """A quaternion compactly represents 3D rotation. Independent principal-axis standard deviations form diag(s^2); changing basis by R produces covariance R diag(s^2) R^T.""",
    'theory_zh': '四元数紧凑表示三维旋转。独立主轴标准差构成 diag(s^2)；通过 R 变换基底后得到协方差 R diag(s^2) R^T。',
    'diagram_en': r"""```text
quaternion -> normalize -> R --\
scales -> square -> diagonal ----> R diag(s^2) R^T -> Sigma
```""",
    'diagram_zh': r"""```text
四元数 -> 归一化 -> R --\
尺度 -> 平方 -> 对角矩阵 ----> R diag(s^2) R^T -> Sigma
```""",
    "tests": [
        {
            "name": 'output shapes and symmetry',
            "code": """import torch
q=torch.randn(5,4); s=torch.rand(5,3)+.1
R,S={fn}(q,s)
assert R.shape==(5,3,3) and S.shape==(5,3,3)
assert torch.allclose(S,S.transpose(-2,-1),atol=1e-6)""",
        },
        {
            "name": 'exact quaternion reference',
            "code": """import torch
q=torch.tensor([[1.,2.,3.,4.]])
s=torch.tensor([[2.,3.,4.]])
R,S={fn}(q,s,eps=0.0)
w,x,y,z=(q/q.norm(dim=-1,keepdim=True)).unbind(-1)
ref=torch.stack((torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),-1),torch.stack((2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),-1),torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)),-2)
assert torch.equal(R,ref) and torch.equal(S,ref@torch.diag_embed(s.square())@ref.transpose(-2,-1))""",
        },
        {
            "name": 'identity quaternion edge case',
            "code": """import torch
q=torch.tensor([[1.,0.,0.,0.]]); s=torch.tensor([[1.,2.,3.]])
R,S={fn}(q,s,eps=0.0)
assert torch.equal(R,torch.eye(3).unsqueeze(0))
assert torch.equal(S,torch.diag(torch.tensor([1.,4.,9.])).unsqueeze(0))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
q=torch.randn(3,4,requires_grad=True); s=(torch.rand(3,3)+.2).requires_grad_()
R,S={fn}(q,s); (R.sum()+S.square().sum()).backward()
assert q.grad is not None and s.grad is not None
assert torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()""",
        },
    ],
    'solution': '''def gaussian_splat_covariance(quaternions, scales, eps=1e-8):
    """构造三维高斯旋转与协方差 (Gaussian Splatting Covariance)。"""
    # Step 1: 归一化按 wxyz 排列的四元数
    normalized = quaternions / (quaternions.norm(dim=-1, keepdim=True) + eps)
    w, x, y, z = normalized.unbind(dim=-1)

    # Step 2: 按四元数公式构造旋转矩阵
    row_0 = torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)), dim=-1)
    row_1 = torch.stack((2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)), dim=-1)
    row_2 = torch.stack((2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1)
    rotation = torch.stack((row_0, row_1, row_2), dim=-2)

    # Step 3: 将局部尺度方差旋转到世界坐标
    local_covariance = torch.diag_embed(scales.square())
    covariance = rotation @ local_covariance @ rotation.transpose(-2, -1)
    return rotation, covariance''',
    'demo': """torch.manual_seed(42)
quaternions = torch.randn(5, 4)
scales = torch.rand(5, 3) + 0.1
rotation, covariance = gaussian_splat_covariance(quaternions, scales)
print("Output shapes:", rotation.shape, covariance.shape)
print("Symmetry error:", (covariance - covariance.transpose(-2, -1)).abs().max().item())
print("Sample diagonal:", covariance[0].diagonal())""",
    'explanation': r"""### 核心思路
先将 $(w,x,y,z)$ 四元数单位化并转换为旋转矩阵 $R$，再用 $\Sigma=R\operatorname{diag}(s^2)R^T$ 将局部轴向方差变换到世界坐标。

### 代码步骤拆解
- **Step 1（四元数归一化）**：`normalized` 保持 `(N, 4)`，`unbind` 得到四个 `(N,)` 分量。
- **Step 2（旋转构造）**：三行堆叠后 `rotation` 为 `(N, 3, 3)`。
- **Step 3（协方差变换）**：`diag_embed` 生成局部方差矩阵，批量矩阵乘得到协方差。

### 重点与易错点分析
- **注意点 1（分量顺序）**：公式按 `w, x, y, z`，若误用 xyzw 顺序会得到错误旋转。
- **注意点 2（eps 位置）**：`eps` 加在范数之后，零四元数可避免除零；`eps=0` 时与精确参考一致。
- **注意点 3（方差平方）**：`scales` 是标准差，构造协方差对角线时必须平方。

### 复杂度分析
- **时间复杂度**：$O(N)$，每个高斯只进行固定规模运算。
- **空间复杂度**：$O(N)$，旋转与协方差均为固定大小矩阵。""",
}
