"""Multimodal 3D Rotary Position Embedding (M-RoPE) task."""

TASK = {
    'title': 'Multimodal 3D Rotary Position Embedding (M-RoPE)',
    'title_zh': '多模态三维旋转位置编码（M-RoPE）',
    'difficulty': 'Hard',
    'category': '多模态',
    'function_name': 'apply_mrope_3d',
    'starter_code': """def apply_mrope_3d(q, k, pos_3d, section_dims=(4, 6, 6), base=10000.0):
    pass""",
    'hint': """Split each embedding into temporal, height, and width sections. For each section, build half as many frequencies and rotate adjacent even/odd coordinate pairs before concatenating.""",
    'hint_zh': '将嵌入拆分为时间、高度和宽度三段。每段构造一半数量的频率，旋转相邻偶数/奇数坐标对后再拼接。',
    'description_en': """Apply axis-specific rotary embeddings to q and k of shape (B,S,D), using per-token temporal, height, and width coordinates in pos_3d and even section dimensions summing to D.""",
    'description_zh': '对形状为 (B,S,D) 的 q 和 k 应用分轴旋转位置编码，使用 pos_3d 中逐 token 的时间、高度、宽度坐标；三个偶数段维度之和为 D。',
    'theory_en': """RoPE represents position by orthogonal rotations of adjacent feature pairs. M-RoPE allocates disjoint channel sections to independent temporal and spatial coordinates, retaining norm while encoding three-dimensional relative phase.""",
    'theory_zh': 'RoPE 通过相邻特征对的正交旋转表示位置。M-RoPE 将互不重叠的通道段分配给时间和空间坐标，在保持范数的同时编码三维相对相位。',
    'diagram_en': """```text
features -> [temporal | height | width]
positions -> [t rotation | h rotation | w rotation] -> concatenate
```""",
    'diagram_zh': """```text
特征 -> [时间段 | 高度段 | 宽度段]
坐标 -> [t 旋转 | h 旋转 | w 旋转] -> 拼接
```""",
    "tests": [
        {
            "name": 'shape and norm preservation',
            "code": """import torch
q=torch.randn(2,5,16); k=torch.randn(2,5,16); p=torch.randint(0,8,(2,5,3))
qr,kr={fn}(q,k,p)
assert qr.shape==q.shape and kr.shape==k.shape
assert torch.allclose(qr.norm(dim=-1),q.norm(dim=-1),atol=1e-5)
assert torch.allclose(kr.norm(dim=-1),k.norm(dim=-1),atol=1e-5)""",
        },
        {
            "name": 'exact axis reference',
            "code": """import torch
q=torch.arange(1.,13.).reshape(1,1,12); k=-q; p=torch.tensor([[[1.,2.,3.]]]); dims=(2,4,6); base=100.
qr,kr={fn}(q,k,p,dims,base)
def rot(x,d,pos):
 f=1/(base**(torch.arange(0,d,2).float()/d)); a=pos*f
 e=x[...,0::2]; o=x[...,1::2]
 return torch.stack((e*torch.cos(a)-o*torch.sin(a),e*torch.sin(a)+o*torch.cos(a)),-1).flatten(-2)
parts=torch.split(q,dims,-1); ref=torch.cat([rot(parts[i],dims[i],p[...,i:i+1]) for i in range(3)],-1)
assert torch.equal(qr,ref) and torch.equal(kr,-ref)""",
        },
        {
            "name": 'zero position identity edge',
            "code": """import torch
q=torch.randn(1,3,8); k=torch.randn(1,3,8); p=torch.zeros(1,3,3)
qr,kr={fn}(q,k,p,(2,2,4))
assert torch.equal(qr,q) and torch.equal(kr,k)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
q=torch.randn(2,3,12,requires_grad=True); k=torch.randn(2,3,12,requires_grad=True)
p=torch.randn(2,3,3,requires_grad=True); qr,kr={fn}(q,k,p,(4,4,4))
(qr.square().sum()+kr.sum()).backward()
assert q.grad is not None and k.grad is not None and p.grad is not None
assert torch.isfinite(p.grad).all()""",
        },
    ],
    'solution': '''def apply_mrope_3d(q, k, pos_3d, section_dims=(4, 6, 6), base=10000.0):
    """多模态三维旋转位置编码（M-RoPE） (Multimodal 3D Rotary Position Embedding (M-RoPE))。"""
    # Step 1: 将通道切分为时间、高度、宽度三段并分别保存 q、k 分量
    q_sections = torch.split(q, section_dims, dim=-1)
    k_sections = torch.split(k, section_dims, dim=-1)
    q_rotated = []
    k_rotated = []
    # Step 2: 为每个轴构造几何频率，并用该轴坐标旋转对应段的偶奇通道对
    for axis in range(3):
        section_dim = section_dims[axis]
        frequencies = 1.0 / base ** (torch.arange(0, section_dim, 2, device=q.device).float() / section_dim)
        angles = pos_3d[..., axis].float().unsqueeze(-1) * frequencies
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        q_even, q_odd = (q_sections[axis][..., 0::2], q_sections[axis][..., 1::2])
        k_even, k_odd = (k_sections[axis][..., 0::2], k_sections[axis][..., 1::2])
        q_rotated.append(torch.stack((q_even * cos_angles - q_odd * sin_angles, q_even * sin_angles + q_odd * cos_angles), dim=-1).flatten(-2))
        k_rotated.append(torch.stack((k_even * cos_angles - k_odd * sin_angles, k_even * sin_angles + k_odd * cos_angles), dim=-1).flatten(-2))
    # Step 3: 按时间、高度、宽度次序拼回完整 D 维旋转查询与键
    return (torch.cat(q_rotated, dim=-1), torch.cat(k_rotated, dim=-1))''',
    'demo': """torch.manual_seed(42)
q = torch.randn(2, 32, 16)
k = torch.randn_like(q)
positions = torch.randint(0, 16, (2, 32, 3))
q_rot, k_rot = apply_mrope_3d(q, k, positions)
print('旋转后 q/k 形状:', tuple(q_rot.shape), tuple(k_rot.shape))
print('q 范数最大偏差:', f'{(q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item():.2e}')
print('首个 token 三维坐标:', positions[0, 0].tolist())""",
    'explanation': r"""### 核心思路
M-RoPE 把 D 个通道分配给时间 t、高度 h、宽度 w 三个独立轴。对轴 a 的偶奇通道对 $(x_{2i},x_{2i+1})$，相位为 $\theta_{a,i}=p_a\,b^{-2i/d_a}$，应用二维旋转矩阵 $\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$。q 与 k 使用同一相位，因此点积编码三轴相对坐标，同时正交旋转保持每对通道的范数。

### 代码步骤拆解
- **Step 1（三轴切片）**：`torch.split` 按 `section_dims=(d_t,d_h,d_w)` 将 `(B,S,D)` 切成三个互不重叠的通道段，要求三段之和为 D。
- **Step 2（轴内成对旋转）**：每段生成 `section_dim/2` 个 `frequencies`；`pos_3d[...,axis,None]` 与其广播得到 `(B,S,d_a/2)` 角度，并同时旋转 q、k 的偶数和奇数位置。
- **Step 3（恢复特征布局）**：`stack(..., dim=-1).flatten(-2)` 恢复交错偶奇顺序，三个旋转段再沿最后一维拼回 `(B,S,D)`。

### 重点与易错点分析
- **注意点 1（每段必须为偶数）**：旋转以相邻两维为一组，任一 `section_dim` 为奇数都会让 even/odd 张量宽度不一致；不能只要求总维度 D 为偶数。
- **注意点 2（三轴不能共用位置）**：时间段只乘 `pos_3d[...,0]`，空间段分别乘高度和宽度；把三坐标先相加会丢失轴向结构。
- **注意点 3（交错布局）**：本题配对 `(0,1),(2,3),...`，必须 stack 后 flatten；直接 `cat([rot_even,rot_odd])` 会把偶数通道和奇数通道分成两块，改变参考布局。

### 复杂度分析
- **时间复杂度**：$O(BSD)$，三段的三角函数和逐元素旋转总通道数恰为 D。
- **空间复杂度**：$O(BSD)$，角度、正余弦及旋转后的 q/k 分段是主要中间张量。""",
}
