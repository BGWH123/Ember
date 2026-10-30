"""NeRF Volume Rendering (Alpha Compositing) task."""

TASK = {
    'title': 'NeRF Volume Rendering (Alpha Compositing)',
    'title_zh': 'NeRF 体渲染方程（Alpha 合成）',
    'difficulty': 'Hard',
    'category': '三维与具身感知',
    'function_name': 'nerf_volume_render',
    'starter_code': """def nerf_volume_render(rgb, sigma, z_vals, white_bkgd=False):
    pass""",
    'hint': """Build adjacent depth intervals and duplicate the final interval (use one for a single sample). Compute alpha, exclusive transmittance with a prepended one, weights, then weighted color and depth.""",
    'hint_zh': '构造相邻深度间隔并复制最后一个间隔（单样本时使用 1）。计算 alpha、前置 1 的排他透射率、权重，再求加权颜色与深度。',
    'description_en': """Render RGB and depth from NeRF samples. rgb is (R,S,3), sigma and sorted z_vals are (R,S). For S>1, deltas concatenates adjacent differences with the final difference repeated; for S=1 it is ones_like(z_vals). Return (comp_rgb,depth,weights), optionally compositing residual opacity over white.""",
    'description_zh': """从 NeRF 样本渲染 RGB 与深度。rgb 为 (R,S,3)，sigma 与有序 z_vals 为 (R,S)。S>1 时，deltas 由相邻差分和重复的最后差分拼接；S=1 时使用 ones_like(z_vals)。返回 (comp_rgb,depth,weights)，并可将剩余透明度合成到白色背景。""",
    'theory_en': """Density over each interval gives alpha=1-exp(-sigma*delta). Exclusive transmittance is the probability a ray reaches a sample; multiplying it by alpha gives that sample's termination weight.""",
    'theory_zh': '每个区间的密度给出 alpha=1-exp(-sigma*delta)。排他透射率表示射线到达样本的概率，与 alpha 相乘得到该样本的终止权重。',
    'diagram_en': """```text
(sigma, depth intervals) -> alpha -> exclusive transmittance -> weights
weights * RGB -> color        weights * z -> depth
```""",
    'diagram_zh': """```text
(sigma, 深度间隔) -> alpha -> 排他透射率 -> 权重
权重 * RGB -> 颜色             权重 * z -> 深度
```""",
    "tests": [
        {
            "name": 'output shapes and finite values',
            "code": """import torch
rgb=torch.rand(4,6,3); sigma=torch.rand(4,6); z=torch.arange(6.).expand(4,6)
c,d,w={fn}(rgb,sigma,z)
assert c.shape==(4,3) and d.shape==(4,) and w.shape==(4,6)
assert torch.isfinite(c).all() and torch.isfinite(d).all() and torch.isfinite(w).all()""",
        },
        {
            "name": 'exact alpha compositing reference',
            "code": """import torch
rgb=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
sigma=torch.tensor([[.2,.4,.6]]); z=torch.tensor([[1.,2.,4.]])
c,d,w={fn}(rgb,sigma,z)
delta=torch.tensor([[1.,2.,2.]]); a=1-torch.exp(-sigma*delta)
T=torch.cumprod(torch.cat((torch.ones(1,1),1-a[:,:-1]+1e-10),-1),-1); rw=T*a
assert torch.equal(w,rw) and torch.equal(c,(rw.unsqueeze(-1)*rgb).sum(1)) and torch.equal(d,(rw*z).sum(1))""",
        },
        {
            "name": 'single sample and white background edge',
            "code": """import torch
rgb=torch.tensor([[[.2,.4,.6]]]); sigma=torch.tensor([[0.]]); z=torch.tensor([[3.]])
c,d,w={fn}(rgb,sigma,z,white_bkgd=True)
assert torch.equal(w,torch.zeros(1,1)) and torch.equal(d,torch.zeros(1))
assert torch.equal(c,torch.ones(1,3))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
rgb=torch.rand(2,4,3,requires_grad=True); sigma=torch.rand(2,4,requires_grad=True)
z=torch.tensor([[0.,.5,1.,2.],[0.,.2,.8,1.5]],requires_grad=True)
c,d,w={fn}(rgb,sigma,z,True); (c.sum()+d.sum()+w.sum()).backward()
assert rgb.grad is not None and sigma.grad is not None and z.grad is not None
assert torch.isfinite(rgb.grad).all() and torch.isfinite(sigma.grad).all() and torch.isfinite(z.grad).all()""",
        },
    ],
    'solution': '''def nerf_volume_render(rgb, sigma, z_vals, white_bkgd=False):
    """NeRF 体渲染方程（Alpha 合成） (NeRF Volume Rendering (Alpha Compositing))。"""
    # Step 1: 由相邻采样深度构造射线区间，末样本复用最后一个区间长度
    if z_vals.size(1) > 1:
        adjacent = z_vals[:, 1:] - z_vals[:, :-1]
        deltas = torch.cat((adjacent, adjacent[:, -1:]), dim=-1)
    else:
        deltas = torch.ones_like(z_vals)
    # Step 2: 将密度转为 alpha，并用排他累乘计算到达各采样点的透射率与终止权重
    alpha = 1.0 - torch.exp(-sigma * deltas)
    prefix = torch.ones_like(alpha[:, :1])
    transmittance = torch.cumprod(torch.cat((prefix, 1.0 - alpha[:, :-1] + 1e-10), dim=-1), dim=-1)
    weights = transmittance * alpha
    # Step 3: 用终止权重积分颜色和深度，并按剩余透射率合成可选白背景
    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - weights.sum(dim=1, keepdim=True))
    depth = (weights * z_vals).sum(dim=1)
    return (comp_rgb, depth, weights)''',
    'demo': """torch.manual_seed(42)
rgb = torch.rand(16, 64, 3)
sigma = torch.rand(16, 64)
z_vals = torch.linspace(0, 1, 64).expand(16, 64)
comp_rgb, depth, weights = nerf_volume_render(rgb, sigma, z_vals, white_bkgd=True)
print('颜色/深度/权重形状:', tuple(comp_rgb.shape), tuple(depth.shape), tuple(weights.shape))
print('首条射线累计不透明度:', f'{weights[0].sum().item():.4f}')
print('平均渲染深度:', f'{depth.mean().item():.4f}')""",
    'explanation': r"""### 核心思路
NeRF 把每条射线上的离散密度转成吸收概率。区间 i 的不透明度为 $\alpha_i=1-\exp(-\sigma_i\delta_i)$，到达该区间且未被前面样本吸收的概率为 $T_i=\prod_{j<i}(1-\alpha_j)$，终止权重 $w_i=T_i\alpha_i$。颜色和深度分别是 $\sum_iw_i c_i$ 与 $\sum_iw_i z_i$；白背景再接收剩余质量 $1-\sum_iw_i$。

### 代码步骤拆解
- **Step 1（离散化射线区间）**：对 `(R,S)` 的 `z_vals` 做相邻差分；为保持 S 个密度各有区间，末尾复制最后一个差分，单样本射线按题意使用长度 1。
- **Step 2（排他透射率）**：由 `sigma*deltas` 得到 `alpha`；在 `1-alpha[:,:-1]` 前补 1 再 `cumprod`，确保第 i 项只乘 $j<i$，而不错误包含自身 alpha。
- **Step 3（期望量与背景）**：`weights[...,None]` 广播到 RGB 三通道求和，`weights*z_vals` 求深度；白色背景给每个颜色通道加相同剩余透明度。

### 重点与易错点分析
- **注意点 1（exclusive cumprod）**：若直接 `cumprod(1-alpha)`，第一个样本透射率会变成 $1-\alpha_0$；前置 1 是实现 $T_0=1$ 的关键。
- **注意点 2（深度必须有序）**：`deltas` 假设 `z_vals` 沿射线递增；乱序会产生负区间，使 alpha 和权重失去概率意义。
- **注意点 3（白背景只补颜色）**：剩余透射率代表射线逃逸并看到白色，不对应有限采样深度，因此只加到 `comp_rgb`，不能加到 `depth` 或已有 `weights`。

### 复杂度分析
- **时间复杂度**：$O(RS)$，R 条射线各扫描 S 个样本，指数、累乘和加权积分均为线性成本。
- **空间复杂度**：$O(RS)$，`deltas`、`alpha`、`transmittance`、`weights` 为主要中间张量；合成颜色仅占 $O(R)$。""",
}
