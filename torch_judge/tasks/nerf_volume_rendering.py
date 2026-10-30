"""NeRF alpha-compositing volume rendering task."""

TASK = {
    "title": "NeRF Volume Rendering (Alpha Compositing)",
    "title_zh": "NeRF 体渲染方程（Alpha 合成）",
    "difficulty": "Hard",
    "category": "三维与具身感知",
    "function_name": "nerf_volume_render",
    "starter_code": '''def nerf_volume_render(rgb, sigma, z_vals, white_bkgd=False):
    pass
''',
    "hint": "Build adjacent depth intervals and duplicate the final interval (use one for a single sample). Compute alpha, exclusive transmittance with a prepended one, weights, then weighted color and depth.",
    "hint_zh": "构造相邻深度间隔并复制最后一个间隔（单样本时使用 1）。计算 alpha、前置 1 的排他透射率、权重，再求加权颜色与深度。",
    "description_en": "Render RGB and depth from NeRF samples. rgb is (R,S,3), sigma and sorted z_vals are (R,S). For S>1, deltas concatenates adjacent differences with the final difference repeated; for S=1 it is ones_like(z_vals). Return (comp_rgb,depth,weights), optionally compositing residual opacity over white.",
    "description_zh": "从 NeRF 样本渲染 RGB 与深度。rgb 为 (R,S,3)，sigma 与有序 z_vals 为 (R,S)。S>1 时，deltas 由相邻差分和重复的最后差分拼接；S=1 时使用 ones_like(z_vals)。返回 (comp_rgb,depth,weights)，并可将剩余透明度合成到白色背景。",
    "theory_en": "Density over each interval gives alpha=1-exp(-sigma*delta). Exclusive transmittance is the probability a ray reaches a sample; multiplying it by alpha gives that sample's termination weight.",
    "theory_zh": "每个区间的密度给出 alpha=1-exp(-sigma*delta)。排他透射率表示射线到达样本的概率，与 alpha 相乘得到该样本的终止权重。",
    "diagram_en": "```text\n(sigma, depth intervals) -> alpha -> exclusive transmittance -> weights\nweights * RGB -> color        weights * z -> depth\n```",
    "diagram_zh": "```text\n(sigma, 深度间隔) -> alpha -> 排他透射率 -> 权重\n权重 * RGB -> 颜色             权重 * z -> 深度\n```",
    "tests": [
        {
            "name": "output shapes and finite values",
            "code": '''import torch
rgb=torch.rand(4,6,3); sigma=torch.rand(4,6); z=torch.arange(6.).expand(4,6)
c,d,w={fn}(rgb,sigma,z)
assert c.shape==(4,3) and d.shape==(4,) and w.shape==(4,6)
assert torch.isfinite(c).all() and torch.isfinite(d).all() and torch.isfinite(w).all()
''',
        },
        {
            "name": "exact alpha compositing reference",
            "code": '''import torch
rgb=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
sigma=torch.tensor([[.2,.4,.6]]); z=torch.tensor([[1.,2.,4.]])
c,d,w={fn}(rgb,sigma,z)
delta=torch.tensor([[1.,2.,2.]]); a=1-torch.exp(-sigma*delta)
T=torch.cumprod(torch.cat((torch.ones(1,1),1-a[:,:-1]+1e-10),-1),-1); rw=T*a
assert torch.equal(w,rw) and torch.equal(c,(rw.unsqueeze(-1)*rgb).sum(1)) and torch.equal(d,(rw*z).sum(1))
''',
        },
        {
            "name": "single sample and white background edge",
            "code": '''import torch
rgb=torch.tensor([[[.2,.4,.6]]]); sigma=torch.tensor([[0.]]); z=torch.tensor([[3.]])
c,d,w={fn}(rgb,sigma,z,white_bkgd=True)
assert torch.equal(w,torch.zeros(1,1)) and torch.equal(d,torch.zeros(1))
assert torch.equal(c,torch.ones(1,3))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
rgb=torch.rand(2,4,3,requires_grad=True); sigma=torch.rand(2,4,requires_grad=True)
z=torch.tensor([[0.,.5,1.,2.],[0.,.2,.8,1.5]],requires_grad=True)
c,d,w={fn}(rgb,sigma,z,True); (c.sum()+d.sum()+w.sum()).backward()
assert rgb.grad is not None and sigma.grad is not None and z.grad is not None
assert torch.isfinite(rgb.grad).all() and torch.isfinite(sigma.grad).all() and torch.isfinite(z.grad).all()
''',
        },
    ],
    "solution": '''import torch

def nerf_volume_render(rgb, sigma, z_vals, white_bkgd=False):
    if z_vals.size(1) > 1:
        adjacent = z_vals[:, 1:] - z_vals[:, :-1]
        deltas = torch.cat((adjacent, adjacent[:, -1:]), dim=-1)
    else:
        deltas = torch.ones_like(z_vals)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    prefix = torch.ones_like(alpha[:, :1])
    transmittance = torch.cumprod(
        torch.cat((prefix, 1.0 - alpha[:, :-1] + 1e-10), dim=-1), dim=-1
    )
    weights = transmittance * alpha
    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - weights.sum(dim=1, keepdim=True))
    depth = (weights * z_vals).sum(dim=1)
    return comp_rgb, depth, weights
''',
    "demo": '''rgb = torch.rand(16, 64, 3)
sigma = torch.rand(16, 64)
z_vals = torch.linspace(0, 1, 64).expand(16, 64)
print([x.shape for x in nerf_volume_render(rgb, sigma, z_vals)])
''',
    "explanation": "The cumulative product excludes the current alpha by prepending one and using only prior survival factors. White compositing adds residual ray probability equally to RGB channels.",
}
