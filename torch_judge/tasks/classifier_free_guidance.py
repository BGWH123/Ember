"""Classifier-free guidance and standard-deviation rescaling task."""

TASK = {
    "title": "Classifier-Free Guidance (CFG) & Std Rescaling",
    "title_zh": "无分类器引导（CFG）与标准差重缩放",
    "difficulty": "Easy",
    "category": "扩散与流模型",
    "function_name": "cfg_guided_prediction",
    "starter_code": '''def cfg_guided_prediction(cond_pred, uncond_pred, guidance_scale=7.5, rescale_phi=0.0, eps=1e-8):
    pass
''',
    "hint": "Extrapolate from the unconditional prediction toward the conditional one. For rescaling, compute population standard deviations over every non-batch axis and blend the rescaled result.",
    "hint_zh": "从无条件预测向条件预测外推。重缩放时，在所有非批次维上计算总体标准差，并混合重缩放结果。",
    "description_en": "Implement classifier-free guidance for predictions shaped (B,...), with optional per-sample standard-deviation rescaling. Return a tensor with the input shape.",
    "description_zh": "对形状为 (B,...) 的预测实现无分类器引导，并可选地按样本进行标准差重缩放。返回与输入形状相同的张量。",
    "theory_en": "CFG combines conditional and unconditional model evaluations as u+s(c-u). Large scales can over-amplify predictions; matching each sample's standard deviation to the conditional prediction mitigates this effect.",
    "theory_zh": "CFG 按 u+s(c-u) 组合条件与无条件模型输出。较大的引导尺度可能放大输出；将每个样本的标准差匹配到条件预测可缓解该问题。",
    "diagram_en": "```text\ncond c --\\\n          u + scale*(c-u) -> optional std match -> blend -> output\nuncond u-/\n```",
    "diagram_zh": "```text\n条件 c --\\\n           u + scale*(c-u) -> 可选标准差匹配 -> 混合 -> 输出\n无条件 u-/\n```",
    "tests": [
        {
            "name": "shape and standard CFG",
            "code": '''import torch
c, u = torch.randn(2,3,4), torch.randn(2,3,4)
out = {fn}(c, u, guidance_scale=2.5)
assert out.shape == c.shape
assert torch.equal(out, u + 2.5 * (c-u))
''',
        },
        {
            "name": "exact rescaling reference",
            "code": '''import torch
torch.manual_seed(2)
c, u = torch.randn(2,3,5), torch.randn(2,3,5)
out = {fn}(c, u, 4.0, .7, 1e-8)
p = u + 4.0*(c-u); dims=(1,2)
r = p * (c.std(dims, keepdim=True, unbiased=False)/(p.std(dims, keepdim=True, unbiased=False)+1e-8))
ref = .7*r + .3*p
assert torch.allclose(out, ref, atol=1e-7, rtol=1e-6)
''',
        },
        {
            "name": "zero scale and constant edge",
            "code": '''import torch
c = torch.ones(3,4); u = torch.arange(12.).reshape(3,4)
plain = {fn}(c, u, guidance_scale=0.0)
rescaled = {fn}(c, u, guidance_scale=2.0, rescale_phi=1.0)
assert torch.equal(plain, u)
assert rescaled.shape == c.shape and torch.isfinite(rescaled).all() and torch.equal(rescaled, torch.zeros_like(c))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
c = torch.randn(2,3,4, requires_grad=True)
u = torch.randn(2,3,4, requires_grad=True)
{fn}(c, u, 3.0, .5).square().sum().backward()
assert c.grad is not None and u.grad is not None
assert torch.isfinite(c.grad).all() and torch.isfinite(u.grad).all()
''',
        },
    ],
    "solution": '''import torch

def cfg_guided_prediction(cond_pred, uncond_pred, guidance_scale=7.5, rescale_phi=0.0, eps=1e-8):
    pred_cfg = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    if rescale_phi > 0.0:
        dims = tuple(range(1, cond_pred.ndim))
        std_cond = cond_pred.std(dim=dims, keepdim=True, unbiased=False)
        std_cfg = pred_cfg.std(dim=dims, keepdim=True, unbiased=False)
        pred_rescaled = pred_cfg * (std_cond / (std_cfg + eps))
        pred_cfg = rescale_phi * pred_rescaled + (1.0 - rescale_phi) * pred_cfg
    return pred_cfg
''',
    "demo": '''cond = torch.randn(2, 4, 8, 8)
uncond = torch.randn_like(cond)
print(cfg_guided_prediction(cond, uncond, 7.5, .7).shape)
''',
    "explanation": "Population standard deviations are reduced only over non-batch dimensions, so every sample receives an independent scale correction.",
}
