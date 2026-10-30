"""Classifier-Free Guidance (CFG) & Std Rescaling task."""

TASK = {
    'title': 'Classifier-Free Guidance (CFG) & Std Rescaling',
    'title_zh': '无分类器引导（CFG）与标准差重缩放',
    'difficulty': 'Easy',
    'category': '扩散与流模型',
    'function_name': 'cfg_guided_prediction',
    'starter_code': """def cfg_guided_prediction(cond_pred, uncond_pred, guidance_scale=7.5, rescale_phi=0.0, eps=1e-8):
    pass""",
    'hint': """Extrapolate from the unconditional prediction toward the conditional one. For rescaling, compute population standard deviations over every non-batch axis and blend the rescaled result.""",
    'hint_zh': '从无条件预测向条件预测外推。重缩放时，在所有非批次维上计算总体标准差，并混合重缩放结果。',
    'description_en': """Implement classifier-free guidance for predictions shaped (B,...), with optional per-sample standard-deviation rescaling. Return a tensor with the input shape.""",
    'description_zh': '对形状为 (B,...) 的预测实现无分类器引导，并可选地按样本进行标准差重缩放。返回与输入形状相同的张量。',
    'theory_en': """CFG combines conditional and unconditional model evaluations as u+s(c-u). Large scales can over-amplify predictions; matching each sample's standard deviation to the conditional prediction mitigates this effect.""",
    'theory_zh': 'CFG 按 u+s(c-u) 组合条件与无条件模型输出。较大的引导尺度可能放大输出；将每个样本的标准差匹配到条件预测可缓解该问题。',
    'diagram_en': r"""```text
cond c --\
          u + scale*(c-u) -> optional std match -> blend -> output
uncond u-/
```""",
    'diagram_zh': r"""```text
条件 c --\
           u + scale*(c-u) -> 可选标准差匹配 -> 混合 -> 输出
无条件 u-/
```""",
    "tests": [
        {
            "name": 'shape and standard CFG',
            "code": """import torch
c, u = torch.randn(2,3,4), torch.randn(2,3,4)
out = {fn}(c, u, guidance_scale=2.5)
assert out.shape == c.shape
assert torch.equal(out, u + 2.5 * (c-u))""",
        },
        {
            "name": 'exact rescaling reference',
            "code": """import torch
torch.manual_seed(2)
c, u = torch.randn(2,3,5), torch.randn(2,3,5)
out = {fn}(c, u, 4.0, .7, 1e-8)
p = u + 4.0*(c-u); dims=(1,2)
r = p * (c.std(dims, keepdim=True, unbiased=False)/(p.std(dims, keepdim=True, unbiased=False)+1e-8))
ref = .7*r + .3*p
assert torch.allclose(out, ref, atol=1e-7, rtol=1e-6)""",
        },
        {
            "name": 'zero scale and constant edge',
            "code": """import torch
c = torch.ones(3,4); u = torch.arange(12.).reshape(3,4)
plain = {fn}(c, u, guidance_scale=0.0)
rescaled = {fn}(c, u, guidance_scale=2.0, rescale_phi=1.0)
assert torch.equal(plain, u)
assert rescaled.shape == c.shape and torch.isfinite(rescaled).all() and torch.equal(rescaled, torch.zeros_like(c))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
c = torch.randn(2,3,4, requires_grad=True)
u = torch.randn(2,3,4, requires_grad=True)
{fn}(c, u, 3.0, .5).square().sum().backward()
assert c.grad is not None and u.grad is not None
assert torch.isfinite(c.grad).all() and torch.isfinite(u.grad).all()""",
        },
    ],
    'solution': '''def cfg_guided_prediction(cond_pred, uncond_pred, guidance_scale=7.5, rescale_phi=0.0, eps=1e-8):
    """无分类器引导预测 (Classifier-Free Guidance)。"""
    # Step 1: 从无条件预测向条件预测方向外推
    guided_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    # Step 2: 可选地按样本匹配条件预测的总体标准差
    if rescale_phi > 0.0:
        reduce_dims = tuple(range(1, cond_pred.ndim))
        cond_std = cond_pred.std(
            dim=reduce_dims, keepdim=True, unbiased=False
        )
        guided_std = guided_pred.std(
            dim=reduce_dims, keepdim=True, unbiased=False
        )
        rescaled_pred = guided_pred * (cond_std / (guided_std + eps))

        # Step 3: 混合重缩放与原始引导结果
        guided_pred = (
            rescale_phi * rescaled_pred
            + (1.0 - rescale_phi) * guided_pred
        )
    return guided_pred''',
    'demo': """torch.manual_seed(42)
B, C, H, W = 2, 4, 8, 8
cond_pred = torch.randn(B, C, H, W)
uncond_pred = torch.randn(B, C, H, W)
out = cfg_guided_prediction(cond_pred, uncond_pred, 7.5, 0.7)
print("Output shape:", out.shape)
print("Output std:", out.std().item())
print("Sample values:", out[0, 0, 0, :3])""",
    'explanation': """### 核心思路
CFG 按 $u+s(c-u)$ 放大条件方向；可选重缩放将每个样本的标准差匹配到条件预测，再以 `rescale_phi` 混合。

### 代码步骤拆解
- **Step 1（引导外推）**：`guided_pred` 与输入保持相同形状。
- **Step 2（样本内统计）**：除批次维外全部归约，`keepdim=True` 得到可广播标准差。
- **Step 3（结果混合）**：在重缩放结果与原始引导结果之间线性插值。

### 重点与易错点分析
- **注意点 1（方差定义）**：使用 `unbiased=False` 的总体标准差，避免小样本修正改变参考结果。
- **注意点 2（数值稳定）**：`eps` 加在 `guided_std` 分母，常量预测不会产生除零。
- **注意点 3（逐样本缩放）**：归约维度不能包含批次轴，否则样本之间会互相影响。

### 复杂度分析
- **时间复杂度**：$O(N)$，外推和标准差统计均逐元素处理。
- **空间复杂度**：$O(N)$，主要保存引导与重缩放预测。""",
}
