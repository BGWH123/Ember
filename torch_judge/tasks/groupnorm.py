"""Group Normalization task."""

TASK = {
    'title': 'Group Normalization',
    'title_zh': '组归一化（GroupNorm）',
    'difficulty': 'Medium',
    'category': '归一化',
    'function_name': 'my_group_norm',
    'starter_code': """def my_group_norm(x, num_groups, gamma, beta, eps=1e-5):
    pass""",
    'hint': """Reshape to `(B,G,C/G,H,W)`, compute population mean/variance over the last three dimensions, restore BCHW, and apply channel-wise affine parameters.""",
    'hint_zh': '重塑为 `(B,G,C/G,H,W)`，在后三维计算总体均值/方差，恢复 BCHW 后应用逐通道仿射参数。',
    'description_en': 'Implement Group Normalization manually for BCHW inputs with channel-wise scale and bias.',
    'description_zh': '为 BCHW 输入手动实现组归一化，并应用逐通道缩放和偏置。',
    'theory_en': """GroupNorm partitions channels into groups and normalizes each group independently over channels and spatial positions. Unlike BatchNorm, it does not depend on other batch samples.""",
    'theory_zh': 'GroupNorm 将通道划分为若干组，在每组的通道和空间位置上独立归一化；与 BatchNorm 不同，它不依赖其他批次样本。',
    'diagram_en': """```text
B,C,H,W -> B,G,C/G,H,W -> group mean/var -> B,C,H,W -> gamma,beta
```""",
    'diagram_zh': """```text
B,C,H,W -> B,G,C/G,H,W -> 组均值/方差 -> B,C,H,W -> gamma,beta
```""",
    "tests": [
        {
            "name": 'output shape',
            "code": """import torch
x = torch.randn(2, 8, 3, 5)
out = {fn}(x, 4, torch.ones(8), torch.zeros(8))
assert out.shape == x.shape""",
        },
        {
            "name": 'matches PyTorch reference',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(3, 6, 4, 2)
gamma, beta = torch.randn(6), torch.randn(6)
out = {fn}(x, 3, gamma, beta, 1e-5)
ref = torch.nn.functional.group_norm(x, 3, gamma, beta, 1e-5)
assert torch.allclose(out, ref, atol=1e-6)""",
        },
        {
            "name": 'one channel per group edge',
            "code": """import torch
x = torch.randn(1, 4, 2, 3)
out = {fn}(x, 4, torch.ones(4), torch.zeros(4))
means = out.mean(dim=(2, 3))
vars = out.var(dim=(2, 3), unbiased=False)
assert torch.allclose(means, torch.zeros_like(means), atol=1e-5) and torch.allclose(vars, torch.ones_like(vars), atol=2e-4)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(2, 4, 3, 3, requires_grad=True)
gamma = torch.ones(4, requires_grad=True)
beta = torch.zeros(4, requires_grad=True)
{fn}(x, 2, gamma, beta).square().sum().backward()
assert x.grad is not None and gamma.grad is not None and beta.grad is not None""",
        },
    ],
    'solution': '''def my_group_norm(x, num_groups, gamma, beta, eps=1e-5):
    """手动实现组归一化 (Group Normalization)。"""
    B, C, H, W = x.shape

    # Step 1: 将通道划分为若干组
    grouped = x.reshape(B, num_groups, C // num_groups, H, W)

    # Step 2: 在组内通道和空间维计算总体统计量
    mean = grouped.mean(dim=(2, 3, 4), keepdim=True)
    variance = grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    normalized = (grouped - mean) / torch.sqrt(variance + eps)

    # Step 3: 恢复 BCHW 并应用逐通道仿射变换
    normalized = normalized.reshape(B, C, H, W)
    return gamma.view(1, C, 1, 1) * normalized + beta.view(1, C, 1, 1)''',
    'demo': """torch.manual_seed(42)
B, C, H, W = 2, 8, 4, 4
x = torch.randn(B, C, H, W)
out = my_group_norm(x, 4, torch.ones(C), torch.zeros(C))
print("Output shape:", out.shape)
print("Output mean:", out.mean().item())
print("Output variance:", out.var(unbiased=False).item())""",
    'explanation': r"""### 核心思路
将输入重塑为 `(B, G, C/G, H, W)`，在每个样本每组内部归一化，再用逐通道 `gamma` 和 `beta` 做仿射变换。

### 代码步骤拆解
- **Step 1（通道分组）**：`grouped` 显式增加组维，不改变元素总数。
- **Step 2（组内归一化）**：均值和方差在后三维计算并保持维度以便广播。
- **Step 3（仿射恢复）**：恢复 BCHW 后将 `(C,)` 参数重塑为 `(1, C, 1, 1)`。

### 重点与易错点分析
- **注意点 1（总体方差）**：必须设置 `unbiased=False` 才与 PyTorch GroupNorm 一致。
- **注意点 2（eps 位置）**：`eps` 加在方差内部再开方，即 $\sqrt{\sigma^2+\epsilon}$。
- **注意点 3（整除约束）**：`C` 必须能被 `num_groups` 整除。

### 复杂度分析
- **时间复杂度**：$O(BCHW)$，每个元素参与统计与归一化。
- **空间复杂度**：$O(BCHW)$，主要保存归一化结果。""",
}
