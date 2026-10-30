"""Dynamic Tanh (DyT) task."""

TASK = {
    'title': 'Dynamic Tanh (DyT)',
    'title_zh': '动态 Tanh 归一化替代层（DyT）',
    'difficulty': 'Easy',
    'category': '归一化',
    'function_name': 'dynamic_tanh',
    'starter_code': """def dynamic_tanh(x, alpha, gamma, beta):
    pass""",
    'hint': 'Scale x by the scalar alpha, apply tanh, then use the broadcastable feature-wise gamma and beta.',
    'hint_zh': '先用标量 alpha 缩放 x，再应用 tanh，最后使用可广播的逐特征 gamma 和 beta。',
    'description_en': """Implement Dynamic Tanh, a simple normalization-layer alternative with a learned input scale and feature-wise affine transform.""",
    'description_zh': '实现 Dynamic Tanh：一种使用可学习输入缩放和逐特征仿射变换的归一化层替代方案。',
    'theory_en': """DyT bounds activations with tanh instead of computing sample statistics. Alpha controls saturation globally, while gamma and beta restore per-feature scale and offset.""",
    'theory_zh': 'DyT 用 tanh 限制激活而不计算样本统计量。alpha 全局控制饱和程度，gamma 和 beta 恢复逐特征尺度与偏移。',
    'diagram_en': """```text
x -> alpha*x -> tanh -> gamma*(.)+beta
```""",
    'diagram_zh': """```text
x -> alpha*x -> tanh -> gamma*(.)+beta
```""",
    "tests": [
        {
            "name": 'output shape',
            "code": """import torch
x = torch.randn(2, 3, 7)
out = {fn}(x, torch.tensor([0.5]), torch.ones(7), torch.zeros(7))
assert out.shape == x.shape""",
        },
        {
            "name": 'exact formula',
            "code": """import torch
torch.manual_seed(1)
x = torch.randn(4, 5)
alpha, gamma, beta = torch.tensor(0.7), torch.randn(5), torch.randn(5)
out = {fn}(x, alpha, gamma, beta)
ref = gamma * torch.tanh(alpha * x) + beta
assert torch.equal(out, ref)""",
        },
        {
            "name": 'zero alpha edge',
            "code": """import torch
x = torch.tensor([[1e6, -1e6, 3.0]])
beta = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, torch.tensor(0.0), torch.randn(3), beta)
assert torch.equal(out, beta.unsqueeze(0))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(2, 4, requires_grad=True)
alpha = torch.tensor([0.5], requires_grad=True)
gamma = torch.ones(4, requires_grad=True)
beta = torch.zeros(4, requires_grad=True)
{fn}(x, alpha, gamma, beta).square().sum().backward()
assert all(t.grad is not None for t in (x, alpha, gamma, beta))""",
        },
    ],
    'solution': '''def dynamic_tanh(x, alpha, gamma, beta):
    """计算动态双曲正切变换 (Dynamic Tanh)。"""
    # Step 1: 用全局可学习尺度控制输入饱和程度
    bounded = torch.tanh(alpha * x)

    # Step 2: 沿最后一维应用逐特征仿射变换
    out = gamma * bounded + beta
    return out''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 10, 64)
out = dynamic_tanh(x, torch.tensor(0.5), torch.ones(64), torch.zeros(64))
print("Output shape:", out.shape)
print("Output range:", out.min().item(), out.max().item())""",
    'explanation': r"""### 核心思路
DyT 不计算样本统计量，而是通过 $y=\gamma\odot\tanh(\alpha x)+\beta$ 限制激活；`alpha` 控制全局饱和度，`gamma`、`beta` 恢复逐特征尺度与偏移。

### 代码步骤拆解
- **Step 1（非线性限幅）**：标量或可广播的 `alpha` 先缩放 `x`，`tanh` 将结果限制在 $[-1,1]$。
- **Step 2（特征仿射）**：形状 `(D,)` 的 `gamma` 与 `beta` 沿输入最后一维广播。
- **Step 3（输出保持）**：所有运算逐元素执行，因此 `out` 与 `x` 形状完全一致。

### 重点与易错点分析
- **注意点 1（广播对齐）**：`gamma`、`beta` 的长度应匹配最后一个特征维，避免意外沿批次维广播。
- **注意点 2（梯度饱和）**：`alpha` 过大时 `tanh` 进入饱和区，导数趋近零；初始化尺度需谨慎。
- **注意点 3（零尺度边界）**：`alpha=0` 时非线性输出严格为零，最终结果应等于广播后的 `beta`。

### 复杂度分析
- **时间复杂度**：$O(N)$，主要开销是逐元素 `tanh`。
- **空间复杂度**：$O(N)$，需保存非线性激活以支持反向传播。""",
}
