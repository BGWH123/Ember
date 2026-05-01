"""Implement Softmax task."""

TASK = {
    "title": "Implement Softmax",
    "title_zh": "实现 Softmax",
    "difficulty": "Easy",
    "category": "激活函数",
    "description_en": (
        "Implement the softmax function.\n\nSoftmax converts raw logits into a probability distribution by exponentiating and normalizing, used in classification and attention.\n\n**Signature:** `my_softmax(x, dim=-1) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor of any shape\n- `dim` — dimension along which to compute softmax\n\n**Returns:** probability tensor (sums to 1 along dim), same shape as input\n\n**Constraints:**\n- Subtract max for numerical stability before exp\n- Must handle large values without NaN/Inf"
    ),
    "description_zh": (
        "实现 softmax 函数。\n\nSoftmax 通过指数化和归一化将原始 logits 转换为概率分布，用于分类和注意力机制。\n\n**签名:** `my_softmax(x, dim=-1) -> Tensor`\n\n**参数:**\n- `x` — 任意形状的输入张量\n- `dim` — 计算 softmax 的维度\n\n**返回:** 概率张量（沿 dim 求和为 1），形状与输入相同\n\n**约束:**\n- 在 exp 之前减去最大值以保证数值稳定\n- 必须处理大值而不产生 NaN/Inf"
    ),
    "function_name": "my_softmax",
    "hint": (
        "1. `x_max = x.max(dim=dim, keepdim=True).values`\n2. `e_x = exp(x - x_max)`\n3. `return e_x / e_x.sum(dim=dim, keepdim=True)`"
    ),
    "hint_zh": (
        "1. `x_max = x.max(dim=dim, keepdim=True).values`\n2. `e_x = exp(x - x_max)`\n3. `return e_x / e_x.sum(dim=dim, keepdim=True)`"
    ),
    "theory_en": (
        "Softmax converts a vector of real values into a probability distribution.\n\n**Formula:**\n$$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j=1}^C e^{x_j}}$$\n\n**Numerical Stability:**\nDirect computation can overflow for large $x_i$. The stable version subtracts $\\max(x)$ before exponentiation:\n$$\\text{softmax}(x_i) = \\frac{e^{x_i - \\max(x)}}{\\sum_{j=1}^C e^{x_j - \\max(x)}}$$\n\n**Temperature Scaling:**\n$$\\text{softmax}_\\tau(x_i) = \\frac{e^{x_i / \\tau}}{\\sum_j e^{x_j / \\tau}}$$\n- $\\tau \\to 0$: sharp (argmax-like)\n- $\\tau \\to \\infty$: uniform distribution"
    ),
    "theory_zh": (
        "Softmax 将实数值向量转换为概率分布。\n\n**公式：**\n$$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j=1}^C e^{x_j}}$$\n\n**数值稳定性：**\n大值 $x_i$ 的直接计算可能溢出。稳定版本在指数化前减去 $\\max(x)$：\n$$\\text{softmax}(x_i) = \\frac{e^{x_i - \\max(x)}}{\\sum_{j=1}^C e^{x_j - \\max(x)}}$$\n\n**温度缩放：**\n$$\\text{softmax}_\\tau(x_i) = \\frac{e^{x_i / \\tau}}{\\sum_j e^{x_j / \\tau}}$$\n- $\\tau \\to 0$：尖锐（接近 argmax）\n- $\\tau \\to \\infty$：均匀分布"
    ),
    "diagram_en": (
        "```mermaid\nflowchart TD\n    X[x] --> MAX[x_max = max(x)]\n    MAX --> SUB[Subtract: x - x_max]\n    SUB --> EXP[Exp: e^(x - x_max)]\n    EXP --> SUM[Sum: sum(e^(x - x_max))]\n    EXP --> DIV[Divide: exp / sum]\n    SUM --> DIV\n    DIV --> OUT[Softmax output]\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart TD\n    X[x] --> MAX[x_max = max(x)]\n    MAX --> SUB[减去: x - x_max]\n    SUB --> EXP[指数: e^(x - x_max)]\n    EXP --> SUM[求和: sum(e^(x - x_max))]\n    EXP --> DIV[相除: exp / sum]\n    SUM --> DIV\n    DIV --> OUT[Softmax 输出]\n```"
    ),
    "tests": [
        {
            "name": "Basic 1-D",
            "code": """









import torch
x = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "2-D along dim=-1",
            "code": """









import torch
x = torch.randn(4, 8)
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert out.shape == expected.shape, f'Shape mismatch'
assert torch.allclose(out, expected, atol=1e-5), 'Values differ'
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5), 'Rows must sum to 1'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Numerical stability",
            "code": """









import torch
x = torch.tensor([1000., 1001., 1002.])
out = {fn}(x, dim=-1)
assert not torch.isnan(out).any(), 'NaN in output — not numerically stable'
assert not torch.isinf(out).any(), 'Inf in output — not numerically stable'
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), 'Values differ on large input'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "dim=0 softmax",
            "code": """









import torch
torch.manual_seed(3)
x = torch.randn(4, 3)
out = {fn}(x, dim=0)
exp_x = (x - x.max(dim=0, keepdim=True).values).exp()
expected = exp_x / exp_x.sum(dim=0, keepdim=True)
assert torch.allclose(out, expected, atol=1e-5), f'dim=0 softmax failed'
assert torch.allclose(out.sum(dim=0), torch.ones(3), atol=1e-5), 'columns should sum to 1'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


def my_softmax(x):
    """
    Numerically stable Softmax implementation.
    Softmax 将任意实数向量映射为概率分布，所有元素非负且和为1。
    """
    # 数值稳定性: 先减去最大值，防止指数溢出
    # x_max 沿最后一个维度（类别维度）取最大值，keepdim=True 保证广播兼容
    x_max = x.max(dim=-1, keepdim=True).values           # max(x) 沿类别维

    # 指数化: exp(x - x_max)，所有值 ≤ 1，避免溢出
    exp_x = torch.exp(x - x_max)                         # e^(x_i - max(x))

    # 归一化: 每个元素除以该维度上的指数和
    # dim=-1 保证每个样本/位置独立归一化
    return exp_x / exp_x.sum(dim=-1, keepdim=True)       # (B, S, C)，每行和为1

    
    
    ''',
    "demo": '''








x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", my_softmax(x, dim=-1))
print("Sum:   ", my_softmax(x, dim=-1).sum())
print("Ref:   ", torch.softmax(x, dim=-1))
    
    
    
    
    
    
    
    
    ''',
}
