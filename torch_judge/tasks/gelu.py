"""GELU Activation task."""

TASK = {
    "title": "GELU Activation",
    "title_zh": "GELU 激活函数",
    "difficulty": "Easy",
    "category": "激活函数",
    "description_en": (
        "Implement the GELU activation function.\n\nGELU (Gaussian Error Linear Unit) smoothly gates inputs based on their value, used in transformers like BERT and GPT.\n\n**Signature:** `my_gelu(x) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor of any shape\n\n**Returns:** element-wise GELU activation, same shape as input\n\n**Constraints:**\n- Exact formula: `x * 0.5 * (1 + erf(x / sqrt(2)))`\n- Must match `F.gelu` within 1e-4\n- `gelu(0) = 0`"
    ),
    "description_zh": (
        "实现 GELU 激活函数。\n\nGELU（高斯误差线性单元）根据输入值平滑地进行门控，广泛用于 BERT 和 GPT 等 Transformer。\n\n**签名:** `my_gelu(x) -> Tensor`\n\n**参数:**\n- `x` — 任意形状的输入张量\n\n**返回:** 逐元素 GELU 激活，形状与输入相同\n\n**约束:**\n- 精确公式：`x * 0.5 * (1 + erf(x / sqrt(2)))`\n- 必须与 `F.gelu` 误差在 1e-4 以内\n- `gelu(0) = 0`"
    ),
    "function_name": "my_gelu",
    "hint": (
        "Exact: `0.5 * x * (1 + torch.erf(x / sqrt(2)))`\nApprox: `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`"
    ),
    "hint_zh": (
        "精确版：`0.5 * x * (1 + torch.erf(x / sqrt(2)))`\n近似版：`0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`"
    ),
    "theory_en": (
        "GELU (Gaussian Error Linear Unit) smoothly gates inputs by their probability under a Gaussian distribution.\n\n**Exact Formula:**\n$$\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2} \\left[1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right]$$\n\n**Approximation (faster):**\n$$\\text{GELU}(x) \\approx 0.5x \\left(1 + \\tanh\\left[\\sqrt{\\frac{2}{\\pi}} \\left(x + 0.044715x^3\\right)\\right]\\right)$$\n\n**Properties:**\n- Smooth everywhere (unlike ReLU's kink at 0)\n- Biased toward 0 for negative inputs (stochastic regularization effect)\n- Used in BERT, GPT, ViT, and most modern Transformers"
    ),
    "theory_zh": (
        "GELU（高斯误差线性单元）根据输入在高斯分布下的概率平滑地进行门控。\n\n**精确公式：**\n$$\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2} \\left[1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right]$$\n\n**近似公式（更快）：**\n$$\\text{GELU}(x) \\approx 0.5x \\left(1 + \\tanh\\left[\\sqrt{\\frac{2}{\\pi}} \\left(x + 0.044715x^3\\right)\\right]\\right)$$\n\n**性质：**\n- 处处光滑（不同于 ReLU 在 0 处的折点）\n- 负输入偏向 0（随机正则化效果）\n- 用于 BERT、GPT、ViT 和大多数现代 Transformer"
    ),
    "tests": [
        {
            "name": "Matches F.gelu",
            "code": """









import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
out = {fn}(x)
ref = torch.nn.functional.gelu(x)
assert torch.allclose(out, ref, atol=1e-4), 'Does not match F.gelu'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "gelu(0) = 0",
            "code": """









import torch
out = {fn}(torch.tensor([0.0]))
assert torch.allclose(out, torch.tensor([0.0]), atol=1e-7), f'gelu(0) = {out.item()}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Shape preservation",
            "code": """









import torch
x = torch.randn(2, 3, 4)
assert {fn}(x).shape == x.shape, 'Shape mismatch'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
x = torch.randn(4, 8, requires_grad=True)
{fn}(x).sum().backward()
assert x.grad is not None and x.grad.shape == x.shape, 'Gradient issue'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


def my_gelu(x):
    """
    GELU (Gaussian Error Linear Unit) implementation.
    使用 tanh 近似公式，比精确实现（erf）更快，被 PyTorch 默认采用。
    公式: GELU(x) ≈ 0.5x * (1 + tanh[√(2/π) * (x + 0.044715x³)])
    """
    # 系数: √(2/π) ≈ 0.7978845608
    sqrt_2_over_pi = (2.0 / 3.141592653589793) ** 0.5

    # 内部多项式: x + 0.044715 * x³
    # 三次项提供非线性，使函数在负值区域平滑趋近于0
    inner = x + 0.044715 * (x ** 3)

    # tanh(√(2/π) * inner)
    tanh_val = torch.tanh(sqrt_2_over_pi * inner)

    # GELU(x) = 0.5x * (1 + tanh(...))
    # 当 x → +∞ 时 tanh → 1，GELU(x) → x；当 x → -∞ 时 tanh → -1，GELU(x) → 0
    return 0.5 * x * (1.0 + tanh_val)

    
    
    ''',
    "demo": '''








x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', my_gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))
    
    
    
    
    
    
    
    
    ''',
}
