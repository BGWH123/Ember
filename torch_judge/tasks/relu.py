"""Implement ReLU task."""

TASK = {
    "title": "Implement ReLU",
    "title_zh": "实现 ReLU",
    "difficulty": "Easy",
    "category": "激活函数",
    "description_en": (
        "Implement the ReLU activation function.\n\nReLU (Rectified Linear Unit) outputs the input directly if positive, otherwise zero. It is the most widely used activation in deep learning.\n\n**Signature:** `relu(x) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor of any shape\n\n**Returns:** element-wise ReLU activation, same shape as input\n\n**Constraints:**\n- `relu(x) = max(0, x)` element-wise\n- Must support gradient flow\n- Must be efficient on large tensors"
    ),
    "description_zh": (
        "实现 ReLU 激活函数。\n\nReLU（修正线性单元）在输入为正时直接输出，否则输出零，是深度学习中最广泛使用的激活函数。\n\n**签名:** `relu(x) -> Tensor`\n\n**参数:**\n- `x` — 任意形状的输入张量\n\n**返回:** 逐元素 ReLU 激活，形状与输入相同\n\n**约束:**\n- `relu(x) = max(0, x)` 逐元素\n- 必须支持梯度流\n- 在大张量上必须高效"
    ),
    "function_name": "relu",
    "hint": "`relu(x) = max(0, x)` element-wise → `x * (x > 0)`",
    "hint_zh": "`relu(x) = max(0, x)` 逐元素 → `x * (x > 0)`",
    "theory_en": (
        "ReLU (Rectified Linear Unit) is the most widely used activation function.\n\n**Formula:**\n$$\\text{ReLU}(x) = \\max(0, x)$$\n\n**Properties:**\n- Non-linear but piecewise linear\n- Computationally cheap (no exponentials or divisions)\n- Sparse activation: ~50% of neurons are inactive on average\n\n**The Dying ReLU Problem:**\nIf a neuron's weights are updated such that it always outputs 0, gradients are zero and it never recovers. Solutions: LeakyReLU, PReLU, ELU."
    ),
    "theory_zh": (
        "ReLU（修正线性单元）是最广泛使用的激活函数。\n\n**公式：**\n$$\\text{ReLU}(x) = \\max(0, x)$$\n\n**性质：**\n- 非线性但分段线性\n- 计算廉价（无指数或除法）\n- 稀疏激活：平均约 50% 神经元不活跃\n\n**死亡 ReLU 问题：**\n如果神经元的权重更新导致其始终输出 0，则梯度为 0 且永远无法恢复。解决方案：LeakyReLU、PReLU、ELU。"
    ),
    "tests": [
        {
            "name": "Basic values",
            "code": """









import torch
x = torch.tensor([-2., -1., 0., 1., 2.])
out = {fn}(x)
expected = torch.tensor([0., 0., 0., 1., 2.])
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert torch.allclose(out, expected), f'Wrong Answer: {out} vs {expected}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "2-D tensor",
            "code": """









import torch
x = torch.randn(4, 8)
out = {fn}(x)
assert out.shape == x.shape, f'Shape mismatch on 2-D input'
assert (out >= 0).all(), 'ReLU output must be non-negative'
assert torch.allclose(out, x.clamp(min=0)), 'Value mismatch on random input'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient check",
            "code": """









import torch
x = torch.tensor([-1., 0., 1., 2.], requires_grad=True)
out = {fn}(x)
out.sum().backward()
assert x.grad is not None, 'Gradient not computed'
assert x.grad[0] == 0., f'grad at x=-1 should be 0, got {x.grad[0]}'
assert x.grad[2] == 1., f'grad at x=1 should be 1, got {x.grad[2]}'
assert x.grad[3] == 1., f'grad at x=2 should be 1, got {x.grad[3]}'
assert x.grad[1] in (0., 1.), f'grad at x=0 should be 0 or 1, got {x.grad[1]}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Performance",
            "code": """









import torch, time
big = torch.randn(1024, 1024)
t0 = time.perf_counter()
for _ in range(100):
    {fn}(big)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s for 100 iterations'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








def relu(x: torch.Tensor) -> torch.Tensor:  # ReLU activation
    return x * (x > 0).float()
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








x = torch.tensor([-2., -1., 0., 1., 2.])
print("Input: ", x)
print("Output:", relu(x))
    
    
    
    
    
    
    
    
    ''',
}
