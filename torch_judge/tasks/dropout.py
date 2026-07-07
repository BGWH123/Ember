"""Implement Dropout task."""

TASK = {
    "title": "Implement Dropout",
    "title_zh": "实现 Dropout",
    "difficulty": "Easy",
    "category": "正则化",
    "description_en": (
        "Implement dropout as an nn.Module.\n\nDropout randomly zeroes elements during training and scales survivors by `1/(1-p)` to maintain expected values. During eval, it is an identity.\n\n**Signature:** `MyDropout(p=0.5)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor of any shape\n\n**Returns:** tensor with dropout applied (training) or unchanged (eval)\n\n**Constraints:**\n- Training: zero with probability p, scale by `1/(1-p)`\n- Eval: return input unchanged"
    ),
    "description_zh": (
        "实现 Dropout（nn.Module）。\n\nDropout 在训练时以概率 p 随机将元素置零，并将存活元素缩放 `1/(1-p)` 以保持期望值不变。推理时为恒等映射。\n\n**签名:** `MyDropout(p=0.5)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 任意形状的输入张量\n\n**返回:** 应用 dropout 后的张量（训练）或原始输入（推理）\n\n**约束:**\n- 训练模式：以概率 p 置零，缩放 `1/(1-p)`\n- 推理模式：返回原始输入"
    ),
    "function_name": "MyDropout",
    "hint": (
        "Train: `mask = (rand_like(x) > p).float()` → `x * mask / (1-p)`\nEval: return `x` unchanged"
    ),
    "hint_zh": (
        "训练：`mask = (rand_like(x) > p).float()` → `x * mask / (1-p)`\n推理：直接返回 `x`"
    ),
    "theory_en": (
        "Dropout randomly sets a fraction of input elements to zero during training, preventing co-adaptation of neurons.\n\n**Training:**\n$$y = m \\odot x \\cdot \\frac{1}{1-p}$$\nwhere $m \\sim \\text{Bernoulli}(1-p)$\n\n**Inference:**\n$$y = x$$ (identity)\n\n**Inverted Dropout:**\nThe scaling $1/(1-p)$ is applied during training, so no adjustment is needed at test time. This is the modern standard.\n\n**Why it works:**\n- Each training iteration uses a different 'thinned' network\n- At test time, averaging predictions over all $2^N$ possible networks approximates the geometric mean"
    ),
    "theory_zh": (
        "Dropout 在训练时随机将一部分输入元素置零，防止神经元共适应。\n\n**训练：**\n$$y = m \\odot x \\cdot \\frac{1}{1-p}$$\n其中 $m \\sim \\text{Bernoulli}(1-p)$\n\n**推理：**\n$$y = x$$（恒等映射）\n\n**Inverted Dropout：**\n缩放 $1/(1-p)$ 在训练时应用，因此测试时无需调整。这是现代标准做法。\n\n**工作原理：**\n- 每次训练迭代使用不同的'稀疏'网络\n- 测试时，对所有 $2^N$ 可能网络的预测取平均近似几何平均"
    ),
    "tests": [
        {
            "name": "Eval mode is identity",
            "code": """









import torch, torch.nn as nn
d = {fn}(p=0.5)
assert isinstance(d, nn.Module), 'Must inherit from nn.Module'
d.eval()
x = torch.randn(4, 8)
assert torch.equal(d(x), x), 'eval mode should return input unchanged'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Training: zeros and scaling",
            "code": """









import torch
torch.manual_seed(42)
d = {fn}(p=0.5)
d.train()
x = torch.ones(1000)
out = d(x)
assert (out == 0).any(), 'No zeros found during training'
non_zero = out[out != 0]
assert torch.allclose(non_zero, torch.full_like(non_zero, 2.0), atol=1e-5), 'Non-zeros should be scaled by 1/(1-p)=2.0'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Drop rate is approximately p",
            "code": """









import torch
torch.manual_seed(0)
d = {fn}(p=0.3)
d.train()
out = d(torch.ones(10000))
frac = (out == 0).float().mean().item()
assert 0.25 < frac < 0.35, f'Expected ~30%% zeros, got {frac*100:.1f}%%'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
d = {fn}(p=0.5)
d.train()
x = torch.randn(4, 8, requires_grad=True)
d(x).sum().backward()
assert x.grad is not None, 'x.grad is None'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


def MyDropout(x, p=0.5, training=True):
    """
    Dropout implementation (Inverted Dropout).
    训练时随机将 p 比例的元素置零，并将保留的元素缩放 1/(1-p)。
    推理时直接返回输入（无需调整，因为训练时已缩放）。
    """
    if not training:
        return x                                         # 推理模式: 恒等映射

    # 生成与 x 同形状的随机掩码，元素以概率 (1-p) 为 1，概率 p 为 0
    # torch.rand_like(x) 生成 [0,1) 均匀分布，> p 的位置保留
    mask = (torch.rand_like(x) > p).float()              # Bernoulli(1-p) 掩码

    # 应用掩码并缩放: 保留的元素乘以 1/(1-p)
    # 缩放保证期望值不变: E[mask] = 1-p, 所以 E[x * mask / (1-p)] = x
    return x * mask / (1 - p)                            # (B, S, D)

    
    
    ''',
    "demo": '''








d = MyDropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))
    
    
    
    
    
    
    
    
    ''',
}
