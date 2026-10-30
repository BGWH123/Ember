"""Implement Dropout task."""

TASK = {
    'title': 'Implement Dropout',
    'title_zh': '实现 Dropout',
    'difficulty': 'Easy',
    'category': '正则化',
    'function_name': 'MyDropout',
    'starter_code': """class MyDropout(nn.Module):
    pass""",
    'hint': """Train: `mask = (rand_like(x) > p).float()` → `x * mask / (1-p)`
Eval: return `x` unchanged""",
    'hint_zh': """训练：`mask = (rand_like(x) > p).float()` → `x * mask / (1-p)`
推理：直接返回 `x`""",
    'description_en': """Implement dropout as an nn.Module.

Dropout randomly zeroes elements during training and scales survivors by `1/(1-p)` to maintain expected values. During eval, it is an identity.

**Signature:** `MyDropout(p=0.5)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor of any shape

**Returns:** tensor with dropout applied (training) or unchanged (eval)

**Constraints:**
- Training: zero with probability p, scale by `1/(1-p)`
- Eval: return input unchanged""",
    'description_zh': """实现 Dropout（nn.Module）。

Dropout 在训练时以概率 p 随机将元素置零，并将存活元素缩放 `1/(1-p)` 以保持期望值不变。推理时为恒等映射。

**签名:** `MyDropout(p=0.5)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 任意形状的输入张量

**返回:** 应用 dropout 后的张量（训练）或原始输入（推理）

**约束:**
- 训练模式：以概率 p 置零，缩放 `1/(1-p)`
- 推理模式：返回原始输入""",
    'theory_en': r"""Dropout randomly sets a fraction of input elements to zero during training, preventing co-adaptation of neurons.

**Training:**
$$y = m \odot x \cdot \frac{1}{1-p}$$
where $m \sim \text{Bernoulli}(1-p)$

**Inference:**
$$y = x$$ (identity)

**Inverted Dropout:**
The scaling $1/(1-p)$ is applied during training, so no adjustment is needed at test time. This is the modern standard.

**Why it works:**
- Each training iteration uses a different 'thinned' network
- At test time, averaging predictions over all $2^N$ possible networks approximates the geometric mean""",
    'theory_zh': r"""Dropout 在训练时随机将一部分输入元素置零，防止神经元共适应。

**训练：**
$$y = m \odot x \cdot \frac{1}{1-p}$$
其中 $m \sim \text{Bernoulli}(1-p)$

**推理：**
$$y = x$$（恒等映射）

**Inverted Dropout：**
缩放 $1/(1-p)$ 在训练时应用，因此测试时无需调整。这是现代标准做法。

**工作原理：**
- 每次训练迭代使用不同的'稀疏'网络
- 测试时，对所有 $2^N$ 可能网络的预测取平均近似几何平均""",
    "tests": [
        {
            "name": 'Eval mode is identity',
            "code": """import torch, torch.nn as nn
d = {fn}(p=0.5)
assert isinstance(d, nn.Module), 'Must inherit from nn.Module'
d.eval()
x = torch.randn(4, 8)
assert torch.equal(d(x), x), 'eval mode should return input unchanged'""",
        },
        {
            "name": 'Training: zeros and scaling',
            "code": """import torch
torch.manual_seed(42)
d = {fn}(p=0.5)
d.train()
x = torch.ones(1000)
out = d(x)
assert (out == 0).any(), 'No zeros found during training'
non_zero = out[out != 0]
assert torch.allclose(non_zero, torch.full_like(non_zero, 2.0), atol=1e-5), 'Non-zeros should be scaled by 1/(1-p)=2.0'""",
        },
        {
            "name": 'Drop rate is approximately p',
            "code": """import torch
torch.manual_seed(0)
d = {fn}(p=0.3)
d.train()
out = d(torch.ones(10000))
frac = (out == 0).float().mean().item()
assert 0.25 < frac < 0.35, f'Expected ~30%% zeros, got {frac*100:.1f}%%'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
d = {fn}(p=0.5)
d.train()
x = torch.randn(4, 8, requires_grad=True)
d(x).sum().backward()
assert x.grad is not None, 'x.grad is None'""",
        },
    ],
    'solution': '''class MyDropout(nn.Module):
    """实现反向缩放 Dropout (Inverted Dropout)。"""
    def __init__(self, p=0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must satisfy 0 <= p < 1")
        self.p = p

    def forward(self, x):
        # Step 1: 推理模式直接返回输入
        if not self.training or self.p == 0.0:
            return x

        # Step 2: 训练模式采样保留掩码
        keep_mask = (torch.rand_like(x) > self.p).to(x.dtype)

        # Step 3: 对存活元素反向缩放以保持期望不变
        out = x * keep_mask / (1.0 - self.p)
        return out''',
    'demo': """torch.manual_seed(42)
layer = MyDropout(p=0.5)
x = torch.ones(12)
layer.train()
print("Training output:", layer(x))
layer.eval()
print("Evaluation output:", layer(x))""",
    'explanation': r"""### 核心思路
训练时采样 $m\sim\operatorname{Bernoulli}(1-p)$ 并计算 $y=x\odot m/(1-p)$，使输出期望保持为输入；推理时执行恒等映射。

### 代码步骤拆解
- **Step 1（模式判断）**：当模块处于 `eval()` 或 `p == 0` 时直接返回 `x`。
- **Step 2（随机掩码）**：用 `torch.rand_like(x) > p` 生成与输入同形状的保留掩码。
- **Step 3（反向缩放）**：训练输出除以 $1-p$，存活值被放大而被丢弃位置为零。

### 重点与易错点分析
- **注意点 1（模块接口）**：应继承 `nn.Module` 并依赖 `self.training`，不能用独立函数参数替代 `train()`/`eval()` 状态。
- **注意点 2（概率边界）**：要求 $0\le p<1$；$p=1$ 会导致缩放分母为零。
- **注意点 3（期望保持）**：缩放必须发生在训练阶段，现代 inverted dropout 在推理阶段无需再乘 $1-p$。

### 复杂度分析
- **时间复杂度**：$O(N)$，训练时需为每个输入元素采样并缩放。
- **空间复杂度**：$O(N)$，训练时随机掩码是主要额外空间。""",
}
