"""Stochastic Depth (DropPath) task."""

TASK = {
    'title': 'Stochastic Depth (DropPath)',
    'title_zh': '随机深度（DropPath）',
    'difficulty': 'Medium',
    'category': '正则化',
    'function_name': 'StochasticDepth',
    'starter_code': """class StochasticDepth(nn.Module):
    pass""",
    'hint': 'Use `torch.rand()` to sample a Bernoulli mask. In eval mode, scale f(x) by (1 - drop_prob).',
    'hint_zh': '使用 `torch.rand()` 采样 Bernoulli 掩码。推理模式下将 f(x) 缩放 (1 - drop_prob)。',
    'description_en': """Implement Stochastic Depth (DropPath) regularization. During training, randomly drop entire residual branches with probability `drop_prob`: output = x + b * f(x), where b ~ Bernoulli(1 - drop_prob). During evaluation, use the full network: output = x + (1 - drop_prob) * f(x) (expectation over Bernoulli).""",
    'description_zh': """实现随机深度（DropPath）正则化。训练时以概率 `drop_prob` 随机丢弃整个残差分支：output = x + b * f(x)，其中 b ~ Bernoulli(1 - drop_prob)。推理时使用完整网络：output = x + (1 - drop_prob) * f(x)（Bernoulli 的期望）。""",
    'theory_en': """Stochastic Depth randomly drops layers during training, effectively training an ensemble of networks with varying depth. It reduces overfitting and improves generalization, especially in deep networks like Vision Transformers.""",
    'theory_zh': '随机深度在训练时随机丢弃层，相当于训练了一个具有不同深度的网络集成。它减少过拟合、提升泛化，尤其在深层网络（如 ViT）中效果显著。',
    "tests": [
        {
            "name": 'Evaluation expectation',
            "code": """layer = {fn}(0.25)
layer.eval()
x = torch.ones(3, 2, 4); branch = torch.full_like(x, 2.)
assert torch.allclose(layer(x, branch), x + 0.75 * branch)""",
        },
        {
            "name": 'Per-sample branch dropping',
            "code": """torch.manual_seed(1234)
layer = {fn}(0.5)
layer.train()
x = torch.zeros(2048, 3, 4); branch = torch.full_like(x, 2.)
out = layer(x, branch)
assert torch.all((out == 0) | (out == 2))
assert torch.equal(out, out[:, :1, :1].expand_as(out))
assert abs(out.mean().item() - 1.) < 0.1""",
        },
        {
            "name": 'Zero and complete drop probabilities',
            "code": """x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
branch = torch.ones_like(x, requires_grad=True)
for p in [0., 1.]:
    layer = {fn}(p)
    for training in [True, False]:
        layer.train(training)
        out = layer(x, branch)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, x + (1 - p) * branch)""",
        },
    ],
    'solution': '''class StochasticDepth(nn.Module):
    """实现随机深度（DropPath）（Stochastic Depth (DropPath)）。"""

    def __init__(self, drop_prob=0.1):
        # Step 1: 保存路径丢弃率及其对应的 Bernoulli 保留概率
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob

    def forward(self, x, f_x):
        # Step 2: 训练时为每个样本独立决定是否保留整条残差分支
        if self.training:
            if self.drop_prob == 0.0:
                return x + f_x
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.rand(shape, device=x.device) < self.keep_prob
            return x + f_x * mask
        else:
            # Step 3: 推理时用保留概率缩放分支以匹配训练随机输出的期望
            return x + self.keep_prob * f_x''',
    'demo': """torch.manual_seed(42)
x = torch.zeros(16, 3, 4)
residual = torch.ones_like(x)
layer = StochasticDepth(drop_prob=0.2)
train_out = layer(x, residual)
layer.eval()
eval_out = layer(x, residual)
print("Kept training paths:", int(train_out[:, 0, 0].sum().item()), "of", x.shape[0])
print("Eval branch value:", eval_out[0, 0, 0].item())
print("Path mask constant within each sample:", torch.equal(train_out, train_out[:, :1, :1].expand_as(train_out)))""",
    'explanation': r"""### 核心思路
随机深度作用于整条残差路径而非单个激活。按本题约定，训练时 $y_b=x_b+m_bf(x_b)$，其中 $m_b\sim\operatorname{Bernoulli}(1-p)$；因此 $\mathbb E[y_b]=x_b+(1-p)f(x_b)$，推理时直接使用该期望，不再采样。

### 代码步骤拆解
- **Step 1（记录保留概率）**：构造函数保存 `drop_prob=p` 与 `keep_prob=1-p`，后者同时决定训练 Bernoulli 参数和推理缩放。
- **Step 2（逐样本采样路径）**：训练时 mask 形状为 `(B,1,...,1)`，每个样本只有一个随机决策，并广播到其全部通道、token 或空间位置。
- **Step 3（确定性期望推理）**：训练返回 `x + f_x * mask`；评估返回 `x + keep_prob * f_x`，使推理输出等于训练随机网络的均值。

### 重点与易错点分析
- **注意点 1（DropPath 不是 Dropout）**：若 mask 与 `x` 同形状，会逐元素丢弃而不是跳过整条残差分支；仅批次维变化才对应每个样本的网络深度随机化。
- **注意点 2（缩放约定）**：常见库采用训练时除以 `keep_prob`、推理时不缩放的 inverted DropPath；本题明确采用训练时不放大、推理时乘 `keep_prob`，两种约定不能混用。
- **注意点 3（端点行为）**：`drop_prob=0` 时训练和推理均为 `x+f_x`；`drop_prob=1` 时训练 mask 全零且推理尺度为零，两种模式都退化为恒等残差 `x`，不会发生除零。

### 复杂度分析
- **时间复杂度**：$O(N)$，其中 $N$ 是 `x` 的元素数，残差乘加遍历全部激活。
- **空间复杂度**：除输出 $O(N)$ 外，随机路径 mask 只需 $O(B)$，远小于逐元素 dropout 的 mask。""",
}
