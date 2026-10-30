"""LoRA (Low-Rank Adaptation) task."""

TASK = {
    'title': 'LoRA (Low-Rank Adaptation)',
    'title_zh': 'LoRA（低秩适配）',
    'difficulty': 'Medium',
    'category': '参数高效微调',
    'function_name': 'LoRALinear',
    'starter_code': """class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        pass

    def forward(self, x):
        pass""",
    'hint': """1. Freeze `linear.weight/bias` with `requires_grad_(False)`
2. `lora_A`: `(rank, in)`, `lora_B`: `(out, rank)` init zeros
3. Forward: `linear(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)`""",
    'hint_zh': """1. `linear.weight/bias` 设 `requires_grad_(False)` 冻结
2. `lora_A`: `(rank, in)`，`lora_B`: `(out, rank)` 初始化为零
3. 前向：`linear(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)`""",
    'description_en': """Implement LoRA (Low-Rank Adaptation) for a linear layer.

LoRA freezes the base weights and adds trainable low-rank matrices A and B, enabling efficient fine-tuning with far fewer parameters.

**Signature:** `LoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (*, in_features)

**Returns:** `linear(x) + (x @ A^T @ B^T) * (alpha/rank)`

**Constraints:**
- Base `nn.Linear` weights must be frozen (requires_grad=False)
- `lora_A`: (rank, in_features), `lora_B`: (out_features, rank) initialized to zeros
- Only LoRA params should receive gradients""",
    'description_zh': """实现线性层的 LoRA（低秩适配）。

LoRA 冻结基础权重并添加可训练的低秩矩阵 A 和 B，以极少的参数实现高效微调。

**签名:** `LoRALinear(in_features, out_features, rank, alpha=1.0)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, in_features)

**返回:** `linear(x) + (x @ A^T @ B^T) * (alpha/rank)`

**约束:**
- 基础 `nn.Linear` 权重必须冻结（requires_grad=False）
- `lora_A`：(rank, in_features)，`lora_B`：(out_features, rank) 初始化为零
- 只有 LoRA 参数应接收梯度""",
    "tests": [
        {
            "name": 'Base weights frozen',
            "code": """import torch, torch.nn as nn
layer = {fn}(in_features=16, out_features=8, rank=4)
assert isinstance(layer, nn.Module)
assert not layer.linear.weight.requires_grad, 'Base weight must be frozen'
assert not layer.linear.bias.requires_grad, 'Base bias must be frozen'""",
        },
        {
            "name": 'LoRA parameter shapes',
            "code": """import torch
layer = {fn}(in_features=16, out_features=8, rank=4)
assert layer.lora_A.shape == (4, 16), f'lora_A: {layer.lora_A.shape}'
assert layer.lora_B.shape == (8, 4), f'lora_B: {layer.lora_B.shape}'""",
        },
        {
            "name": 'B=0 means output equals base',
            "code": """import torch
torch.manual_seed(0)
layer = {fn}(in_features=8, out_features=4, rank=2)
x = torch.randn(2, 8)
assert torch.allclose(layer(x), layer.linear(x), atol=1e-5), 'With B=0, should equal base'""",
        },
        {
            "name": 'Only LoRA params get gradients',
            "code": """import torch
layer = {fn}(in_features=8, out_features=4, rank=2)
layer(torch.randn(2, 8)).sum().backward()
assert layer.lora_A.grad is not None, 'lora_A.grad is None'
assert layer.lora_B.grad is not None, 'lora_B.grad is None'
assert layer.linear.weight.grad is None, 'Base weight should not have grad'""",
        },
        {
            "name": 'Forward computation',
            "code": """import torch
torch.manual_seed(0)
layer = {fn}(in_features=8, out_features=4, rank=2, alpha=2.0)
layer.lora_B.data.normal_()
x = torch.randn(3, 8)
ref = layer.linear(x) + (x @ layer.lora_A.T @ layer.lora_B.T) * (2.0 / 2)
assert torch.allclose(layer(x), ref, atol=1e-5), 'Forward mismatch'""",
        },
    ],
    'solution': '''class LoRALinear(nn.Module):
    """实现线性层低秩适配 (Low-Rank Adaptation)。"""

    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        # Step 1: 创建并冻结基础线性层
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)

        # Step 2: 初始化低秩可训练参数与缩放系数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        # Step 3: 将低秩增量叠加到冻结基础输出
        base_out = self.linear(x)
        low_rank_out = x @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * low_rank_out''',
    'demo': """torch.manual_seed(42)
layer = LoRALinear(16, 8, rank=4, alpha=8.0)
x = torch.randn(2, 16)
out = layer(x)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print("Output shape:", out.shape)
print("Trainable parameters:", trainable)
print("Initial base match:", torch.allclose(out, layer.linear(x)))""",
    'explanation': r"""### 核心思路
LoRA 冻结基础映射并学习低秩增量 $\Delta W=(\alpha/r)BA$，前向为 `linear(x) + scaling * x @ A.T @ B.T`。中间低秩维为 `rank`，显著减少可训练参数。

### 代码步骤拆解
- **Step 1（冻结主干）**：关闭基础权重和偏置的梯度。
- **Step 2（低秩参数）**：建立 `(rank, in)` 的 `lora_A` 与 `(out, rank)` 的 `lora_B`。
- **Step 3（增量合并）**：两次小矩阵乘法后按 `alpha / rank` 缩放并加到基础输出。

### 重点与易错点分析
- **注意点 1（初始化策略）**：`lora_B` 置零保证初始输出严格等于预训练基础层，同时 `lora_A` 保持非零以便 B 获得梯度。
- **注意点 2（梯度现象）**：首个反向步中 `lora_A.grad` 可为全零但不能为 `None`，因为零 B 暂时屏蔽其数值梯度。
- **注意点 3（秩与缩放）**：用 `alpha / rank` 可让调整 rank 时更新幅度更可控。

### 复杂度分析
- **时间复杂度**：$O(M(in\cdot out+rank(in+out)))$，基础线性层仍是主要开销。
- **空间复杂度**：$O(M(out+rank))$，额外保存低秩中间激活。""",
}
