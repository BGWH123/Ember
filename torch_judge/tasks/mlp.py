"""SwiGLU MLP task."""

TASK = {
    'title': 'SwiGLU MLP',
    'title_zh': 'SwiGLU MLP',
    'difficulty': 'Medium',
    'category': 'Transformer组件',
    'function_name': 'SwiGLUMLP',
    'starter_code': """class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        pass

    def forward(self, x):
        pass""",
    'hint': """`gate_proj(d→d_ff)`, `up_proj(d→d_ff)`, `down_proj(d_ff→d)`. Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`. `silu(x) = x * sigmoid(x)`.""",
    'hint_zh': """`gate_proj(d→d_ff)`、`up_proj(d→d_ff)`、`down_proj(d_ff→d)`。前向：`down_proj(silu(gate_proj(x)) * up_proj(x))`。`silu(x) = x * sigmoid(x)`。""",
    'description_en': """Implement a SwiGLU MLP as an nn.Module.

SwiGLU is a gated MLP variant used in LLaMA and other modern LLMs: it applies SiLU gating to one projection and element-wise multiplies with another.

**Signature:** `SwiGLUMLP(d_model, d_ff)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (*, d_model)

**Returns:** output tensor (*, d_model)

**Constraints:**
- Three projections: gate_proj(d, d_ff), up_proj(d, d_ff), down_proj(d_ff, d)
- `forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))`
- SiLU(x) = x * sigmoid(x)""",
    'description_zh': """实现 SwiGLU MLP（nn.Module）。

SwiGLU 是 LLaMA 等现代 LLM 使用的门控 MLP 变体：对一个投影应用 SiLU 门控，与另一个投影逐元素相乘。

**签名:** `SwiGLUMLP(d_model, d_ff)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, d_model)

**返回:** 输出张量 (*, d_model)

**约束:**
- 三个投影：gate_proj(d, d_ff)、up_proj(d, d_ff)、down_proj(d_ff, d)
- `forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))`
- SiLU(x) = x * sigmoid(x)""",
    "tests": [
        {
            "name": 'Parameter shapes',
            "code": """import torch, torch.nn as nn
mlp = {fn}(d_model=64, d_ff=128)
assert isinstance(mlp, nn.Module), 'Must inherit from nn.Module'
assert hasattr(mlp, 'gate_proj'), 'Need self.gate_proj'
assert hasattr(mlp, 'up_proj'), 'Need self.up_proj'
assert hasattr(mlp, 'down_proj'), 'Need self.down_proj'
assert mlp.gate_proj.weight.shape == (128, 64), f'gate_proj shape: {mlp.gate_proj.weight.shape}'
assert mlp.up_proj.weight.shape == (128, 64), f'up_proj shape: {mlp.up_proj.weight.shape}'
assert mlp.down_proj.weight.shape == (64, 128), f'down_proj shape: {mlp.down_proj.weight.shape}'""",
        },
        {
            "name": 'Forward output shape',
            "code": """import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(2, 8, 32)
out = mlp(x)
assert out.shape == (2, 8, 32), f'Output shape: {out.shape}'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch, torch.nn.functional as F
torch.manual_seed(0)
mlp = {fn}(d_model=16, d_ff=32)
x = torch.randn(1, 4, 16)
out = mlp(x)
gate = mlp.gate_proj(x)
up = mlp.up_proj(x)
ref = mlp.down_proj(F.silu(gate) * up)
assert torch.allclose(out, ref, atol=1e-5), 'Output != down(silu(gate(x)) * up(x))'""",
        },
        {
            "name": '2-D input',
            "code": """import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(4, 32)
out = mlp(x)
assert out.shape == (4, 32), f'2-D output shape: {out.shape}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(2, 4, 32, requires_grad=True)
mlp(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in mlp.parameters())
n_grad = sum(1 for p in mlp.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'""",
        },
    ],
    'solution': '''class SwiGLUMLP(nn.Module):
    """实现门控前馈网络 (SwiGLU MLP)。"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        # Step 1: 创建门控、上投影与下投影层
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Step 2: 对门控投影应用 SiLU 并与上投影逐元素相乘
        gate = self.gate_proj(x)
        gated_hidden = gate * torch.sigmoid(gate) * self.up_proj(x)

        # Step 3: 将中间特征投影回模型维度
        out = self.down_proj(gated_hidden)
        return out''',
    'demo': """torch.manual_seed(42)
mlp = SwiGLUMLP(d_model=32, d_ff=64)
x = torch.randn(2, 8, 32)
out = mlp(x)
print("Output shape:", out.shape)
print("Parameter count:", sum(p.numel() for p in mlp.parameters()))
print("Sample values:", out[0, 0, :4])""",
    'explanation': r"""### 核心思路
SwiGLU 计算 $W_{down}(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x)$，用门控分支动态调制上投影特征，再恢复到 `d_model`。支持任意前导维。

### 代码步骤拆解
- **Step 1（三路投影）**：建立两个 `d_model -> d_ff` 层和一个反向下投影。
- **Step 2（门控融合）**：显式计算 `gate * sigmoid(gate)` 并与 `up_proj(x)` 相乘。
- **Step 3（维度恢复）**：`down_proj` 把最后一维从 `d_ff` 恢复为 `d_model`。

### 重点与易错点分析
- **注意点 1（SiLU 公式）**：不能只使用 sigmoid；SiLU 是 $x\sigma(x)$。
- **注意点 2（逐元素门控）**：两个上行分支必须都是 `d_ff`，才能逐位置逐通道相乘。
- **注意点 3（参数规模）**：相比普通两层 MLP 多一个上行投影，通常会相应调整 `d_ff` 控制总参数量。

### 复杂度分析
- **时间复杂度**：$O(Md_{model}d_{ff})$，三次线性投影占主导。
- **空间复杂度**：$O(Md_{ff})$，保存门控与上投影激活。""",
}
