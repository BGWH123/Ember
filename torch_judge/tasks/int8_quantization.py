"""INT8 Quantized Linear task."""

TASK = {
    'title': 'INT8 Quantized Linear',
    'title_zh': 'INT8 量化线性层',
    'difficulty': 'Hard',
    'category': '量化',
    'function_name': 'Int8Linear',
    'starter_code': """class Int8Linear(nn.Module):
    pass""",
    'hint': """1. scale = abs(weight).max(dim=1) / 127  (per output channel, keepdim)
2. weight_int8 = round(weight / scale).clamp(-128, 127).to(int8)
3. register both as buffers (not parameters)
4. forward: dequant = weight_int8.float() * scale → x @ dequant.T""",
    'hint_zh': """1. scale = abs(weight).max(dim=1) / 127（逐输出通道，keepdim）
2. weight_int8 = round(weight / scale).clamp(-128, 127).to(int8)
3. 两者均注册为 buffer（非 parameter）
4. 前向：dequant = weight_int8.float() * scale → x @ dequant.T""",
    'description_en': """Implement INT8 per-channel weight quantization for a linear layer.

Quantization converts float32 weights to int8 with per-channel scaling, reducing model size by 4x while preserving accuracy.

**Signature:** `Int8Linear(weight, bias=None)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (*, in_features)

**Returns:** linear output with dequantized weights

**Constraints:**
- Per-channel scale: `abs(weight).max(dim=1) / 127`
- Quantize: `round(weight/scale).clamp(-128, 127).to(int8)`
- Store weight_int8 and scale as buffers, not parameters""",
    'description_zh': """实现 INT8 逐通道权重量化线性层。

量化将 float32 权重转换为 int8 并使用逐通道缩放，将模型大小减少 4 倍同时保持精度。

**签名:** `Int8Linear(weight, bias=None)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, in_features)

**返回:** 使用反量化权重的线性输出

**约束:**
- 逐通道缩放：`abs(weight).max(dim=1) / 127`
- 量化：`round(weight/scale).clamp(-128, 127).to(int8)`
- weight_int8 和 scale 存储为 buffer 而非 parameter""",
    "tests": [
        {
            "name": 'Weight is int8',
            "code": """import torch, torch.nn as nn
w = torch.randn(32, 16)
q = {fn}(w)
assert isinstance(q, nn.Module)
assert q.weight_int8.dtype == torch.int8, f'dtype: {q.weight_int8.dtype}'""",
        },
        {
            "name": 'Values in [-128, 127]',
            "code": """import torch
q = {fn}(torch.randn(64, 32) * 10)
assert q.weight_int8.min() >= -128 and q.weight_int8.max() <= 127""",
        },
        {
            "name": 'Dequantized close to original',
            "code": """import torch
torch.manual_seed(0)
w = torch.randn(16, 8)
q = {fn}(w)
w_recon = q.weight_int8.float() * q.scale
assert (w - w_recon).abs().max() < 0.1, 'Quantization error too large'""",
        },
        {
            "name": 'Forward with bias correctness',
            "code": """import torch
torch.manual_seed(0)
w = torch.randn(8, 4)
b = torch.randn(8)
q_bias = {fn}(w, b)
q_no_bias = {fn}(w)
x = torch.randn(2, 4)
out_bias = q_bias(x)
out_no = q_no_bias(x)
assert out_bias.shape == (2, 8), f'Shape: {out_bias.shape}'
assert torch.allclose(out_bias - out_no, b.unsqueeze(0).expand(2, -1), atol=1e-5), 'Bias not correctly added'""",
        },
        {
            "name": 'Weight is buffer not parameter',
            "code": """import torch
q = {fn}(torch.randn(4, 4))
param_names = [n for n, _ in q.named_parameters()]
assert 'weight_int8' not in param_names, 'weight_int8 should be a buffer'
assert 'scale' not in param_names, 'scale should be a buffer'""",
        },
    ],
    'solution': '''class Int8Linear(nn.Module):
    """实现逐输出通道量化线性层 (INT8 Quantized Linear)。"""

    def __init__(self, weight, bias=None):
        super().__init__()
        # Step 1: 计算逐输出通道尺度并量化权重
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        safe_scale = scale.clamp(min=1e-10)
        weight_int8 = torch.round(weight / safe_scale).clamp(-128, 127).to(torch.int8)

        # Step 2: 将量化权重和尺度注册为缓冲区
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x):
        # Step 3: 动态反量化并执行线性映射
        weight = self.weight_int8.float() * self.scale
        out = x @ weight.T
        return out if self.bias is None else out + self.bias''',
    'demo': """torch.manual_seed(42)
weight = torch.randn(8, 4)
layer = Int8Linear(weight, torch.randn(8))
out = layer(torch.randn(2, 4))
print("Output shape:", out.shape)
print("Weight dtype:", layer.weight_int8.dtype)
print("Max weight error:", (weight - layer.weight_int8.float() * layer.scale).abs().max().item())""",
    'explanation': r"""### 核心思路
对每个输出通道使用 $s_i=\max|W_i|/127$，将权重舍入为 INT8；前向时按通道反量化后执行普通线性映射。

### 代码步骤拆解
- **Step 1（逐通道量化）**：`scale` 为 `(out_features, 1)`，广播覆盖每行权重。
- **Step 2（状态注册）**：`weight_int8` 与 `scale` 是 buffer，偏置仍可作为参数。
- **Step 3（反量化前向）**：恢复浮点 `weight`，计算 `x @ weight.T` 并加偏置。

### 重点与易错点分析
- **注意点 1（量化轴）**：最大值沿输入维 `dim=1` 计算，每个输出通道独享尺度。
- **注意点 2（零权重行）**：量化除法使用 `safe_scale` 防止零除；保存尺度仍遵循原公式。
- **注意点 3（缓冲区语义）**：量化权重不能注册为参数，否则优化器会尝试更新整数张量。

### 复杂度分析
- **时间复杂度**：量化为 $O(OI)$，前向为 $O(MOI)$。
- **空间复杂度**：$O(OI)$，前向反量化会临时创建浮点权重。""",
}
