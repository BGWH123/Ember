"""QLoRA task."""

TASK = {
    'title': 'QLoRA',
    'title_zh': 'QLoRA',
    'difficulty': 'Hard',
    'category': '量化',
    'function_name': 'QLoRALinear',
    'starter_code': """class QLoRALinear(nn.Module):
    pass""",
    'hint': """set_weight: `scale = W.abs().max(dim=1,keepdim=True).values.clamp(1e-8)/127`
           `q = (W/scale).round().clamp(-127,127).to(int8)`
Forward: `W_fp = q.float()*scale` → `x@W_fp.T + x@lora_A@lora_B*(alpha/rank)`""",
    'hint_zh': """set_weight：`scale = W.abs().max(dim=1,keepdim=True).values.clamp(1e-8)/127`
            `q = (W/scale).round().clamp(-127,127).to(int8)`
前向：`W_fp = q.float()*scale` → `x@W_fp.T + x@lora_A@lora_B*(alpha/rank)`""",
    'description_en': """Implement QLoRA: a quantized linear layer with LoRA adapters.

QLoRA stores the base weight in int8 (simulating 4-bit quantization) with a per-row fp32 scale, while LoRA adapters remain in full precision for efficient fine-tuning.

**Signature:** `QLoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)

**Parameters stored:**
- `quantized_weight`: (out_features, in_features), dtype=int8, requires_grad=False
- `scale`: (out_features, 1), fp32 per-row scale
- `lora_A`: (in_features, rank), initialized ~ N(0, 0.01²)
- `lora_B`: (rank, out_features), initialized to zeros

**Forward:**
1. Dequantize: `W_fp = quantized_weight.float() * scale`
2. Base output: `x @ W_fp.T`
3. LoRA delta: `x @ lora_A @ lora_B * (alpha / rank)`
4. Return base + delta

**Helper:** `set_weight(W_fp32)` quantizes a float weight per-row:
- `scale = W.abs().max(dim=1, keepdim=True).values / 127`
- `quantized = (W / scale).round().clamp(-127, 127).to(int8)`""",
    'description_zh': """实现 QLoRA：带 LoRA 适配器的量化线性层。

QLoRA 将基础权重以 int8 格式存储（模拟 4-bit 量化），配合每行 fp32 缩放因子，同时 LoRA 适配器保持全精度，实现高效微调。

**签名:** `QLoRALinear(in_features, out_features, rank, alpha=1.0)`（nn.Module）

**存储的参数:**
- `quantized_weight`：(out_features, in_features)，dtype=int8，requires_grad=False
- `scale`：(out_features, 1)，fp32 每行缩放因子
- `lora_A`：(in_features, rank)，初始化 ~ N(0, 0.01²)
- `lora_B`：(rank, out_features)，初始化为零

**前向传播:**
1. 反量化：`W_fp = quantized_weight.float() * scale`
2. 基础输出：`x @ W_fp.T`
3. LoRA 增量：`x @ lora_A @ lora_B * (alpha / rank)`
4. 返回 base + delta

**辅助方法:** `set_weight(W_fp32)` 按行量化浮点权重：
- `scale = W.abs().max(dim=1, keepdim=True).values / 127`
- `quantized = (W / scale).round().clamp(-127, 127).to(int8)`""",
    "tests": [
        {
            "name": 'Output shape correct',
            "code": """import torch, torch.nn as nn
layer = {fn}(in_features=16, out_features=8, rank=4)
assert isinstance(layer, nn.Module)
x = torch.randn(3, 16)
out = layer(x)
assert out.shape == (3, 8), f'Expected (3, 8), got {out.shape}'""",
        },
        {
            "name": 'Dequantized weight close to original',
            "code": """import torch
torch.manual_seed(42)
layer = {fn}(in_features=16, out_features=8, rank=4)
W = torch.randn(8, 16)
layer.set_weight(W)
W_fp = layer.quantized_weight.float() * layer.scale
assert torch.allclose(W, W_fp, atol=0.02), f'Max dequant error: {(W - W_fp).abs().max().item():.4f}'""",
        },
        {
            "name": 'LoRA delta is zero when lora_B is zero',
            "code": """import torch
torch.manual_seed(0)
layer = {fn}(in_features=8, out_features=4, rank=2)
W = torch.randn(4, 8)
layer.set_weight(W)
x = torch.randn(2, 8)
W_fp = layer.quantized_weight.float() * layer.scale
base = x @ W_fp.T
out = layer(x)
assert torch.allclose(out, base, atol=1e-5), 'With lora_B=0, output should equal base'""",
        },
        {
            "name": 'Gradient flows through LoRA, not quantized_weight',
            "code": """import torch
layer = {fn}(in_features=8, out_features=4, rank=2)
layer(torch.randn(2, 8)).sum().backward()
assert layer.lora_A.grad is not None, 'lora_A should have grad'
assert layer.lora_B.grad is not None, 'lora_B should have grad'
assert layer.quantized_weight.grad is None, 'quantized_weight must not have grad'""",
        },
        {
            "name": 'set_weight then forward close to full-precision linear',
            "code": """import torch
torch.manual_seed(7)
layer = {fn}(in_features=16, out_features=8, rank=4)
W = torch.randn(8, 16)
layer.set_weight(W)
# zero out LoRA so we isolate the quantized base
layer.lora_B.data.zero_()
x = torch.randn(5, 16)
out = layer(x)
ref = x @ W.T
assert torch.allclose(out, ref, atol=0.05), f'Max error vs fp32: {(out - ref).abs().max().item():.4f}'""",
        },
    ],
    'solution': '''class QLoRALinear(nn.Module):
    """QLoRA (QLoRA)。"""

    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        # Step 1: 保存冻结的逐行 int8 基座，并初始化可训练的低秩 A、B 因子
        self.rank = rank
        self.alpha = alpha
        self.quantized_weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def set_weight(self, W_fp32):
        # Step 2: 以每个输出通道的绝对最大值为尺度，将浮点权重量化到 [-127,127]
        scale = W_fp32.abs().max(dim=1, keepdim=True).values.clamp(min=1e-08) / 127.0
        quantized_weight = (W_fp32 / scale).round().clamp(-127, 127).to(torch.int8)
        self.quantized_weight.data.copy_(quantized_weight)
        self.scale.data.copy_(scale)

    def forward(self, x):
        # Step 3: 反量化基座并叠加缩放后的 xAB 低秩增量
        dequantized_weight = self.quantized_weight.float() * self.scale
        base_out = x @ dequantized_weight.T
        lora_out = x @ self.lora_A @ self.lora_B * (self.alpha / self.rank)
        return base_out + lora_out''',
    'demo': """torch.manual_seed(42)
in_f, out_f, rank = (64, 32, 4)
layer = QLoRALinear(in_f, out_f, rank, alpha=2.0)
W_ref = torch.randn(out_f, in_f)
layer.set_weight(W_ref)
x = torch.randn(8, in_f)
y_qlora = layer(x)
y_ref = x @ W_ref.T
print('Output shape:', y_qlora.shape)
print('Max abs error vs fp32:', (y_qlora - y_ref).abs().max().item())""",
    'explanation': r"""### 核心思路
该层将基础权重 $W\in\mathbb R^{O\times I}$ 按输出行量化为整数 $Q$ 和尺度 $s$，前向时近似恢复 $\hat W=Q\odot s$；只有低秩适配器参与训练。输出为 $Y=X\hat W^\top+(\alpha/r)XAB$，其中 $A\in\mathbb R^{I\times r}$、$B\in\mathbb R^{r\times O}$，以 $r\ll\min(I,O)$ 降低可训练参数量。

### 代码步骤拆解
- **Step 1（建立冻结基座与适配器）**：`quantized_weight` 是 `(out_features,in_features)` 的不可训练 int8 参数，`scale` 为逐行 `(out_features,1)`；`lora_A` 小随机初始化，`lora_B` 置零以保证初始增量为零。
- **Step 2（逐行对称量化）**：`set_weight` 令每行 `scale=max(abs(W_row))/127`，将 `W_fp32/scale` 四舍五入并裁到 `[-127,127]`，分别复制到整数权重和尺度。
- **Step 3（基座与低秩双路径）**：`dequantized_weight.T` 参与普通线性映射；并行计算 `x @ lora_A @ lora_B * (alpha/rank)`，两路同为 `(...,out_features)` 后相加。

### 重点与易错点分析
- **注意点 1（逐行尺度轴）**：尺度必须沿 `in_features` 求绝对最大值并保留列维，才能广播到每个输出通道整行；全零行用 `clamp(1e-8)` 避免量化除零。
- **注意点 2（初始化与梯度）**：`lora_B=0` 使首次前向完全等于量化基座；首个反向中 B 有梯度而 A 的梯度可为零，更新 B 后 A 才接收非零信号。整数基座不能求梯度。
- **注意点 3（教学量化与真实 QLoRA）**：本题用 int8 对称量化模拟低比特基座；真实 QLoRA 常用 NF4、双重量化和分页优化器。反量化成完整浮点矩阵便于理解，但不会实现真实低比特 GEMM 的峰值显存优势。

### 复杂度分析
- **时间复杂度**：设输入含 $N$ 个向量，基座乘法为 $O(NIO)$，LoRA 两次乘法为 $O(Nr(I+O))$；量化设置权重为 $O(IO)$。
- **空间复杂度**：持久存储为 $O(IO)$ 个 int8 权重、$O(O)$ 个尺度及 $O(r(I+O))$ 个适配器参数；此实现前向反量化临时张量占 $O(IO)$。""",
}
