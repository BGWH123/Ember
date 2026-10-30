"""Mixed Precision Training Step task."""

TASK = {
    'title': 'Mixed Precision Training Step',
    'title_zh': '混合精度训练步骤',
    'difficulty': 'Medium',
    'category': '高效训练',
    'function_name': 'mixed_precision_step',
    'starter_code': """def mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0):
    pass""",
    'hint': """1. model.half() → forward with fp16 input
2. compute loss → scale by loss_scale → backward
3. divide all param.grad by loss_scale (unscale)
4. model.float() → optimizer.step()
5. return unscaled loss as float""",
    'hint_zh': """1. model.half() → fp16 输入前向传播
2. 计算 loss → 乘以 loss_scale → 反向传播
3. 所有 param.grad 除以 loss_scale（反缩放）
4. model.float() → optimizer.step()
5. 返回未缩放的 loss（float）""",
    'description_en': """Implement one mixed precision training step.

Mixed precision training uses fp16 for the forward/backward pass (faster, less memory) while keeping fp32 master weights for numerical stability. Loss scaling prevents fp16 gradient underflow.

**Signature:** `mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0) -> float`

**Parameters:**
- `model` — nn.Module with fp32 parameters
- `optimizer` — optimizer holding fp32 params
- `loss_fn` — callable `(output, target) -> scalar loss`
- `x, y` — input and target tensors
- `loss_scale` — scalar to multiply loss before backward

**Returns:** unscaled loss value as float

**Steps:**
1. Cast model to fp16, run forward pass
2. Compute loss, scale it by `loss_scale`
3. Run backward on scaled loss
4. Unscale gradients (divide by `loss_scale`)
5. Update optimizer (fp32 master weights)
6. Restore model to fp32
7. Return original (unscaled) loss""",
    'description_zh': """实现一步混合精度训练。

混合精度训练用 fp16 做前向/反向传播（更快、更省内存），同时保留 fp32 主权重以保证数值稳定性。Loss scaling 防止 fp16 梯度下溢。

**签名:** `mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0) -> float`

**参数:**
- `model` — 持有 fp32 参数的 nn.Module
- `optimizer` — 持有 fp32 参数的优化器
- `loss_fn` — 可调用对象 `(output, target) -> 标量损失`
- `x, y` — 输入和目标张量
- `loss_scale` — 反向传播前乘以 loss 的缩放系数

**返回:** 未缩放的 loss 值（float）

**步骤:**
1. 将模型转为 fp16，运行前向传播
2. 计算 loss，乘以 `loss_scale`
3. 对缩放后的 loss 做反向传播
4. 反缩放梯度（除以 `loss_scale`）
5. 更新优化器（fp32 主权重）
6. 将模型恢复为 fp32
7. 返回原始（未缩放）loss""",
    "tests": [
        {
            "name": 'Returns float loss',
            "code": """import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(8, 4)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(4, 8)
y = torch.randn(4, 4)
loss_val = {fn}(model, opt, nn.functional.mse_loss, x, y)
assert isinstance(loss_val, float), f'Expected float, got {type(loss_val)}'
assert loss_val > 0, 'Loss should be positive'""",
        },
        {
            "name": 'Parameters are updated',
            "code": """import torch
import torch.nn as nn
torch.manual_seed(1)
model = nn.Linear(8, 4)
w_before = model.weight.data.clone()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(4, 8)
y = torch.randn(4, 4)
{fn}(model, opt, nn.functional.mse_loss, x, y)
assert not torch.allclose(model.weight.data, w_before), 'Weights should be updated after step'""",
        },
        {
            "name": 'Model restored to fp32 after step',
            "code": """import torch
import torch.nn as nn
torch.manual_seed(2)
model = nn.Linear(8, 4)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(4, 8)
y = torch.randn(4, 4)
{fn}(model, opt, nn.functional.mse_loss, x, y)
for p in model.parameters():
    assert p.dtype == torch.float32, f'Parameter should be fp32 after step, got {p.dtype}'""",
        },
        {
            "name": 'fp16 used in forward pass',
            "code": """import torch
import torch.nn as nn

class DtypeCapture(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.captured_dtype = None
    def forward(self, x):
        self.captured_dtype = x.dtype
        return self.linear(x)

torch.manual_seed(0)
model = nn.Sequential(DtypeCapture(nn.Linear(8, 8)), nn.Linear(8, 4))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(2, 8)
y = torch.randn(2, 4)
loss_fn = nn.MSELoss()
{fn}(model, optimizer, loss_fn, x, y)
assert model[0].captured_dtype == torch.float16, f'Expected fp16 in forward, got {model[0].captured_dtype}'""",
        },
        {
            "name": 'Loss scale affects gradients',
            "code": """import torch
import torch.nn as nn
torch.manual_seed(3)
model1 = nn.Linear(4, 2)
model2 = nn.Linear(4, 2)
model2.load_state_dict(model1.state_dict())
opt1 = torch.optim.SGD(model1.parameters(), lr=0.0)  # lr=0 to keep grads
opt2 = torch.optim.SGD(model2.parameters(), lr=0.0)
x = torch.randn(2, 4)
y = torch.randn(2, 2)
{fn}(model1, opt1, nn.functional.mse_loss, x, y, loss_scale=1.0)
{fn}(model2, opt2, nn.functional.mse_loss, x, y, loss_scale=1024.0)
# After unscaling, gradients should be the same regardless of loss_scale
for p1, p2 in zip(model1.parameters(), model2.parameters()):
    if p1.grad is not None and p2.grad is not None:
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), 'Unscaled grads should match regardless of loss_scale'""",
        },
    ],
    'solution': '''def mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0):
    """执行一次手动损失缩放的混合精度训练 (Mixed Precision Step)。"""
    # Step 1: 将模型和输入转为 FP16 完成前向计算
    model.half()
    output = model(x.half())
    loss = loss_fn(output.float(), y)
    loss_value = loss.item()

    # Step 2: 缩放损失反传并将梯度反缩放到 FP32
    optimizer.zero_grad()
    (loss * loss_scale).backward()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = param.grad.data.float() / loss_scale

    # Step 3: 恢复 FP32 参数后执行优化器更新
    model.float()
    optimizer.step()
    return loss_value''',
    'demo': """torch.manual_seed(42)
model = nn.Linear(8, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(4, 8)
y = torch.randn(4, 4)
loss = mixed_precision_step(model, optimizer, F.mse_loss, x, y)
print("Loss:", loss)
print("Model dtype:", model.weight.dtype)
print("Weight shape:", model.weight.shape)""",
    'explanation': """### 核心思路
混合精度步骤在 FP16 中执行前向与反向，通过 `loss_scale` 放大微小梯度，再除回原尺度并恢复 FP32 后更新参数。函数返回未缩放损失的 Python `float`。

### 代码步骤拆解
- **Step 1（半精度前向）**：调用 `model.half()` 并把 `x` 转为 FP16。
- **Step 2（缩放反传）**：损失乘 `loss_scale` 后 backward，再逐参数将梯度转 FP32 并除缩放。
- **Step 3（全精度更新）**：`model.float()` 后调用 `optimizer.step()`。

### 重点与易错点分析
- **注意点 1（目标精度）**：损失输入 `output.float()` 可避免 CPU 上半精度 MSE 反向算子受限，并保持目标 `y` 的精度。
- **注意点 2（更新顺序）**：必须在优化器 step 前恢复参数和梯度到 FP32。
- **注意点 3（工程限制）**：这是教学版手动实现；生产环境应使用 autocast 与 GradScaler 检测溢出并维护真正的主权重。

### 复杂度分析
- **时间复杂度**：$O(P+F)$，参数类型转换与模型前后向共同构成开销。
- **空间复杂度**：$O(P+A)$，保存参数梯度和前向激活。""",
}
