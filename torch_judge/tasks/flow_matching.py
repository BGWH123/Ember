"""Flow Matching Loss task."""

TASK = {
    'title': 'Flow Matching Loss',
    'title_zh': '流匹配损失',
    'difficulty': 'Easy',
    'category': '损失函数',
    'function_name': 'flow_matching_loss',
    'starter_code': """def flow_matching_loss(model_output, x0, x1, t):
    pass""",
    'hint': """target = x1 - x0  (straight-line velocity from noise to data)
loss = ((model_output - target) ** 2).mean()
`t` is not used in the loss itself.""",
    'hint_zh': """target = x1 - x0（从噪声到数据的直线速度）
loss = ((model_output - target) ** 2).mean()
损失计算本身不需要 `t`。""",
    'description_en': """Implement the flow matching training loss, the generative modeling paradigm behind Stable Diffusion 3, Flux, and Sora.

Flow matching trains a neural network to predict the velocity field that transports noise to data along straight paths. The target velocity at any interpolated point is simply the direction from noise to data.

**Signature:** `flow_matching_loss(model_output, x0, x1, t) -> Tensor`

**Parameters:**
- `model_output` — predicted velocity v_θ(x_t, t), shape (B, D)
- `x0` — noise samples, shape (B, D)
- `x1` — data samples, shape (B, D)
- `t` — timesteps in [0, 1], shape (B,)

**Returns:** scalar MSE loss

**Formula:**
- Target velocity: `u_t = x1 - x0` (straight-line direction from noise to data)
- Loss: `mean(||model_output - u_t||²)`

Note: the interpolated point `x_t = t * x1 + (1 - t) * x0` is computed externally before calling the model; `t` is passed here only for interface completeness.""",
    'description_zh': """实现流匹配训练损失——Stable Diffusion 3、Flux 和 Sora 背后的生成建模范式。

流匹配训练神经网络预测速度场，将噪声沿直线路径传输到数据。在任意插值点处，目标速度就是从噪声指向数据的方向。

**签名:** `flow_matching_loss(model_output, x0, x1, t) -> Tensor`

**参数:**
- `model_output` — 预测速度 v_θ(x_t, t)，形状 (B, D)
- `x0` — 噪声样本，形状 (B, D)
- `x1` — 数据样本，形状 (B, D)
- `t` — [0, 1] 内的时间步，形状 (B,)

**返回:** 标量 MSE 损失

**公式:**
- 目标速度：`u_t = x1 - x0`（从噪声到数据的直线方向）
- 损失：`mean(||model_output - u_t||²)`

注意：插值点 `x_t = t * x1 + (1 - t) * x0` 在调用模型前已在外部计算；`t` 在此仅为接口完整性而传入。""",
    "tests": [
        {
            "name": 'Returns scalar',
            "code": """import torch
torch.manual_seed(0)
B, D = 4, 32
model_output = torch.randn(B, D)
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
loss = {fn}(model_output, x0, x1, t)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'""",
        },
        {
            "name": 'Perfect prediction gives zero loss',
            "code": """import torch
torch.manual_seed(1)
B, D = 8, 16
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
model_output = x1 - x0   # perfect velocity prediction
loss = {fn}(model_output, x0, x1, t)
assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f'Perfect prediction should give 0 loss, got {loss.item()}'""",
        },
        {
            "name": 'Loss is non-negative',
            "code": """import torch
torch.manual_seed(2)
for _ in range(10):
    B, D = 4, 8
    model_output = torch.randn(B, D)
    x0 = torch.randn(B, D)
    x1 = torch.randn(B, D)
    t = torch.rand(B)
    loss = {fn}(model_output, x0, x1, t)
    assert loss.item() >= 0, f'Loss must be non-negative, got {loss.item()}'""",
        },
        {
            "name": 'Gradient flows through model_output',
            "code": """import torch
B, D = 4, 16
model_output = torch.randn(B, D, requires_grad=True)
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
loss = {fn}(model_output, x0, x1, t)
loss.backward()
assert model_output.grad is not None, 'No gradient for model_output'""",
        },
        {
            "name": 'Exact numerical value',
            "code": """import torch
torch.manual_seed(0)
B, D = 4, 8
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
model_output = torch.randn(B, D)
target = x1 - x0
expected = ((model_output - target) ** 2).mean()
out = {fn}(model_output, x0, x1, t)
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected.item():.6f}, got {out.item():.6f}'""",
        },
        {
            "name": 'Loss scales with prediction error',
            "code": """import torch
torch.manual_seed(3)
B, D = 4, 8
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
target = x1 - x0
small_err = target + 0.01 * torch.randn(B, D)
large_err = target + 1.0 * torch.randn(B, D)
loss_small = {fn}(small_err, x0, x1, t)
loss_large = {fn}(large_err, x0, x1, t)
assert loss_small.item() < loss_large.item(), 'Larger prediction error should give larger loss'""",
        },
    ],
    'solution': '''def flow_matching_loss(model_output, x0, x1, t):
    """计算直线路径流匹配损失 (Flow Matching Loss)。"""
    # Step 1: 由噪声端点到数据端点构造恒定目标速度
    target_velocity = x1 - x0

    # Step 2: 计算预测速度与目标速度的逐元素误差
    velocity_error = model_output - target_velocity

    # Step 3: 对批次与特征维统一取均方误差
    loss = velocity_error.square().mean()
    return loss''',
    'demo': """torch.manual_seed(42)
B, D = 16, 8
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
perfect_loss = flow_matching_loss(x1 - x0, x0, x1, t)
random_loss = flow_matching_loss(torch.randn(B, D), x0, x1, t)
print("Perfect prediction loss:", perfect_loss.item())
print("Random prediction loss:", random_loss.item())""",
    'explanation': """### 核心思路
直线条件路径 $x_t=(1-t)x_0+tx_1$ 的速度恒为 $u_t=x_1-x_0$，因此训练目标是模型输出与该速度之间的全元素均方误差。

### 代码步骤拆解
- **Step 1（目标速度）**：对同形状 `(B, D)` 的 `x1` 与 `x0` 相减得到 `target_velocity`。
- **Step 2（速度残差）**：`model_output - target_velocity` 形成每个样本、每个特征的预测误差。
- **Step 3（标量归约）**：平方后对批次和特征维共同取均值，返回零维 `loss`。

### 重点与易错点分析
- **注意点 1（时间变量）**：直线路径的目标速度与 `t` 无关；`t` 仅用于外部构造插值点和模型条件。
- **注意点 2（方向约定）**：本题路径从噪声 `x0` 指向数据 `x1`，目标必须是 `x1 - x0`，不能反向。
- **注意点 3（归约口径）**：`.mean()` 同时平均批次与特征；若先求特征和再平均批次，损失会额外乘 $D$。

### 复杂度分析
- **时间复杂度**：$O(BD)$，主要是逐元素减法、平方与归约。
- **空间复杂度**：$O(BD)$，保存目标速度和误差张量。""",
}
