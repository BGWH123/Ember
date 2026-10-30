"""Gradient Accumulation task."""

TASK = {
    'title': 'Gradient Accumulation',
    'title_zh': '梯度累积',
    'difficulty': 'Easy',
    'category': '优化器与学习率',
    'function_name': 'accumulated_step',
    'starter_code': """def accumulated_step(model, optimizer, loss_fn, micro_batches):
    pass""",
    'hint': """1. optimizer.zero_grad() once
2. for each micro-batch: forward → loss/n → backward
3. optimizer.step()
   Dividing by n ensures accumulated grads match a single full-batch update.""",
    'hint_zh': """1. optimizer.zero_grad() 一次
2. 对每个微批次：前向 → loss/n → 反向
3. optimizer.step()
   除以 n 确保累积梯度与单次全批次更新一致。""",
    'description_en': """Implement gradient accumulation over micro-batches.

Gradient accumulation simulates a large batch by accumulating gradients from multiple smaller batches before a single optimizer step.

**Signature:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`

**Parameters:**
- `model` — nn.Module
- `optimizer` — torch optimizer
- `loss_fn` — loss function
- `micro_batches` — list of (x, y) tuples

**Returns:** total loss as a float

**Constraints:**
- Scale each micro-batch loss by `1/n` before backward
- Must match a single full-batch update numerically""",
    'description_zh': """实现微批次梯度累积。

梯度累积通过在多个小批次上累积梯度后执行一次优化器步骤，模拟大批次训练。

**签名:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`

**参数:**
- `model` — nn.Module
- `optimizer` — torch 优化器
- `loss_fn` — 损失函数
- `micro_batches` — (x, y) 元组列表

**返回:** 总损失（浮点数）

**约束:**
- 每个微批次损失在反向传播前除以 `n`
- 必须与单次全批次更新数值一致""",
    "tests": [
        {
            "name": 'Matches full batch update',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(4, 2, bias=False)
model_ref = nn.Linear(4, 2, bias=False)
model_ref.load_state_dict(model.state_dict())
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)
x1, y1 = torch.randn(2, 4), torch.randn(2, 2)
x2, y2 = torch.randn(2, 4), torch.randn(2, 2)
{fn}(model, opt, loss_fn, [(x1, y1), (x2, y2)])
opt_ref.zero_grad()
loss_ref = loss_fn(model_ref(torch.cat([x1, x2])), torch.cat([y1, y2]))
loss_ref.backward()
opt_ref.step()
assert torch.allclose(model.weight.data, model_ref.weight.data, atol=1e-5), 'Must match full batch'""",
        },
        {
            "name": 'Returns loss value',
            "code": """import torch, torch.nn as nn
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = {fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])
assert isinstance(loss, float), f'Should return float, got {type(loss)}'
assert loss > 0, 'Loss should be positive'""",
        },
        {
            "name": 'Parameters actually update',
            "code": """import torch, torch.nn as nn
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
w_before = model.weight.data.clone()
{fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])
assert not torch.equal(model.weight.data, w_before), 'Should change'""",
        },
    ],
    'solution': '''def accumulated_step(model, optimizer, loss_fn, micro_batches):
    """执行微批次梯度累积 (Gradient Accumulation)。"""
    # Step 1: 仅在累积开始前清空梯度
    optimizer.zero_grad()
    num_batches = len(micro_batches)
    total_loss = 0.0

    # Step 2: 缩放各微批次损失并累积梯度
    for x, y in micro_batches:
        scaled_loss = loss_fn(model(x), y) / num_batches
        scaled_loss.backward()
        total_loss += scaled_loss.item()

    # Step 3: 所有微批次完成后统一更新参数
    optimizer.step()
    return total_loss''',
    'demo': """torch.manual_seed(42)
model = nn.Linear(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
micro_batches = [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)]
loss = accumulated_step(model, optimizer, nn.MSELoss(), micro_batches)
print("Output shape:", model.weight.shape)
print("Accumulated loss:", loss)
print("Weight norm:", model.weight.norm().item())""",
    'explanation': """### 核心思路
将每个微批次损失除以微批次数 $n$ 后反向传播，使累积梯度等价于等大小微批次拼接后的平均损失梯度，并只执行一次优化器更新。

### 代码步骤拆解
- **Step 1（梯度清零）**：`zero_grad` 只在循环前调用，避免擦除已累积梯度。
- **Step 2（反向累积）**：每个 `scaled_loss` 贡献原梯度的 $1/n$。
- **Step 3（参数更新）**：循环结束后调用一次 `optimizer.step()` 并返回平均损失和。

### 重点与易错点分析
- **注意点 1（缩放位置）**：必须在 `.backward()` 前除以 `num_batches`，仅缩放返回值不会修正梯度。
- **注意点 2（批次权重）**：该等价性假设各微批次大小相同且 `loss_fn` 按样本取均值。
- **注意点 3（空输入）**：`micro_batches` 不应为空，否则分母为零且没有可执行的更新。

### 复杂度分析
- **时间复杂度**：$O(nC)$，其中 $C$ 是一次微批次前向与反向的成本。
- **空间复杂度**：$O(A+P)$，仅保存一个微批次激活与参数梯度。""",
}
