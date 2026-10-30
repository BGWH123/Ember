"""Activation Checkpointing task."""

TASK = {
    'title': 'Activation Checkpointing',
    'title_zh': '激活检查点',
    'difficulty': 'Medium',
    'category': '高效训练',
    'function_name': 'checkpoint_sequential',
    'starter_code': """def checkpoint_sequential(fns, x):
    pass""",
    'hint': """import torch.utils.checkpoint as cp
Loop over fns: x = cp.checkpoint(fn, x, use_reentrant=False)
Activations are recomputed on backward instead of stored.""",
    'hint_zh': """import torch.utils.checkpoint as cp
遍历 fns：x = cp.checkpoint(fn, x, use_reentrant=False)
激活在反向传播时重新计算，而非存储。""",
    'description_en': """Implement gradient checkpointing for a sequence of functions.

Activation checkpointing trades compute for memory: instead of storing all intermediate activations during the forward pass, they are recomputed on-the-fly during backpropagation.

**Signature:** `checkpoint_sequential(fns, x) -> Tensor`

**Parameters:**
- `fns` — list of callables, each mapping Tensor -> Tensor
- `x` — input tensor

**Returns:** output tensor after applying all functions sequentially

**Constraints:**
- Intermediate activations must NOT be stored during the forward pass
- Gradients must flow back to `x`
- Output must be numerically identical to naive sequential application
- Use `torch.utils.checkpoint.checkpoint` (allowed — it is not `F.*` or `nn.functional.*`)""",
    'description_zh': """为一系列函数实现梯度检查点。

激活检查点以计算换内存：在前向传播期间不存储所有中间激活，而是在反向传播时按需重新计算。

**签名:** `checkpoint_sequential(fns, x) -> Tensor`

**参数:**
- `fns` — 可调用对象列表，每个接受 Tensor 并返回 Tensor
- `x` — 输入张量

**返回:** 依次应用所有函数后的输出张量

**约束:**
- 前向传播期间不得存储中间激活
- 梯度必须能流回 `x`
- 输出必须与朴素顺序应用在数值上完全一致
- 使用 `torch.utils.checkpoint.checkpoint`（允许——它不是 `F.*` 或 `nn.functional.*`）""",
    "tests": [
        {
            "name": 'Output matches naive sequential',
            "code": """import torch, torch.nn as nn
torch.manual_seed(42)
fns = [nn.Linear(16, 16) for _ in range(4)]
for fn in fns:
    fn.eval()
x = torch.randn(2, 16)
expected = x
for fn in fns:
    expected = fn(expected)
result = {fn}(fns, x)
assert result.shape == expected.shape, f'Shape mismatch: {result.shape}'
assert torch.allclose(result, expected, atol=1e-5), 'Output does not match naive sequential'""",
        },
        {
            "name": 'Gradient flows to input',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
fns = [nn.Linear(8, 8) for _ in range(3)]
x = torch.randn(1, 8, requires_grad=True)
out = {fn}(fns, x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None — gradient did not flow to input'
assert x.grad.shape == x.shape, f'x.grad shape mismatch: {x.grad.shape}'""",
        },
        {
            "name": 'Works with 4 linear layers',
            "code": """import torch, torch.nn as nn
torch.manual_seed(7)
fns = [nn.Linear(32, 32) for _ in range(4)]
x = torch.randn(4, 32, requires_grad=True)
out = {fn}(fns, x)
out.sum().backward()
assert out.shape == (4, 32), f'Output shape: {out.shape}'
assert x.grad is not None, 'No gradient on input'""",
        },
        {
            "name": 'Numerically identical to non-checkpointed version',
            "code": """import torch, torch.nn as nn
torch.manual_seed(3)
fns = [nn.Linear(16, 16) for _ in range(3)]
for fn in fns:
    fn.eval()
x = torch.randn(2, 16)
naive = x.clone()
for fn in fns:
    naive = fn(naive)
ckpt = {fn}(fns, x)
assert torch.allclose(naive, ckpt, atol=1e-5), f'Max diff: {(naive - ckpt).abs().max().item()}'""",
        },
    ],
    'solution': '''def checkpoint_sequential(fns, x):
    """顺序激活检查点 (Activation Checkpointing)。"""
    # Step 1: 逐层建立检查点，前向阶段不保存中间激活
    for fn in fns:
        x = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)

    # Step 2: 返回最终输出，反向传播时按需重算各层
    return x''',
    'demo': """torch.manual_seed(42)
layers = [nn.Linear(16, 16) for _ in range(4)]
x = torch.randn(4, 16, requires_grad=True)
out = checkpoint_sequential(layers, x)
out.sum().backward()
print("Output shape:", out.shape)
print("Gradient shape:", x.grad.shape)""",
    'explanation': """### 核心思路
激活检查点在前向阶段只保留检查点输入，反向阶段重新执行函数，以额外计算换取更低的激活显存。

### 代码步骤拆解
- **Step 1（顺序检查点）**：遍历 `fns`，将当前 `x` 交给 `checkpoint` 并得到下一层输入。
- **Step 2（延迟重算）**：`use_reentrant=False` 保持现代自动微分语义，反向时重建中间激活。
- **Step 3（返回结果）**：最终 `x` 与普通顺序前向数值一致，梯度仍可回传到原输入。

### 重点与易错点分析
- **注意点 1（随机算子）**：检查点会重算前向，框架需保存随机数状态才能保证含 dropout 的结果一致。
- **注意点 2（计算显存折中）**：显存下降的代价是反向阶段增加一次局部前向计算。
- **注意点 3（空函数列表）**：当 `fns` 为空时循环不执行，应直接返回输入 `x`。

### 复杂度分析
- **时间复杂度**：前向为 $O(C)$，训练时重算使总计算约增加到 $O(2C)$。
- **空间复杂度**：$O(M_{checkpoint})$，不再同时保存全部中间激活。""",
}
