"""FSDP Training Step task."""

TASK = {
    'title': 'FSDP Training Step',
    'title_zh': 'FSDP 训练步骤',
    'difficulty': 'Hard',
    'category': '高效训练',
    'function_name': 'fsdp_step',
    'starter_code': """def fsdp_step(param_shards, grad_fn, world_size):
    pass""",
    'hint': """1. all-gather: full_param = torch.cat(param_shards)
2. grad = grad_fn(full_param)
3. reduce-scatter: grad_shards = list(grad.chunk(world_size))
4. new_shards[i] = param_shards[i] - 0.01 * grad_shards[i]""",
    'hint_zh': """1. all-gather：full_param = torch.cat(param_shards)
2. grad = grad_fn(full_param)
3. reduce-scatter：grad_shards = list(grad.chunk(world_size))
4. new_shards[i] = param_shards[i] - 0.01 * grad_shards[i]""",
    'description_en': """Implement a simplified FSDP (Fully Sharded Data Parallel) training step.

FSDP shards model parameters across workers to reduce per-device memory. Before the forward pass, each worker all-gathers the full parameters. After the backward pass, gradients are reduce-scattered so each worker only holds its shard's gradient.

**Signature:** `fsdp_step(param_shards, grad_fn, world_size) -> list[Tensor]`

**Parameters:**
- `param_shards` — list of `world_size` parameter shards (each a flat 1D tensor)
- `grad_fn` — callable that takes the full (unsharded) parameter tensor and returns the gradient w.r.t. it
- `world_size` — number of virtual workers

**Returns:** updated list of parameter shards (after one gradient step with lr=0.01)

**Steps:**
1. All-gather: concatenate shards to get full parameter
2. Compute gradient via grad_fn(full_param)
3. Reduce-scatter: split gradient into world_size chunks, each worker keeps its chunk
4. Update each shard: shard -= 0.01 * grad_shard""",
    'description_zh': """实现简化版 FSDP（全分片数据并行）训练步骤。

FSDP 将模型参数分片到各 worker 以减少单设备内存。前向传播前，每个 worker all-gather 完整参数。反向传播后，梯度 reduce-scatter，每个 worker 只保留自己分片的梯度。

**签名:** `fsdp_step(param_shards, grad_fn, world_size) -> list[Tensor]`

**参数:**
- `param_shards` — `world_size` 个参数分片的列表（每个是扁平 1D 张量）
- `grad_fn` — 接收完整（未分片）参数张量，返回对应梯度的可调用对象
- `world_size` — 虚拟 worker 数量

**返回:** 更新后的参数分片列表（一步梯度更新，lr=0.01）

**步骤:**
1. All-gather：拼接分片得到完整参数
2. 通过 grad_fn(full_param) 计算梯度
3. Reduce-scatter：将梯度分成 world_size 块，每个 worker 保留自己的块
4. 更新每个分片：shard -= 0.01 * grad_shard""",
    "tests": [
        {
            "name": 'Returns correct number of shards',
            "code": """import torch
world_size = 4
shard_size = 8
param_shards = [torch.randn(shard_size) for _ in range(world_size)]
def grad_fn(p):
    return torch.ones_like(p)
new_shards = {fn}(param_shards, grad_fn, world_size)
assert len(new_shards) == world_size, f'Expected {world_size} shards, got {len(new_shards)}'
assert all(s.shape == (shard_size,) for s in new_shards), 'Shard shapes should be preserved'""",
        },
        {
            "name": 'Matches equivalent non-sharded SGD update',
            "code": """import torch
torch.manual_seed(0)
world_size = 2
shard_size = 4
param_shards = [torch.randn(shard_size) for _ in range(world_size)]
full_param_ref = torch.cat(param_shards).clone()

# grad_fn: gradient is 2 * full_param (from loss = ||param||^2)
def grad_fn(p):
    return 2.0 * p

new_shards = {fn}(param_shards, grad_fn, world_size)
new_full = torch.cat(new_shards)

# Expected: SGD step on full param
grad_ref = grad_fn(full_param_ref)
expected = full_param_ref - 0.01 * grad_ref
assert torch.allclose(new_full, expected, atol=1e-6), f'FSDP step should match SGD on full param'""",
        },
        {
            "name": 'Each shard gets its own gradient slice',
            "code": """import torch
torch.manual_seed(1)
world_size = 4
shard_size = 4
param_shards = [torch.randn(shard_size) for _ in range(world_size)]
full_param = torch.cat(param_shards)

# Gradient is position-dependent: grad[i] = i
def grad_fn(p):
    return torch.arange(len(p), dtype=p.dtype)

new_shards = {fn}(param_shards, grad_fn, world_size)
full_grad = grad_fn(full_param)
grad_chunks = list(full_grad.chunk(world_size))
for i, (shard, orig, gc) in enumerate(zip(new_shards, param_shards, grad_chunks)):
    expected = orig - 0.01 * gc
    assert torch.allclose(shard, expected, atol=1e-6), f'Shard {i} update incorrect'""",
        },
        {
            "name": 'Works with world_size=1',
            "code": """import torch
torch.manual_seed(2)
param_shards = [torch.randn(8)]
def grad_fn(p):
    return p * 2
new_shards = {fn}(param_shards, grad_fn, world_size=1)
assert len(new_shards) == 1
expected = param_shards[0] - 0.01 * grad_fn(param_shards[0])
assert torch.allclose(new_shards[0], expected, atol=1e-6)""",
        },
    ],
    'solution': '''def fsdp_step(param_shards, grad_fn, world_size):
    """执行简化的全分片数据并行更新 (FSDP Step)。"""
    # Step 1: 聚合参数分片并计算完整梯度
    full_param = torch.cat(param_shards)
    full_grad = grad_fn(full_param)

    # Step 2: 将梯度重新切分到各个虚拟工作进程
    grad_shards = full_grad.chunk(world_size)

    # Step 3: 对每个参数分片执行一次 SGD 更新
    updated_shards = [
        shard - 0.01 * grad_shard
        for shard, grad_shard in zip(param_shards, grad_shards)
    ]
    return updated_shards''',
    'demo': """torch.manual_seed(42)
world_size = 4
shard_size = 8
param_shards = [torch.randn(shard_size) for _ in range(world_size)]
grad_fn = lambda param: 2.0 * param
updated_shards = fsdp_step(param_shards, grad_fn, world_size)
updated_param = torch.cat(updated_shards)
print("Output shape:", updated_param.shape)
print("Number of shards:", len(updated_shards))
print("Sample values:", updated_param[:4])""",
    'explanation': r"""### 核心思路
先用 all-gather 将参数分片拼成完整参数，再计算完整梯度并按 $\text{world\_size}$ 等分；每个分片使用对应梯度执行 $\theta_i \leftarrow \theta_i-0.01g_i$。

### 代码步骤拆解
- **Step 1（参数聚合）**：`torch.cat` 将 `param_shards` 沿一维拼接为 `full_param`。
- **Step 2（梯度分片）**：`grad_fn` 生成 `full_grad`，随后 `chunk(world_size)` 恢复与参数相同的分片布局。
- **Step 3（局部更新）**：`zip` 对齐 `shard` 与 `grad_shard`，得到新的 `updated_shards` 列表。

### 重点与易错点分析
- **注意点 1（分片一致性）**：参数总长度必须能按 `world_size` 产生与输入列表一一对应的梯度块，否则更新会错位。
- **注意点 2（非原地更新）**：表达式创建新张量，不会覆盖调用者持有的原参数分片，便于比较更新前后结果。
- **注意点 3（通信抽象）**：这里用 `cat` 和 `chunk` 模拟 all-gather/reduce-scatter，真实 FSDP 会通过分布式通信避免每个阶段长期保存完整状态。

### 复杂度分析
- **时间复杂度**：$O(P)$，其中 $P$ 是完整参数元素数，主要开销是拼接、梯度计算和逐元素更新。
- **空间复杂度**：$O(P)$，完整参数、完整梯度及更新结果需要线性额外存储。""",
}
