"""Tensor Parallel MLP task."""

TASK = {
    'title': 'Tensor Parallel MLP',
    'title_zh': '张量并行 MLP',
    'difficulty': 'Hard',
    'category': '高效训练',
    'function_name': 'TensorParallelMLP',
    'starter_code': """class TensorParallelMLP(nn.Module):
    pass""",
    'hint': """1. W1: column-split → w1_shards[i] shape `(d_model, d_ff/world_size)`
2. W2: row-split → w2_shards[i] shape `(d_ff/world_size, d_model)`
3. for each shard i: h_i = gelu(x @ w1_i) → partial_i = h_i @ w2_i
4. all-reduce = sum(partial_i for all i)""",
    'hint_zh': """1. W1 按列分片 → w1_shards[i] 形状 `(d_model, d_ff/world_size)`
2. W2 按行分片 → w2_shards[i] 形状 `(d_ff/world_size, d_model)`
3. 对每个分片 i：h_i = gelu(x @ w1_i) → partial_i = h_i @ w2_i
4. all-reduce = sum(所有 partial_i)""",
    'description_en': """Implement a Megatron-style tensor parallel MLP.

Tensor parallelism splits the MLP weight matrices across devices. The first (up-projection) layer is column-parallel: each shard computes a slice of the hidden dimension. The second (down-projection) is row-parallel: each shard takes a slice of the input and produces the full output, then results are summed (all-reduce).

**Signature:** `TensorParallelMLP(d_model, d_ff, world_size)` (nn.Module)

**Parameters:**
- `d_model` — input/output dimension
- `d_ff` — hidden dimension (divisible by world_size)
- `world_size` — number of virtual shards

**Forward:** `forward(x) -> Tensor` where x is `(B, d_model)`

**Constraint:** Store sharded weights as `nn.ParameterList`. The forward pass must simulate column-parallel + row-parallel + all-reduce. Result must match a standard `d_model -> d_ff -> d_model` MLP with GELU activation.""",
    'description_zh': """实现 Megatron 风格的张量并行 MLP。

张量并行将 MLP 权重矩阵分片到各设备。第一层（上投影）是列并行：每个分片计算隐藏维度的一个切片。第二层（下投影）是行并行：每个分片接收输入的一个切片并产生完整输出，然后求和（all-reduce）。

**签名:** `TensorParallelMLP(d_model, d_ff, world_size)`（nn.Module）

**参数:**
- `d_model` — 输入/输出维度
- `d_ff` — 隐藏维度（可被 world_size 整除）
- `world_size` — 虚拟分片数

**前向:** `forward(x) -> Tensor`，x 形状 `(B, d_model)`

**约束:** 将分片权重存为 `nn.ParameterList`。前向传播必须模拟列并行 + 行并行 + all-reduce。结果必须与带 GELU 激活的标准 `d_model -> d_ff -> d_model` MLP 一致。""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
import torch.nn as nn
torch.manual_seed(0)
model = {fn}(16, 32, world_size=2)
x = torch.randn(4, 16)
out = model(x)
assert out.shape == (4, 16), f'Expected (4, 16), got {out.shape}'""",
        },
        {
            "name": 'Matches standard MLP numerically',
            "code": """import torch
import torch.nn as nn
torch.manual_seed(0)
d_model, d_ff, world_size = 16, 32, 4
tp_mlp = {fn}(d_model, d_ff, world_size)

# Build equivalent standard MLP by concatenating shards
W1_full = torch.cat([p.data for p in tp_mlp.w1_shards], dim=1)  # (d_model, d_ff)
W2_full = torch.cat([p.data for p in tp_mlp.w2_shards], dim=0)  # (d_ff, d_model)

torch.manual_seed(42)
x = torch.randn(3, d_model)
out_tp = tp_mlp(x)
out_std = torch.nn.functional.gelu(x @ W1_full) @ W2_full
assert torch.allclose(out_tp, out_std, atol=1e-5), f'TP MLP should match standard MLP'""",
        },
        {
            "name": 'world_size=1 works',
            "code": """import torch
torch.manual_seed(0)
model = {fn}(8, 16, world_size=1)
x = torch.randn(2, 8)
out = model(x)
assert out.shape == (2, 8)""",
        },
        {
            "name": 'Gradient flows through all shards',
            "code": """import torch
torch.manual_seed(0)
model = {fn}(8, 16, world_size=2)
x = torch.randn(2, 8, requires_grad=True)
out = model(x)
out.sum().backward()
assert x.grad is not None, 'No gradient for input'
for p in model.parameters():
    assert p.grad is not None, f'No gradient for parameter {p.shape}'""",
        },
    ],
    'solution': '''class TensorParallelMLP(nn.Module):
    """实现张量并行 MLP（Tensor Parallel MLP）。"""

    def __init__(self, d_model, d_ff, world_size):
        # Step 1: 将 d_ff 均分为 R 份，成对建立 W1 的列分片与 W2 的行分片
        super().__init__()
        self.world_size = world_size
        shard_size = d_ff // world_size
        self.w1_shards = nn.ParameterList([nn.Parameter(torch.randn(d_model, shard_size) * (2 / d_model) ** 0.5) for _ in range(world_size)])
        self.w2_shards = nn.ParameterList([nn.Parameter(torch.randn(shard_size, d_model) * (2 / d_ff) ** 0.5) for _ in range(world_size)])

    def forward(self, x):
        # Step 2: 各虚拟 rank 独立完成局部上投影、精确 GELU 与局部下投影
        output = None
        for w1, w2 in zip(self.w1_shards, self.w2_shards):
            z = x @ w1
            h = z * 0.5 * (1.0 + torch.erf(z / 2.0 ** 0.5))
            partial = h @ w2
            output = partial if output is None else output + partial
        # Step 3: 对所有 (B, d_model) 局部结果求和，模拟行并行层的 all-reduce
        return output''',
    'demo': """torch.manual_seed(42)
d_model, d_ff, world_size = (64, 256, 4)
x = torch.randn(2, d_model)
tp_mlp = TensorParallelMLP(d_model, d_ff, world_size)
out = tp_mlp(x)
print('Output shape:', out.shape)
w1_full = torch.cat([p.data for p in tp_mlp.w1_shards], dim=1)
w2_full = torch.cat([p.data for p in tp_mlp.w2_shards], dim=0)
ref = torch.nn.functional.gelu(x @ w1_full) @ w2_full
print('Max diff vs reference:', (out - ref).abs().max().item())""",
    'explanation': r"""### 核心思路
令 $R=\text{world\_size}$，完整权重按 $W_1=[W_{1,1},\ldots,W_{1,R}]$ 列切、按 $W_2=[W_{2,1}^{\top},\ldots,W_{2,R}^{\top}]^{\top}$ 行切。因 GELU 对隐藏单元逐元素作用，完整 MLP 可严格写成 $Y=\sum_{r=1}^{R}\operatorname{GELU}(XW_{1,r})W_{2,r}$；最后的求和正是 Megatron 行并行层的 all-reduce。

### 代码步骤拆解
- **Step 1（互补权重分片）**：`shard_size=d_ff/world_size`，`w1_shards[r]` 为 `(d_model, shard_size)`，对应的 `w2_shards[r]` 为 `(shard_size, d_model)`。
- **Step 2（rank 内局部 MLP）**：对 `(B,d_model)` 的 `x` 计算 `z=x@w1`、精确 erf-GELU，再得到 `(B,d_model)` 的 `partial=h@w2`。
- **Step 3（模拟 all-reduce）**：逐项累加所有 `partial`，恢复与拼接完整权重后标准 MLP 完全一致的 `(B,d_model)` 输出。

### 重点与易错点分析
- **注意点 1（切分轴必须配对）**：第一层必须沿输出维列切，第二层必须沿输入维行切；若两层都列切，就无法在不拼接激活的情况下求和恢复结果。
- **注意点 2（非线性的位置）**：GELU 要在每个隐藏分片内部、下投影之前执行；不能先合并线性结果再激活，因为 $\phi(\sum x_r)\ne\sum\phi(x_r)$。
- **注意点 3（通信与显存）**：每个真实 rank 只保存约 $1/R$ 的两层权重与隐藏激活，但下投影产生完整输出并需一次 all-reduce；计算量每 rank 约为完整 MLP 的 $1/R$。

### 复杂度分析
- **时间复杂度**：模拟全部 $R$ 个分片总计 $O(B\,d_{\text{model}}\,d_{\text{ff}})$；真实单 rank 为其 $1/R$，另有 all-reduce 通信。
- **空间复杂度**：模型参数总量为 $O(d_{\text{model}}d_{\text{ff}})$；单 rank 隐藏激活为 $O(Bd_{\text{ff}}/R)$，并保留 $O(Bd_{\text{model}})$ 的局部输出。""",
}
