"""ZeRO Memory Accounting & 1F1B Pipeline Bubble task."""

TASK = {
    'title': 'ZeRO Memory Accounting & 1F1B Pipeline Bubble',
    'title_zh': 'ZeRO 显存核算与 1F1B 流水线气泡率',
    'difficulty': 'Easy',
    'category': '高效训练',
    'function_name': 'distributed_efficiency_stats',
    'starter_code': """def distributed_efficiency_stats(num_params, world_size, stage=0, num_stages=1, num_microbatches=1):
    pass""",
    'hint': """Start from 2P parameter bytes, 2P gradient bytes, and 12P optimizer bytes. Divide optimizer, then gradients, then parameters by world size as the ZeRO stage increases.""",
    'hint_zh': '从参数 2P、梯度 2P、优化器 12P 字节开始；随着 ZeRO 阶段提高，依次将优化器、梯度、参数按 world size 分片。',
    'description_en': """Report per-rank mixed-precision Adam model-state memory under ZeRO stages 0–3 and the idealized 1F1B pipeline bubble fraction.""",
    'description_zh': '报告 ZeRO 0–3 阶段下每个 rank 的混合精度 Adam 模型状态显存，以及理想化 1F1B 流水线气泡率。',
    'theory_en': """ZeRO progressively shards optimizer state, gradients, and parameters. For 1F1B scheduling, pipeline fill and drain cost `stages-1` slots around the useful microbatches.""",
    'theory_zh': 'ZeRO 依次对优化器状态、梯度和参数分片。对 1F1B 调度，流水线填充与排空相对有效微批次需要 `stages-1` 个时隙。',
    'diagram_en': """```text
stage 0: P G O replicated
stage 1: P G | O sharded
stage 2: P | G O sharded
stage 3: P G O sharded
bubble = (S-1)/(M+S-1)
```""",
    'diagram_zh': """```text
阶段 0: 参数/梯度/优化器均复制
阶段 1: 仅优化器分片
阶段 2: 梯度与优化器分片
阶段 3: 三者均分片
气泡率 = (S-1)/(M+S-1)
```""",
    "tests": [
        {
            "name": 'dictionary shape and float values',
            "code": """import torch
stats = {fn}(1_000_000, 8, stage=2, num_stages=4, num_microbatches=16)
keys = {"param_bytes", "grad_bytes", "optim_bytes", "total_bytes", "bubble_fraction"}
assert set(stats) == keys
assert all(isinstance(stats[k], float) for k in keys)""",
        },
        {
            "name": 'exact ZeRO stage accounting',
            "code": """import torch
P, N = 120.0, 4
expected = {
    0: (240.0, 240.0, 1440.0),
    1: (240.0, 240.0, 360.0),
    2: (240.0, 60.0, 360.0),
    3: (60.0, 60.0, 360.0),
}
for stage, ref in expected.items():
    stats = {fn}(P, N, stage=stage)
    actual = (stats["param_bytes"], stats["grad_bytes"], stats["optim_bytes"])
    assert actual == ref and stats["total_bytes"] == sum(ref)""",
        },
        {
            "name": 'pipeline edge cases',
            "code": """import torch
no_pipe = {fn}(10, 2, num_stages=1, num_microbatches=1)
deep = {fn}(10, 2, num_stages=8, num_microbatches=1)
many = {fn}(10, 2, num_stages=8, num_microbatches=1000)
assert no_pipe["bubble_fraction"] == 0.0
assert deep["bubble_fraction"] == 7.0 / 8.0
assert many["bubble_fraction"] == 7.0 / 1007.0""",
        },
        {
            "name": 'tensor scalar and gradient compatibility',
            "code": """import torch
P = torch.tensor(1e9, requires_grad=True)
stats = {fn}(P, 4, stage=3, num_stages=2, num_microbatches=8)
probe = P * stats["bubble_fraction"]
probe.backward()
assert stats["total_bytes"] == 16.0e9 / 4.0
assert P.grad is not None and torch.isfinite(P.grad)""",
        },
    ],
    'solution': '''def distributed_efficiency_stats(num_params, world_size, stage=0, num_stages=1, num_microbatches=1):
    """实现ZeRO 显存核算与 1F1B 流水线气泡率（ZeRO Memory Accounting & 1F1B Pipeline Bubble）。"""
    # Step 1: 从 FP16 参数/梯度与混合精度 Adam 状态基线按 ZeRO stage 逐项分片
    param_count = float(num_params.detach()) if isinstance(num_params, torch.Tensor) else float(num_params)
    if stage == 0:
        param_bytes = 2 * param_count
        grad_bytes = 2 * param_count
        optim_bytes = 12 * param_count
    elif stage == 1:
        param_bytes = 2 * param_count
        grad_bytes = 2 * param_count
        optim_bytes = 12 * param_count / world_size
    elif stage == 2:
        param_bytes = 2 * param_count
        grad_bytes = 2 * param_count / world_size
        optim_bytes = 12 * param_count / world_size
    else:
        param_bytes = 2 * param_count / world_size
        grad_bytes = 2 * param_count / world_size
        optim_bytes = 12 * param_count / world_size
    # Step 2: 汇总单 rank 模型状态字节，并由填充排空时隙计算 1F1B 气泡占比
    total_bytes = param_bytes + grad_bytes + optim_bytes
    bubble_fraction = (num_stages - 1) / (num_microbatches + num_stages - 1)
    # Step 3: 将三类显存、总量与气泡率统一返回为 Python 浮点统计
    return {'param_bytes': float(param_bytes), 'grad_bytes': float(grad_bytes), 'optim_bytes': float(optim_bytes), 'total_bytes': float(total_bytes), 'bubble_fraction': float(bubble_fraction)}''',
    'demo': """torch.manual_seed(42)
stats = distributed_efficiency_stats(70e9, world_size=64, stage=3, num_stages=8, num_microbatches=32)
print("Total bytes per rank:", stats["total_bytes"])
print("Pipeline bubble fraction:", stats["bubble_fraction"])""",
    'explanation': """### 核心思路
设参数量为 $P$、数据并行规模为 $N$。混合精度 Adam 的模型状态基线是 FP16 参数 $2P$ 字节、FP16 梯度 $2P$、以及 FP32 master 参数与两个 Adam 矩共 $12P$。ZeRO-1/2/3 依次分片优化器、梯度、参数；理想 1F1B 的填充/排空损失为 $S-1$ 个时隙，气泡率为 $(S-1)/(M+S-1)$。

### 代码步骤拆解
- **Step 1（按 stage 核算状态）**：stage 0 全复制；stage 1 仅 `optim_bytes/ N`；stage 2 再令 `grad_bytes/N`；stage 3 连 `param_bytes` 也除以 `world_size`。
- **Step 2（模型状态与流水效率）**：三类字节相加得每 rank `total_bytes`，再用 `num_stages` 和 `num_microbatches` 计算独立的 `bubble_fraction`。
- **Step 3（标量统计输出）**：所有字段转成 Python float，字典同时保留分项显存，便于辨认当前 stage 的节省来源。

### 重点与易错点分析
- **注意点 1（12P 的组成）**：不能把 Adam 优化器只按两个矩的 8P 计算；混合精度训练通常还保存 4P 的 FP32 master 权重，因此合计 12P。
- **注意点 2（ZeRO 并非均匀全除）**：stage 1 的参数和梯度仍复制，stage 2 的参数仍复制；只有 stage 3 才使全部 16P 模型状态理想化为每 rank $16P/N$。
- **注意点 3（气泡与 microbatch）**：当 `S=1` 气泡为 0；固定流水深度时增大 $M$ 可摊薄填充排空成本，但本公式不计通信、负载不均与激活显存。

### 复杂度分析
- **时间复杂度**：仅进行固定次数的标量算术和分支，为 $O(1)$。
- **空间复杂度**：返回五个标量字段，不随 $P$、$N$ 或流水深度增长，为 $O(1)$。""",
}
