"""ZeRO memory accounting and pipeline-bubble task."""

TASK = {
    "title": "ZeRO Memory Accounting & 1F1B Pipeline Bubble",
    "title_zh": "ZeRO 显存核算与 1F1B 流水线气泡率",
    "difficulty": "Easy",
    "category": "高效训练",
    "function_name": "distributed_efficiency_stats",
    "starter_code": '''def distributed_efficiency_stats(num_params, world_size, stage=0, num_stages=1, num_microbatches=1):
    pass
''',
    "hint": "Start from 2P parameter bytes, 2P gradient bytes, and 12P optimizer bytes. Divide optimizer, then gradients, then parameters by world size as the ZeRO stage increases.",
    "hint_zh": "从参数 2P、梯度 2P、优化器 12P 字节开始；随着 ZeRO 阶段提高，依次将优化器、梯度、参数按 world size 分片。",
    "description_en": "Report per-rank mixed-precision Adam model-state memory under ZeRO stages 0–3 and the idealized 1F1B pipeline bubble fraction.",
    "description_zh": "报告 ZeRO 0–3 阶段下每个 rank 的混合精度 Adam 模型状态显存，以及理想化 1F1B 流水线气泡率。",
    "theory_en": "ZeRO progressively shards optimizer state, gradients, and parameters. For 1F1B scheduling, pipeline fill and drain cost `stages-1` slots around the useful microbatches.",
    "theory_zh": "ZeRO 依次对优化器状态、梯度和参数分片。对 1F1B 调度，流水线填充与排空相对有效微批次需要 `stages-1` 个时隙。",
    "diagram_en": "```text\nstage 0: P G O replicated\nstage 1: P G | O sharded\nstage 2: P | G O sharded\nstage 3: P G O sharded\nbubble = (S-1)/(M+S-1)\n```",
    "diagram_zh": "```text\n阶段 0: 参数/梯度/优化器均复制\n阶段 1: 仅优化器分片\n阶段 2: 梯度与优化器分片\n阶段 3: 三者均分片\n气泡率 = (S-1)/(M+S-1)\n```",
    "tests": [
        {
            "name": "dictionary shape and float values",
            "code": '''import torch
stats = {fn}(1_000_000, 8, stage=2, num_stages=4, num_microbatches=16)
keys = {"param_bytes", "grad_bytes", "optim_bytes", "total_bytes", "bubble_fraction"}
assert set(stats) == keys
assert all(isinstance(stats[k], float) for k in keys)
''',
        },
        {
            "name": "exact ZeRO stage accounting",
            "code": '''import torch
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
    assert actual == ref and stats["total_bytes"] == sum(ref)
''',
        },
        {
            "name": "pipeline edge cases",
            "code": '''import torch
no_pipe = {fn}(10, 2, num_stages=1, num_microbatches=1)
deep = {fn}(10, 2, num_stages=8, num_microbatches=1)
many = {fn}(10, 2, num_stages=8, num_microbatches=1000)
assert no_pipe["bubble_fraction"] == 0.0
assert deep["bubble_fraction"] == 7.0 / 8.0
assert many["bubble_fraction"] == 7.0 / 1007.0
''',
        },
        {
            "name": "tensor scalar and gradient compatibility",
            "code": '''import torch
P = torch.tensor(1e9, requires_grad=True)
stats = {fn}(P, 4, stage=3, num_stages=2, num_microbatches=8)
probe = P * stats["bubble_fraction"]
probe.backward()
assert stats["total_bytes"] == 16.0e9 / 4.0
assert P.grad is not None and torch.isfinite(P.grad)
''',
        },
    ],
    "solution": '''import torch

def distributed_efficiency_stats(num_params, world_size, stage=0, num_stages=1, num_microbatches=1):
    P = float(num_params.detach()) if isinstance(num_params, torch.Tensor) else float(num_params)
    N = world_size
    if stage == 0:
        param_bytes = 2 * P
        grad_bytes = 2 * P
        optim_bytes = 12 * P
    elif stage == 1:
        param_bytes = 2 * P
        grad_bytes = 2 * P
        optim_bytes = 12 * P / N
    elif stage == 2:
        param_bytes = 2 * P
        grad_bytes = 2 * P / N
        optim_bytes = 12 * P / N
    else:
        param_bytes = 2 * P / N
        grad_bytes = 2 * P / N
        optim_bytes = 12 * P / N
    total_bytes = param_bytes + grad_bytes + optim_bytes
    bubble_fraction = (num_stages - 1) / (num_microbatches + num_stages - 1)
    return {
        "param_bytes": float(param_bytes),
        "grad_bytes": float(grad_bytes),
        "optim_bytes": float(optim_bytes),
        "total_bytes": float(total_bytes),
        "bubble_fraction": float(bubble_fraction),
    }
''',
    "demo": '''stats = distributed_efficiency_stats(
    num_params=70e9, world_size=64, stage=3, num_stages=8, num_microbatches=32
)
print(stats)
''',
    "explanation": "The returned memory is per rank and covers only parameters, gradients, and Adam states. Activation memory and communication buffers are intentionally excluded.",
}
