"""Cosine LR Scheduler with Warmup task."""

TASK = {
    'title': 'Cosine LR Scheduler with Warmup',
    'title_zh': '余弦学习率调度（含预热）',
    'difficulty': 'Medium',
    'category': '优化器与学习率',
    'function_name': 'cosine_lr_schedule',
    'starter_code': """def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    pass""",
    'hint': """1. step < warmup_steps → lr = max_lr * step / warmup_steps
2. step >= total_steps → lr = min_lr
3. progress = (step - warmup_steps) / (total_steps - warmup_steps)
   → min_lr + 0.5*(max_lr-min_lr)*(1 + cos(π·progress))""",
    'hint_zh': """1. step < warmup_steps → lr = max_lr * step / warmup_steps
2. step >= total_steps → lr = min_lr
3. progress = (step - warmup_steps) / (total_steps - warmup_steps)
   → min_lr + 0.5*(max_lr-min_lr)*(1 + cos(π·progress))""",
    'description_en': """Implement a cosine learning rate schedule with linear warmup.

This scheduler linearly ramps the LR during warmup, then decays it following a cosine curve. Widely used in transformer training.

**Signature:** `cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0) -> float`

**Parameters:**
- `step` — current training step
- `total_steps` — total number of steps
- `warmup_steps` — number of warmup steps
- `max_lr`, `min_lr` — peak and minimum learning rates

**Returns:** learning rate as a float

**Constraints:**
- Warmup: linear from 0 to max_lr
- Decay: `min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))`""",
    'description_zh': """实现带线性预热的余弦学习率调度。

该调度器在预热阶段线性增加学习率，之后按余弦曲线衰减，广泛用于 Transformer 训练。

**签名:** `cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0) -> float`

**参数:**
- `step` — 当前训练步数
- `total_steps` — 总步数
- `warmup_steps` — 预热步数
- `max_lr`, `min_lr` — 峰值和最小学习率

**返回:** 浮点数学习率

**约束:**
- 预热阶段：从 0 线性增加到 max_lr
- 衰减阶段：`min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))`""",
    "tests": [
        {
            "name": 'Start of warmup',
            "code": """lr = {fn}(step=0, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)
assert abs(lr) < 1e-8, f'lr at step 0: {lr}'""",
        },
        {
            "name": 'End of warmup',
            "code": """lr = {fn}(step=10, total_steps=100, warmup_steps=10, max_lr=0.001)
assert abs(lr - 0.001) < 1e-8, f'lr at warmup end: {lr}'""",
        },
        {
            "name": 'End of schedule',
            "code": """lr = {fn}(step=100, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0001)
assert abs(lr - 0.0001) < 1e-6, f'lr at end: {lr}'""",
        },
        {
            "name": 'Warmup is monotonically increasing',
            "code": """lrs = [{fn}(step=i, total_steps=100, warmup_steps=10, max_lr=0.001) for i in range(11)]
for i in range(len(lrs) - 1):
    assert lrs[i] <= lrs[i+1] + 1e-10, f'Not increasing at step {i}'""",
        },
        {
            "name": 'Cosine shape',
            "code": """import math
lr = {fn}(step=55, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)
progress = (55 - 10) / (100 - 10)
expected = 0.5 * 0.001 * (1 + math.cos(math.pi * progress))
assert abs(lr - expected) < 1e-8, f'{lr} vs {expected}'""",
        },
    ],
    'solution': '''def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    """预热余弦学习率调度 (Warmup Cosine Schedule)。"""
    # Step 1: 预热阶段从零线性增长到最大学习率
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # Step 2: 训练结束后固定为最小学习率
    if step >= total_steps:
        return min_lr

    # Step 3: 在剩余区间执行半周期余弦衰减
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_factor''',
    'demo': """torch.manual_seed(42)
learning_rates = [
    cosine_lr_schedule(step, 100, 10, 1e-3, 1e-5)
    for step in (0, 10, 55, 100)
]
print("Output shape:", len(learning_rates))
print("Learning rates:", learning_rates)
print("Final value:", learning_rates[-1])""",
    'explanation': r"""### 核心思路
前 `warmup_steps` 按线性函数升温，随后用半周期余弦从 `max_lr` 平滑衰减到 `min_lr`。

### 代码步骤拆解
- **Step 1（线性预热）**：`step / warmup_steps` 将区间映射到 $[0,1)$。
- **Step 2（结束边界）**：当 `step >= total_steps` 时直接返回 `min_lr`。
- **Step 3（余弦衰减）**：将剩余进度映射到 $0.5(1+\cos(\pi p))$。

### 重点与易错点分析
- **注意点 1（分段连续性）**：`step == warmup_steps` 时余弦因子为 1，恰好得到 `max_lr`。
- **注意点 2（结束截断）**：显式处理超出总步数的输入，防止余弦继续回升。
- **注意点 3（零预热）**：`warmup_steps=0` 时首个分支不会执行，可安全直接进入余弦阶段。

### 复杂度分析
- **时间复杂度**：$O(1)$，每次查询只执行常数次标量运算。
- **空间复杂度**：$O(1)$，不保存调度历史。""",
}
