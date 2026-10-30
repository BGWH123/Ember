"""Warmup-Stable-Decay (WSD) LR Scheduler task."""

TASK = {
    'title': 'Warmup-Stable-Decay (WSD) LR Scheduler',
    'title_zh': 'WSD（预热-稳定-衰减）学习率调度',
    'difficulty': 'Easy',
    'category': '优化器与学习率',
    'function_name': 'wsd_lr_schedule',
    'starter_code': """def wsd_lr_schedule(step, warmup_steps, stable_steps, decay_steps, max_lr, min_lr=0.0):
    pass""",
    'hint': """Use three ordered boundary checks: warmup, stable plateau, and cosine decay. Return `min_lr` after the total duration.""",
    'hint_zh': '依次判断预热、稳定平台和余弦衰减三个区间；超过总步数后返回 `min_lr`。',
    'description_en': """Implement a Warmup-Stable-Decay learning-rate schedule: linear warmup, a constant plateau, then cosine decay to a minimum learning rate.""",
    'description_zh': '实现预热-稳定-衰减学习率调度：线性预热、恒定平台，最后余弦衰减至最小学习率。',
    'theory_en': """Warmup limits early optimization shocks, the stable phase performs most training at peak rate, and smooth cosine decay reduces the step size toward convergence.""",
    'theory_zh': '预热减轻训练初期冲击，稳定阶段以峰值学习率完成主要训练，平滑余弦衰减则逐步减小步长以收敛。',
    'diagram_en': """```mermaid
flowchart LR
W[linear warmup] --> S[stable max LR] --> D[cosine decay] --> M[min LR]
```""",
    'diagram_zh': """```mermaid
flowchart LR
W[线性预热] --> S[峰值稳定期] --> D[余弦衰减] --> M[最小学习率]
```""",
    "tests": [
        {
            "name": 'linear warmup',
            "code": """import torch
lr = {fn}(2, 4, 3, 5, 0.01, 0.001)
assert isinstance(lr, float)
assert abs(lr - 0.005) < 1e-12
assert torch.tensor(lr).shape == ()""",
        },
        {
            "name": 'stable boundaries',
            "code": """import torch
a = {fn}(4, 4, 3, 5, 0.01, 0.001)
b = {fn}(6, 4, 3, 5, 0.01, 0.001)
assert torch.tensor([a, b]).shape == (2,)
assert a == 0.01 and b == 0.01""",
        },
        {
            "name": 'cosine exact reference',
            "code": """import torch
import math
lr = {fn}(9, 4, 3, 5, 0.01, 0.001)
progress = (9 - 4 - 3) / 5
expected = 0.001 + 0.5 * 0.009 * (1.0 + math.cos(math.pi * progress))
assert torch.allclose(torch.tensor(lr), torch.tensor(expected), atol=0, rtol=0)""",
        },
        {
            "name": 'end and after end',
            "code": """import torch
end = {fn}(12, 4, 3, 5, 0.01, 0.001)
after = {fn}(99, 4, 3, 5, 0.01, 0.001)
assert torch.tensor([end, after]).shape == (2,)
assert abs(end - 0.001) < 1e-12
assert after == 0.001""",
        },
    ],
    'solution': '''def wsd_lr_schedule(step, warmup_steps, stable_steps, decay_steps, max_lr, min_lr=0.0):
    """实现WSD（预热-稳定-衰减）学习率调度（Warmup-Stable-Decay (WSD) LR Scheduler）。"""
    # Step 1: 在 warmup 区间按已完成步数线性升至 max_lr，随后保持峰值平台
    if step < warmup_steps:
        return float(max_lr * (step / warmup_steps))
    if step < warmup_steps + stable_steps:
        return float(max_lr)
    # Step 2: 将 decay 区间归一化到 [0,1]，用半余弦从 max_lr 平滑降至 min_lr
    total = warmup_steps + stable_steps + decay_steps
    if step <= total:
        progress = (step - warmup_steps - stable_steps) / decay_steps
        return float(min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress)))
    # Step 3: 超过预定训练长度后钳制为 min_lr，避免余弦周期反弹
    return float(min_lr)''',
    'demo': """torch.manual_seed(42)
values = [wsd_lr_schedule(i, 2, 3, 5, 1e-3, 1e-5) for i in range(12)]
print("Schedule values:", values)
print("Peak / final LR:", max(values), values[-1])""",
    'explanation': r"""### 核心思路
WSD 把训练步分为线性预热、峰值平台和半余弦退火。其分段函数为 $\eta(t)=\eta_{\max}t/W$（$t<W$），平台为 $\eta_{\max}$，衰减段为 $\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi u))$，其中 $u=(t-W-S)/D$。

### 代码步骤拆解
- **Step 1（预热与稳定平台）**：`step<warmup_steps` 返回 `max_lr*step/warmup_steps`；接下来的 `stable_steps` 精确保持 `max_lr`。
- **Step 2（余弦退火）**：在 `[W+S,W+S+D]` 内计算 `progress`，从 `cos(0)=1` 连续下降到 `cos(pi)=-1`。
- **Step 3（末端钳制）**：总时长之后直接返回 `min_lr`，避免继续代入 cosine 导致学习率重新上升。

### 重点与易错点分析
- **注意点 1（端点连续性）**：`step=W` 应落入平台并等于 `max_lr`，`step=W+S+D` 应落入衰减端点并等于 `min_lr`；比较符号决定这两个边界。
- **注意点 2（进度分母）**：余弦进度必须除以 `decay_steps` 且先减去 warmup、stable 长度，否则相位偏移，衰减起点不再是峰值。
- **注意点 3（WSD 与纯 cosine）**：WSD 的稳定平台让多数训练以峰值率进行，衰减长度可在训练后期单独决定；调度只返回 Python float，不属于模型梯度图。

### 复杂度分析
- **时间复杂度**：固定次数的比较和一次余弦求值，为 $O(1)$。
- **空间复杂度**：只保存 `total`、`progress` 等标量，为 $O(1)$。""",
}
