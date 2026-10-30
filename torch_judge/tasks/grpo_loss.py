"""GRPO (Group Relative Policy Optimization) Loss task."""

TASK = {
    'title': 'GRPO (Group Relative Policy Optimization) Loss',
    'title_zh': 'GRPO 损失',
    'difficulty': 'Hard',
    'category': '损失函数',
    'function_name': 'grpo_loss',
    'starter_code': """def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    pass""",
    'hint': """1. for each group g: A_i = (r_i - mean_g) / (std_g + eps)
2. advantages = A.detach()  (no grad through rewards)
3. loss = -(advantages * logps).mean()""",
    'hint_zh': """1. 对每个组 g：A_i = (r_i - mean_g) / (std_g + eps)
2. advantages = A.detach()（梯度不流经奖励）
3. loss = -(advantages * logps).mean()""",
    'description_en': """Implement the GRPO (Group Relative Policy Optimization) loss.

GRPO normalizes rewards within each prompt group to compute advantages, then optimizes the policy using these group-relative advantages.

**Signature:** `grpo_loss(logps, rewards, group_ids, eps=1e-5) -> Tensor`

**Parameters:**
- `logps` — policy log-probabilities (B,)
- `rewards` — scalar rewards (B,)
- `group_ids` — integer group identifiers (B,)
- `eps` — epsilon for numerical stability

**Returns:** scalar loss

**Constraints:**
- Per-group z-score normalization: `A_i = (r_i - mean_g) / (std_g + eps)`
- Advantages must be detached (gradients flow only through logps)
- Loss = `-mean(A_i * logps_i)`""",
    'description_zh': """实现 GRPO（组相对策略优化）损失。

GRPO 在每个提示组内归一化奖励以计算优势值，然后使用这些组相对优势优化策略。

**签名:** `grpo_loss(logps, rewards, group_ids, eps=1e-5) -> Tensor`

**参数:**
- `logps` — 策略对数概率 (B,)
- `rewards` — 标量奖励 (B,)
- `group_ids` — 整数组标识符 (B,)
- `eps` — 数值稳定性的 epsilon

**返回:** 标量损失

**约束:**
- 组内 z-score 归一化：`A_i = (r_i - mean_g) / (std_g + eps)`
- 优势值必须 detach（梯度仅通过 logps 流动）
- 损失 = `-mean(A_i * logps_i)`""",
    "tests": [
        {
            "name": 'Basic shape & type',
            "code": """import torch
from torch import Tensor
logps = torch.randn(6, requires_grad=True)
rewards = torch.randn(6)
group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
loss = {fn}(logps, rewards, group_ids)
assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be scalar Tensor'""",
        },
        {
            "name": 'Numeric check vs reference',
            "code": """import torch
from torch import Tensor

def _reference_grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor, eps: float = 1e-5) -> Tensor:
    # Same semantics as the reference solution: per-group z-score then -E[A_i * logp_i].
    logps = logps.view(-1)
    rewards = rewards.view(-1)
    group_ids = group_ids.view(-1)
    unique_ids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    for gid in unique_ids:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = r_g.mean()
        std_g = r_g.std(unbiased=False)
        advantages[mask] = (r_g - mean_g) / (std_g + eps)
    advantages_detached = advantages.detach()
    return -(advantages_detached * logps).mean()

logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
loss_student = {fn}(logps, rewards, group_ids)
loss_ref = _reference_grpo_loss(logps, rewards, group_ids)
assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), 'Loss should match reference implementation numerically on a fixed example'""",
        },
        {
            "name": 'Gradient flows to logps only',
            "code": """import torch
logps = torch.randn(4, requires_grad=True)
rewards = torch.randn(4, requires_grad=True)
group_ids = torch.tensor([0, 0, 1, 1])
loss = {fn}(logps, rewards, group_ids)
loss.backward()
assert logps.grad is not None and rewards.grad is None, 'Gradients should flow only through logps'""",
        },
        {
            "name": 'Group-wise normalization',
            "code": """import torch
logps = torch.zeros(4, requires_grad=True)
rewards = torch.tensor([0.0, 1.0, 10.0, 11.0])
group_ids = torch.tensor([0, 0, 1, 1])
loss = {fn}(logps, rewards, group_ids)
loss.backward()
# Since each group has rewards [0,1] and [10,11], the normalized advantages
# should be identical across groups, leading to identical gradients per position.
assert torch.allclose(logps.grad[:2], logps.grad[2:]), 'Groups should be treated independently but symmetrically'""",
        },
    ],
    'solution': '''def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    """计算组相对策略优化损失 (GRPO Loss)。"""
    # Step 1: 在每个提示组内标准化奖励
    advantages = torch.empty_like(rewards)
    for group_id in group_ids.unique():
        mask = group_ids == group_id
        group_rewards = rewards[mask]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False)
        advantages[mask] = (group_rewards - group_mean) / (group_std + eps)

    # Step 2: 阻断奖励优势分支的梯度
    advantages = advantages.detach()

    # Step 3: 计算策略梯度代理目标
    return -(advantages * logps).mean()''',
    'demo': """torch.manual_seed(42)
logps = torch.randn(6, requires_grad=True)
rewards = torch.tensor([1.0, 0.5, 0.0, 2.0, 1.5, 1.0])
group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
loss = grpo_loss(logps, rewards, group_ids)
print("Output shape:", loss.shape)
print("Loss value:", loss.item())
print("Number of groups:", group_ids.unique().numel())""",
    'explanation': r"""### 核心思路
对每组奖励计算 $A_i=(r_i-\mu_g)/(\sigma_g+\epsilon)$，分离优势梯度后优化 $-\mathbb{E}[A_i\log\pi_i]$。

### 代码步骤拆解
- **Step 1（组内标准化）**：按 `group_ids` 写入与 `rewards` 同形的 `advantages`。
- **Step 2（优势停梯度）**：`.detach()` 保证奖励仅作为训练信号。
- **Step 3（策略目标）**：与 `logps` 逐元素相乘并取负均值，得到标量。

### 重点与易错点分析
- **注意点 1（总体标准差）**：使用 `unbiased=False`，单样本组标准差才为 0 而非 NaN。
- **注意点 2（eps 位置）**：`eps` 加在标准差之后，遵循题目指定公式。
- **注意点 3（梯度边界）**：奖励可要求梯度，但最终优势必须分离，使梯度只流向 `logps`。

### 复杂度分析
- **时间复杂度**：$O(BG)$ 的直接掩码实现，$G$ 为组数。
- **空间复杂度**：$O(B)$，主要保存优势与布尔掩码。""",
}
