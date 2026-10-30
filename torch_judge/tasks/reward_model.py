"""Bradley-Terry Reward Model Loss task."""

TASK = {
    'title': 'Bradley-Terry Reward Model Loss',
    'title_zh': 'Bradley-Terry 奖励模型',
    'difficulty': 'Medium',
    'category': '强化学习',
    'function_name': 'reward_model_loss',
    'starter_code': """def reward_model_loss(chosen_hidden, rejected_hidden, reward_head):
    pass""",
    'hint': """1. r_chosen = (chosen_hidden @ reward_head).squeeze(-1)  → (B,)
2. r_rejected = (rejected_hidden @ reward_head).squeeze(-1)
3. margin = r_chosen - r_rejected
4. loss = -log(1 / (1 + exp(-margin))).mean()  (manual sigmoid)""",
    'hint_zh': """1. r_chosen = (chosen_hidden @ reward_head).squeeze(-1)  → (B,)
2. r_rejected = (rejected_hidden @ reward_head).squeeze(-1)
3. margin = r_chosen - r_rejected
4. loss = -log(1 / (1 + exp(-margin))).mean()  (手动 sigmoid)""",
    'description_en': """Implement the canonical RLHF reward model training loss based on the Bradley-Terry preference model.

Given hidden states for chosen and rejected responses, project them to scalar rewards and compute the preference loss.

**Signature:** `reward_model_loss(chosen_hidden, rejected_hidden, reward_head) -> Tensor`

**Parameters:**
- `chosen_hidden` — last-token hidden states for chosen responses (B, D)
- `rejected_hidden` — last-token hidden states for rejected responses (B, D)
- `reward_head` — linear projection to scalar reward (D, 1)

**Returns:** scalar loss tensor

**Formula:**
1. `r_chosen = chosen_hidden @ reward_head` — shape (B, 1)
2. `r_rejected = rejected_hidden @ reward_head` — shape (B, 1)
3. `loss = -mean(log(sigmoid(r_chosen - r_rejected)))`

**Constraint:** implement sigmoid manually — `σ(x) = 1 / (1 + exp(-x))` — do not use library sigmoid functions.""",
    'description_zh': """实现基于 Bradley-Terry 偏好模型的标准 RLHF 奖励模型训练损失。

给定选中和拒绝响应的隐藏状态，将其投影为标量奖励并计算偏好损失。

**签名:** `reward_model_loss(chosen_hidden, rejected_hidden, reward_head) -> Tensor`

**参数:**
- `chosen_hidden` — 选中响应的最后一个 token 隐藏状态 (B, D)
- `rejected_hidden` — 拒绝响应的最后一个 token 隐藏状态 (B, D)
- `reward_head` — 投影到标量奖励的线性层权重 (D, 1)

**返回:** 标量损失张量

**公式:**
1. `r_chosen = chosen_hidden @ reward_head` — 形状 (B, 1)
2. `r_rejected = rejected_hidden @ reward_head` — 形状 (B, 1)
3. `loss = -mean(log(sigmoid(r_chosen - r_rejected)))`

**约束:** 手动实现 sigmoid — `σ(x) = 1 / (1 + exp(-x))` — 不得使用库函数。""",
    "tests": [
        {
            "name": 'Returns scalar',
            "code": """import torch
torch.manual_seed(0)
B, D = 4, 16
chosen = torch.randn(B, D)
rejected = torch.randn(B, D)
head = torch.randn(D, 1)
loss = {fn}(chosen, rejected, head)
assert loss.shape == torch.Size([]), f'Expected scalar, got shape {loss.shape}'
assert loss.item() > 0, 'Loss should be positive'""",
        },
        {
            "name": 'Clear preference -> low loss',
            "code": """import torch
torch.manual_seed(1)
B, D = 8, 32
head = torch.ones(D, 1) / D
# chosen hidden states much larger than rejected
chosen = torch.ones(B, D) * 10.0
rejected = torch.ones(B, D) * -10.0
loss = {fn}(chosen, rejected, head)
assert loss.item() < 0.01, f'Loss should be near 0 for clear preference, got {loss.item():.4f}'""",
        },
        {
            "name": 'Equal hidden states -> loss near log(2)',
            "code": """import torch
import math
torch.manual_seed(2)
B, D = 16, 64
chosen = torch.randn(B, D)
rejected = chosen.clone()  # identical
head = torch.randn(D, 1)
loss = {fn}(chosen, rejected, head)
assert abs(loss.item() - math.log(2)) < 0.01, f'Expected log(2)={math.log(2):.4f}, got {loss.item():.4f}'""",
        },
        {
            "name": 'Gradient flows through all inputs',
            "code": """import torch
torch.manual_seed(3)
B, D = 4, 8
chosen = torch.randn(B, D, requires_grad=True)
rejected = torch.randn(B, D, requires_grad=True)
head = torch.randn(D, 1, requires_grad=True)
loss = {fn}(chosen, rejected, head)
loss.backward()
assert chosen.grad is not None, 'No gradient for chosen_hidden'
assert rejected.grad is not None, 'No gradient for rejected_hidden'
assert head.grad is not None, 'No gradient for reward_head'""",
        },
        {
            "name": 'Larger margin -> lower loss',
            "code": """import torch
torch.manual_seed(4)
B, D = 8, 16
head = torch.ones(D, 1) / D
base = torch.zeros(B, D)
# small margin
chosen_small = base + 0.5
rejected_small = base - 0.5
loss_small = {fn}(chosen_small, rejected_small, head)
# large margin
chosen_large = base + 5.0
rejected_large = base - 5.0
loss_large = {fn}(chosen_large, rejected_large, head)
assert loss_large.item() < loss_small.item(), 'Larger margin should yield lower loss'""",
        },
    ],
    'solution': '''def reward_model_loss(chosen_hidden, rejected_hidden, reward_head):
    """Bradley-Terry 奖励模型 (Bradley-Terry Reward Model Loss)。"""
    # Step 1: 用同一个 reward_head 将 chosen 隐状态投影为标量奖励
    r_chosen = (chosen_hidden @ reward_head).squeeze(-1)
    # Step 2: 投影 rejected 隐状态并形成成对奖励差 r_chosen-r_rejected
    r_rejected = (rejected_hidden @ reward_head).squeeze(-1)
    margin = r_chosen - r_rejected
    # Step 3: 最大化 Bradley-Terry 偏好概率，返回负对数似然均值
    loss = -torch.log(1.0 / (1.0 + torch.exp(-margin))).mean()
    return loss''',
    'demo': """torch.manual_seed(42)
B, D = (8, 64)
reward_head = torch.ones(D, 1) / D
h = torch.randn(B, D)
loss_equal = reward_model_loss(h, h, reward_head)
print(f'Equal hidden states => loss = {loss_equal.item():.4f}  (expected ≈ {math.log(2):.4f})')
chosen = torch.ones(B, D) * 3.0
rejected = -torch.ones(B, D) * 3.0
loss_good = reward_model_loss(chosen, rejected, reward_head)
print(f'Positive reward margin => loss = {loss_good.item():.4f}')
print(f'Loss decreases: {loss_good.item() < loss_equal.item()}')""",
    'explanation': r"""### 核心思路
Bradley-Terry 模型把 chosen 胜过 rejected 的概率写成 $P(c\succ r)=\sigma(r_c-r_r)$。共享线性头将两条响应的末 token 隐状态投影到标量奖励，训练最小化 $L=-\frac1B\sum_i\log\sigma(r_{c,i}-r_{r,i})$；奖励差越大，偏好概率越接近 1、损失越小。

### 代码步骤拆解
- **Step 1（chosen 奖励投影）**：`chosen_hidden` 的 `(B,D)` 与 `reward_head` 的 `(D,1)` 相乘，再 `squeeze(-1)` 得到 `(B,)` 的 `r_chosen`。
- **Step 2（构造成对 margin）**：同一个头投影 `rejected_hidden`，以 `margin=r_chosen-r_rejected` 消除对两者共同奖励平移的依赖。
- **Step 3（偏好负对数似然）**：手动计算 `1/(1+exp(-margin))`，取对数、负号和批均值，返回标量训练目标。

### 重点与易错点分析
- **注意点 1（差值方向）**：必须是 chosen 减 rejected；反向会训练模型偏好被拒答案。margin 为 0 时概率为 $1/2$，损失严格为 $\log 2$。
- **注意点 2（共享奖励尺度）**：两条响应必须使用同一 `reward_head`，否则差值无法表示同一标尺上的偏好；该损失只约束相对差值，不能确定奖励的整体平移。
- **注意点 3（手写 sigmoid 的极值）**：题目要求显式 `exp`；当 margin 极负时 `exp(-margin)` 可能溢出。生产实现通常等价改写为 `softplus(-margin)` 或 `logsigmoid(margin)`，但这里保留手写约束。

### 复杂度分析
- **时间复杂度**：两次 `(B,D)@(D,1)` 投影为 $O(BD)$，其余逐样本标量运算为 $O(B)$。
- **空间复杂度**：`r_chosen`、`r_rejected` 与 `margin` 均为长度 $B$，额外空间 $O(B)$；输入隐藏状态本身为 $O(BD)$。""",
}
