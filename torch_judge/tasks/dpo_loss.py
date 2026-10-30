"""DPO (Direct Preference Optimization) Loss task."""

TASK = {
    'title': 'DPO (Direct Preference Optimization) Loss',
    'title_zh': 'DPO 损失',
    'difficulty': 'Medium',
    'category': '损失函数',
    'function_name': 'dpo_loss',
    'starter_code': """def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pass""",
    'hint': """1. chosen_logratios = β·(π_chosen - ref_chosen)
2. rejected_logratios = β·(π_rejected - ref_rejected)
3. loss = -log(sigmoid(chosen_logratios - rejected_logratios)).mean()""",
    'hint_zh': """1. chosen_logratios = β·(π_chosen - ref_chosen)
2. rejected_logratios = β·(π_rejected - ref_rejected)
3. loss = -log(sigmoid(chosen_logratios - rejected_logratios)).mean()""",
    'description_en': """Implement the DPO (Direct Preference Optimization) loss.

DPO aligns language models with human preferences without reinforcement learning, using paired chosen/rejected log-probabilities.

**Signature:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor`

**Parameters:**
- `policy_chosen_logps` — policy log-probs for chosen responses (B,)
- `policy_rejected_logps` — policy log-probs for rejected responses (B,)
- `ref_chosen_logps`, `ref_rejected_logps` — reference model log-probs (B,)
- `beta` — temperature scaling factor

**Returns:** scalar loss

**Constraints:**
- `L = -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))).mean()`""",
    'description_zh': """实现 DPO（直接偏好优化）损失。

DPO 无需强化学习即可将语言模型与人类偏好对齐，使用配对的选中/拒绝对数概率。

**签名:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor`

**参数:**
- `policy_chosen_logps` — 策略模型对选中回复的对数概率 (B,)
- `policy_rejected_logps` — 策略模型对拒绝回复的对数概率 (B,)
- `ref_chosen_logps`, `ref_rejected_logps` — 参考模型的对数概率 (B,)
- `beta` — 温度缩放因子

**返回:** 标量损失

**约束:**
- `L = -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))).mean()`""",
    "tests": [
        {
            "name": 'Easy pair: small loss',
            "code": """import torch
chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-10.0, -10.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
loss = {fn}(chosen, rejected, ref_c, ref_r, beta=0.1)
assert loss.dim() == 0, 'Must be scalar'
assert loss.item() < 0.5, f'Easy pair loss too high: {loss.item():.4f}'""",
        },
        {
            "name": 'Hard pair: large loss',
            "code": """import torch
loss = {fn}(torch.tensor([-10.0]), torch.tensor([0.0]),
            torch.tensor([-1.0]), torch.tensor([-1.0]), beta=0.1)
assert loss.item() > 0.5, f'Hard pair loss too low: {loss.item():.4f}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
c = torch.randn(4, requires_grad=True)
r = torch.randn(4, requires_grad=True)
{fn}(c, r, torch.randn(4), torch.randn(4)).backward()
assert c.grad is not None and r.grad is not None, 'Missing gradients'""",
        },
        {
            "name": 'Mathematical correctness',
            "code": """import torch
torch.manual_seed(0)
c, r, rc, rr = torch.randn(3), torch.randn(3), torch.randn(3), torch.randn(3)
beta = 0.5
loss = {fn}(c, r, rc, rr, beta=beta)
ref = -torch.log(torch.sigmoid(beta * ((c - rc) - (r - rr)))).mean()
assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'""",
        },
        {
            "name": 'Beta scaling',
            "code": """import torch
torch.manual_seed(0)
c, r, rc, rr = torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4)
l1 = {fn}(c, r, rc, rr, beta=0.1)
l2 = {fn}(c, r, rc, rr, beta=1.0)
assert not torch.allclose(l1, l2), 'Different beta should give different loss'""",
        },
    ],
    'solution': '''def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """计算直接偏好优化损失 (Direct Preference Optimization Loss)。"""
    # Step 1: 计算策略相对参考模型的胜者与败者对数比
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    # Step 2: 构造温度缩放的偏好间隔
    preference_margin = beta * (chosen_log_ratio - rejected_log_ratio)

    # Step 3: 用稳定的对数 sigmoid 计算批次平均损失
    loss = -F.logsigmoid(preference_margin).mean()
    return loss''',
    'demo': """torch.manual_seed(42)
policy_chosen = torch.randn(4)
policy_rejected = torch.randn(4)
ref_chosen = torch.randn(4)
ref_rejected = torch.randn(4)
loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
print("DPO loss:", loss.item())
print("Batch size:", policy_chosen.shape[0])""",
    'explanation': r"""### 核心思路
DPO 最大化策略相对参考模型对胜者的偏好优势，损失为 $-\log\sigma(\beta[(\log\pi_c-\log\pi_{ref,c})-(\log\pi_r-\log\pi_{ref,r})])$ 的批次均值。

### 代码步骤拆解
- **Step 1（相对奖励）**：分别计算胜者和败者中策略与参考模型的对数概率差。
- **Step 2（偏好间隔）**：两组 `(B,)` 对数比相减并乘 `beta`，得到 `preference_margin`。
- **Step 3（逻辑损失）**：对间隔应用 `F.logsigmoid`，取负并平均为标量损失。

### 重点与易错点分析
- **注意点 1（数值稳定性）**：`F.logsigmoid` 避免先 sigmoid 再 log 在大负间隔下发生下溢。
- **注意点 2（差分顺序）**：必须是胜者间隔减败者间隔，反向会训练模型偏好拒绝回答。
- **注意点 3（温度作用）**：`beta` 缩放的是完整偏好间隔，它控制参考约束与偏好信号的敏感度。

### 复杂度分析
- **时间复杂度**：$O(B)$，全部操作均为逐样本标量运算。
- **空间复杂度**：$O(B)$，保存两组对数比与偏好间隔。""",
}
