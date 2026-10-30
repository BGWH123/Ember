"""SimPO (Simple Preference Optimization) Loss task."""

TASK = {
    'title': 'SimPO (Simple Preference Optimization) Loss',
    'title_zh': 'SimPO（无参考模型偏好优化）损失',
    'difficulty': 'Easy',
    'category': '损失函数',
    'function_name': 'simpo_loss',
    'starter_code': """def simpo_loss(chosen_logps, rejected_logps, chosen_lengths, rejected_lengths, beta=2.0, gamma=0.5):
    pass""",
    'hint': """Divide each sequence log-probability by its own floating-point length, scale by beta, subtract the rejected score and target margin, then use negative log-sigmoid.""",
    'hint_zh': '将每个序列的对数概率除以对应的浮点长度，乘以 beta，减去拒绝样本分数和目标间隔，再计算负 log-sigmoid。',
    'description_en': """Implement reference-free SimPO from sums of response-token log-probabilities and response lengths. The chosen normalized score should exceed the rejected score by gamma.""",
    'description_zh': '根据回复 token 对数概率之和及回复长度实现无参考模型的 SimPO。选中回复的归一化分数应比拒绝回复高 gamma。',
    'theory_en': """Length normalization removes preference for short responses. A logistic loss trains the beta-scaled chosen-minus-rejected score to clear a fixed margin gamma.""",
    'theory_zh': '长度归一化消除对短回复的偏好；逻辑损失促使经 beta 缩放的选中与拒绝分数之差超过固定间隔 gamma。',
    'diagram_en': """```mermaid
flowchart LR
C[chosen sum / length] --> D[beta score difference]
R[rejected sum / length] --> D
G[gamma] --> D
D --> L[-log sigmoid mean]
```""",
    'diagram_zh': """```mermaid
flowchart LR
C[选中总分/长度] --> D[beta 分数差]
R[拒绝总分/长度] --> D
G[gamma] --> D
D --> L[负 log-sigmoid 均值]
```""",
    "tests": [
        {
            "name": 'scalar shape',
            "code": """import torch
c = torch.tensor([-2.0, -4.0])
r = torch.tensor([-3.0, -6.0])
loss = {fn}(c, r, torch.tensor([2, 4]), torch.tensor([3, 6]))
assert loss.shape == torch.Size([]) and torch.isfinite(loss)""",
        },
        {
            "name": 'exact reference',
            "code": """import torch
import torch.nn.functional as F
c = torch.tensor([-2.0, -9.0], dtype=torch.double)
r = torch.tensor([-4.0, -8.0], dtype=torch.double)
cl, rl = torch.tensor([2, 3]), torch.tensor([4, 2])
got = {fn}(c, r, cl, rl, beta=1.7, gamma=0.3)
logits = 1.7 / cl.float() * c - 1.7 / rl.float() * r - 0.3
assert torch.allclose(got, -F.logsigmoid(logits).mean(), atol=1e-12, rtol=1e-12)""",
        },
        {
            "name": 'single item edge case',
            "code": """import torch
import torch.nn.functional as F
got = {fn}(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([1]), torch.tensor([1]), gamma=0.0)
assert got.shape == torch.Size([])
assert torch.equal(got, torch.tensor(float(torch.log(torch.tensor(2.0)))))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
c = torch.randn(4, dtype=torch.double, requires_grad=True)
r = torch.randn(4, dtype=torch.double, requires_grad=True)
loss = {fn}(c, r, torch.tensor([1, 2, 3, 4]), torch.tensor([4, 3, 2, 1]))
loss.backward()
assert c.grad is not None and r.grad is not None
assert c.grad.shape == c.shape and torch.isfinite(r.grad).all()""",
        },
    ],
    'solution': '''def simpo_loss(chosen_logps, rejected_logps, chosen_lengths, rejected_lengths, beta=2.0, gamma=0.5):
    """实现SimPO（无参考模型偏好优化）损失（SimPO (Simple Preference Optimization) Loss）。"""
    # Step 1: 用 beta 缩放 chosen 的逐 token 平均对数概率
    chosen_scores = beta / chosen_lengths.float() * chosen_logps
    # Step 2: 减去同样缩放的 rejected 分数和 SimPO 目标间隔
    rejected_scores = beta / rejected_lengths.float() * rejected_logps
    logits = chosen_scores - rejected_scores - gamma
    # Step 3: 用稳定的负 log-sigmoid 聚合批量偏好判别损失
    loss = -F.logsigmoid(logits).mean()
    return loss''',
    'demo': """torch.manual_seed(42)
chosen_logps = torch.tensor([-4.0, -6.0])
rejected_logps = torch.tensor([-7.0, -8.0])
chosen_lengths = torch.tensor([4, 3])
rejected_lengths = torch.tensor([7, 4])
loss = simpo_loss(chosen_logps, rejected_logps, chosen_lengths, rejected_lengths)
print("Chosen average log-probs:", chosen_logps / chosen_lengths)
print("Rejected average log-probs:", rejected_logps / rejected_lengths)
print("SimPO loss:", loss.item())""",
    'explanation': r"""### 核心思路
SimPO 不依赖参考模型，而用每 token 平均对数概率作为响应奖励：$r_c=\ell_c/L_c$、$r_r=\ell_r/L_r$。它最小化 $\mathcal L=-\frac1B\sum_i\log\sigma\!\left(\beta(r_{c,i}-r_{r,i})-\gamma\right)$，推动 chosen 的平均分不仅高于 rejected，还要跨过间隔 $\gamma/\beta$。

### 代码步骤拆解
- **Step 1（响应长度归一化）**：`chosen_logps` 与 `rejected_logps` 是 $(B,)$ 的 token 对数概率和，分别除以自身的 `chosen_lengths`、`rejected_lengths` 得到可比较的平均分。
- **Step 2（构造带间隔偏好 logit）**：`beta * (chosen_scores - rejected_scores)` 控制偏好梯度陡峭度，再减 `gamma` 形成目标 margin。
- **Step 3（稳定逻辑损失）**：`F.logsigmoid` 避免显式计算 sigmoid 后取对数的下溢，批均值返回零维标量。

### 重点与易错点分析
- **注意点 1（各自长度）**：chosen 与 rejected 常有不同 token 数，必须分别除以 $L_c$、$L_r$；共用一个长度会重新引入长短回复偏差。
- **注意点 2（间隔位置）**：本题 logit 是 $\beta(r_c-r_r)-\gamma$，不是 $\beta(r_c-r_r-\gamma)$；二者对 margin 的尺度定义不同。
- **注意点 3（与 DPO 区别）**：DPO 使用策略与参考模型的 log-ratio，SimPO 完全移除参考模型并增加长度归一化与显式 margin，因而节省参考前向和对应显存。

### 复杂度分析
- **时间复杂度**：$O(B)$，输入已是序列 log-prob 总和，函数只对批中每个偏好对做常数次标量运算。
- **空间复杂度**：$O(B)$，`chosen_scores`、`rejected_scores` 和 `logits` 为主要中间向量。""",
}
