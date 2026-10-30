"""Double DQN & TD3 Clipped Double-Q Targets task."""

TASK = {
    'title': 'Double DQN & TD3 Clipped Double-Q Targets',
    'title_zh': 'Double DQN 与 TD3 双 Q 目标计算',
    'difficulty': 'Medium',
    'category': '强化学习',
    'function_name': 'double_q_targets',
    'starter_code': """def double_q_targets(q_online_next, q1_target_next, q2_target_next, rewards, dones, gamma=0.99):
    pass""",
    'hint': """Choose one greedy action per batch row from online Q. Gather q1 for Double DQN and gather min(q1,q2) for the clipped target, detach bootstraps, and mask terminals.""",
    'hint_zh': '从在线 Q 的每个批次行选择贪心动作。Double DQN 收集 q1，裁剪双 Q 目标收集 min(q1,q2)；分离 bootstrap 并屏蔽终止状态。',
    'description_en': """Compute Double DQN and TD3-style clipped Double-Q temporal-difference targets at the action selected greedily by the online network.""",
    'description_zh': '在在线网络贪心选择的动作上，计算 Double DQN 和 TD3 风格的裁剪双 Q 时序差分目标。',
    'theory_en': """Decoupling action selection from target evaluation reduces maximization bias. Taking the smaller of two target critics adds a conservative clipped Double-Q estimate.""",
    'theory_zh': '将动作选择与目标评估解耦可减少最大化偏差；取两个目标评论家的较小值可得到更保守的裁剪双 Q 估计。',
    'diagram_en': """```mermaid
flowchart LR
O[online next Q] --> A[argmax action]
Q1[target Q1] --> G1[gather action]
Q1 --> M[elementwise minimum]
Q2[target Q2] --> M
A --> G1
A --> G2[gather action]
M --> G2
G1 --> D[Double DQN target]
G2 --> T[clipped Double-Q target]
```""",
    'diagram_zh': """```mermaid
flowchart LR
O[在线下一状态 Q] --> A[贪心动作]
Q1[目标 Q1] --> G1[收集动作值]
Q1 --> M[逐元素最小]
Q2[目标 Q2] --> M
A --> G1
A --> G2[收集动作值]
M --> G2
G1 --> D[Double DQN 目标]
G2 --> T[裁剪双 Q 目标]
```""",
    "tests": [
        {
            "name": 'target vector shapes',
            "code": """import torch
q = torch.randn(4, 3)
ddqn, td3 = {fn}(q, torch.randn(4, 3), torch.randn(4, 3), torch.randn(4), torch.zeros(4))
assert ddqn.shape == (4,) and td3.shape == (4,)
assert torch.isfinite(ddqn).all() and torch.isfinite(td3).all()""",
        },
        {
            "name": 'exact selected-action targets',
            "code": """import torch
qo = torch.tensor([[1., 3., 2.], [4., 0., 1.]], dtype=torch.double)
q1 = torch.tensor([[5., 6., 7.], [2., 3., 4.]], dtype=torch.double)
q2 = torch.tensor([[8., 1., 9.], [1., 9., 5.]], dtype=torch.double)
r = torch.tensor([.5, -1.], dtype=torch.double); d = torch.tensor([0., 0.])
ddqn, td3 = {fn}(qo, q1, q2, r, d, gamma=.9)
a=qo.argmax(-1,keepdim=True)
mask = 1.0 - d.float()
ref1=r+.9*mask*q1.gather(-1,a).squeeze(-1)
ref2=r+.9*mask*torch.minimum(q1,q2).gather(-1,a).squeeze(-1)
assert torch.equal(ddqn, ref1) and torch.equal(td3, ref2)""",
        },
        {
            "name": 'terminal transitions',
            "code": """import torch
r = torch.tensor([2.0, -3.0])
q = torch.tensor([[1., 2.], [3., 4.]])
ddqn, td3 = {fn}(q, q*10, q*-10, r, torch.ones(2))
assert torch.equal(ddqn, r) and torch.equal(td3, r)
assert ddqn.shape == r.shape""",
        },
        {
            "name": 'bootstrap detached',
            "code": """import torch
qo = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
q1 = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
q2 = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
r = torch.randn(3, dtype=torch.double, requires_grad=True)
ddqn, td3 = {fn}(qo, q1, q2, r, torch.zeros(3)); (ddqn+td3).sum().backward()
assert r.grad is not None and torch.equal(r.grad, torch.full_like(r, 2.0))
assert qo.grad is None and q1.grad is None and q2.grad is None""",
        },
    ],
    'solution': '''def double_q_targets(q_online_next, q1_target_next, q2_target_next, rewards, dones, gamma=0.99):
    """计算 Double DQN 与裁剪双 Q 目标 (Double-Q Targets)。"""
    # Step 1: 用在线网络选择每个样本的贪心动作
    best_actions = q_online_next.argmax(dim=-1, keepdim=True)

    # Step 2: 分别用第一目标网络和双网络最小值评估该动作
    ddqn_bootstrap = q1_target_next.gather(-1, best_actions).squeeze(-1)
    clipped_values = torch.minimum(q1_target_next, q2_target_next)
    td3_bootstrap = clipped_values.gather(-1, best_actions).squeeze(-1)

    # Step 3: 停止 bootstrap 梯度并屏蔽终止状态
    nonterminal = 1.0 - dones.float()
    ddqn_target = rewards + gamma * nonterminal * ddqn_bootstrap.detach()
    td3_target = rewards + gamma * nonterminal * td3_bootstrap.detach()
    return ddqn_target, td3_target''',
    'demo': """torch.manual_seed(42)
q_online = torch.randn(3, 4)
q1_target = torch.randn(3, 4)
q2_target = torch.randn(3, 4)
ddqn_target, td3_target = double_q_targets(q_online, q1_target, q2_target, torch.randn(3), torch.tensor([0.0, 1.0, 0.0]))
print("Double DQN target:", ddqn_target)
print("Clipped Double-Q target:", td3_target)""",
    'explanation': """### 核心思路
Double DQN 用在线网络选择动作、目标网络评估动作以降低最大化偏差；裁剪双 Q 进一步用两个目标评论家的逐动作最小值形成保守 bootstrap。

### 代码步骤拆解
- **Step 1（动作选择）**：对 `(B, A)` 的 `q_online_next` 沿动作维取 `argmax`，得到 `(B, 1)` 索引。
- **Step 2（目标评估）**：用相同索引分别收集 `q1_target_next` 与 `minimum(q1, q2)`，得到 `(B,)` bootstrap。
- **Step 3（TD 目标）**：用 `nonterminal` 屏蔽终止样本，并加到 `rewards` 形成两个目标向量。

### 重点与易错点分析
- **注意点 1（梯度截断）**：bootstrap 必须 `.detach()`，目标网络和在线动作选择均不应通过目标构造接收梯度。
- **注意点 2（选择评估解耦）**：不能直接在目标 Q 上取最大值，否则会退化为普通 DQN 的过估计形式。
- **注意点 3（终止掩码）**：`dones` 转为浮点后计算 $1-d$；终止样本目标必须严格等于即时奖励。

### 复杂度分析
- **时间复杂度**：$O(BA)$，`argmax` 与双 Q 最小值需扫描全部动作。
- **空间复杂度**：$O(BA)$，`clipped_values` 是主要中间张量。""",
}
