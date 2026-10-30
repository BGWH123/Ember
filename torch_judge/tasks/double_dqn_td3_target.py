"""Double DQN and clipped Double-Q target task."""

TASK = {
    "title": "Double DQN & TD3 Clipped Double-Q Targets",
    "title_zh": "Double DQN 与 TD3 双 Q 目标计算",
    "difficulty": "Medium",
    "category": "强化学习",
    "function_name": "double_q_targets",
    "starter_code": '''import torch

def double_q_targets(q_online_next, q1_target_next, q2_target_next,
                     rewards, dones, gamma=0.99):
    pass
''',
    "hint": "Choose one greedy action per batch row from online Q. Gather q1 for Double DQN and gather min(q1,q2) for the clipped target, detach bootstraps, and mask terminals.",
    "hint_zh": "从在线 Q 的每个批次行选择贪心动作。Double DQN 收集 q1，裁剪双 Q 目标收集 min(q1,q2)；分离 bootstrap 并屏蔽终止状态。",
    "description_en": "Compute Double DQN and TD3-style clipped Double-Q temporal-difference targets at the action selected greedily by the online network.",
    "description_zh": "在在线网络贪心选择的动作上，计算 Double DQN 和 TD3 风格的裁剪双 Q 时序差分目标。",
    "theory_en": "Decoupling action selection from target evaluation reduces maximization bias. Taking the smaller of two target critics adds a conservative clipped Double-Q estimate.",
    "theory_zh": "将动作选择与目标评估解耦可减少最大化偏差；取两个目标评论家的较小值可得到更保守的裁剪双 Q 估计。",
    "diagram_en": "```mermaid\nflowchart LR\nO[online next Q] --> A[argmax action]\nQ1[target Q1] --> G1[gather action]\nQ1 --> M[elementwise minimum]\nQ2[target Q2] --> M\nA --> G1\nA --> G2[gather action]\nM --> G2\nG1 --> D[Double DQN target]\nG2 --> T[clipped Double-Q target]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nO[在线下一状态 Q] --> A[贪心动作]\nQ1[目标 Q1] --> G1[收集动作值]\nQ1 --> M[逐元素最小]\nQ2[目标 Q2] --> M\nA --> G1\nA --> G2[收集动作值]\nM --> G2\nG1 --> D[Double DQN 目标]\nG2 --> T[裁剪双 Q 目标]\n```",
    "tests": [
        {"name": "target vector shapes", "code": """import torch
q = torch.randn(4, 3)
ddqn, td3 = {fn}(q, torch.randn(4, 3), torch.randn(4, 3), torch.randn(4), torch.zeros(4))
assert ddqn.shape == (4,) and td3.shape == (4,)
assert torch.isfinite(ddqn).all() and torch.isfinite(td3).all()
"""},
        {"name": "exact selected-action targets", "code": """import torch
qo = torch.tensor([[1., 3., 2.], [4., 0., 1.]], dtype=torch.double)
q1 = torch.tensor([[5., 6., 7.], [2., 3., 4.]], dtype=torch.double)
q2 = torch.tensor([[8., 1., 9.], [1., 9., 5.]], dtype=torch.double)
r = torch.tensor([.5, -1.], dtype=torch.double); d = torch.tensor([0., 0.])
ddqn, td3 = {fn}(qo, q1, q2, r, d, gamma=.9)
a=qo.argmax(-1,keepdim=True)
mask = 1.0 - d.float()
ref1=r+.9*mask*q1.gather(-1,a).squeeze(-1)
ref2=r+.9*mask*torch.minimum(q1,q2).gather(-1,a).squeeze(-1)
assert torch.equal(ddqn, ref1) and torch.equal(td3, ref2)
"""},
        {"name": "terminal transitions", "code": """import torch
r = torch.tensor([2.0, -3.0])
q = torch.tensor([[1., 2.], [3., 4.]])
ddqn, td3 = {fn}(q, q*10, q*-10, r, torch.ones(2))
assert torch.equal(ddqn, r) and torch.equal(td3, r)
assert ddqn.shape == r.shape
"""},
        {"name": "bootstrap detached", "code": """import torch
qo = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
q1 = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
q2 = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
r = torch.randn(3, dtype=torch.double, requires_grad=True)
ddqn, td3 = {fn}(qo, q1, q2, r, torch.zeros(3)); (ddqn+td3).sum().backward()
assert r.grad is not None and torch.equal(r.grad, torch.full_like(r, 2.0))
assert qo.grad is None and q1.grad is None and q2.grad is None
"""},
    ],
    "solution": '''import torch

def double_q_targets(q_online_next, q1_target_next, q2_target_next,
                     rewards, dones, gamma=0.99):
    best_actions = q_online_next.argmax(dim=-1, keepdim=True)
    q_ddqn_next = q1_target_next.gather(
        -1, best_actions
    ).squeeze(-1)
    mask = 1.0 - dones.float()
    ddqn_target = rewards + gamma * mask * q_ddqn_next.detach()
    q_min_next = torch.minimum(
        q1_target_next, q2_target_next
    ).gather(-1, best_actions).squeeze(-1)
    td3_target = rewards + gamma * mask * q_min_next.detach()
    return ddqn_target, td3_target
''',
    "demo": '''online = torch.tensor([[1.0, 3.0, 2.0]])
q1 = torch.tensor([[2.0, 4.0, 5.0]])
q2 = torch.tensor([[3.0, 1.0, 6.0]])
print(double_q_targets(online, q1, q2, torch.tensor([0.5]), torch.tensor([0.])))
''',
    "explanation": "Both methods share online-network action selection. The second target evaluates that action with the lower target critic, and terminal masks remove all bootstrap contribution.",
}
