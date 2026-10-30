"""PUCT selection score task."""

TASK = {
    "title": "AlphaZero / Reasoning MCTS PUCT Selection",
    "title_zh": "PUCT 树搜索节点选择评分",
    "difficulty": "Easy",
    "category": "采样与解码",
    "function_name": "puct_scores",
    "starter_code": '''import torch
import math

def puct_scores(q_values, priors, visit_counts, parent_visits, c_puct=1.5):
    pass
''',
    "hint": "Add Q-values to a prior-weighted exploration bonus proportional to sqrt(parent visits) and inversely proportional to one plus child visits; then take argmax.",
    "hint_zh": "将 Q 值与探索奖励相加：奖励正比于先验概率和父节点访问次数平方根，反比于子节点访问次数加一；最后取最大值索引。",
    "description_en": "Compute PUCT scores for all child actions and return both the score vector and selected action index.",
    "description_zh": "计算所有子动作的 PUCT 分数，并返回分数向量和所选动作索引。",
    "theory_en": "PUCT balances exploitation through Q with exploration through a policy prior. Frequently visited children receive a smaller exploration bonus.",
    "theory_zh": "PUCT 通过 Q 值利用已知收益，并通过策略先验探索；访问越频繁的子节点，其探索奖励越小。",
    "diagram_en": "```mermaid\nflowchart LR\nQ[Q values] --> S[PUCT scores]\nP[priors] --> B[exploration bonus]\nN[parent/child visits] --> B\nB --> S\nS --> A[argmax action]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nQ[Q 值] --> S[PUCT 分数]\nP[先验] --> B[探索奖励]\nN[父/子访问次数] --> B\nB --> S\nS --> A[最大分数动作]\n```",
    "tests": [
        {"name": "shape and index", "code": """import torch
scores, action = {fn}(torch.tensor([0.1, 0.2, 0.0]), torch.tensor([0.2, 0.7, 0.1]), torch.tensor([1, 0, 3]), 4.0)
assert scores.shape == (3,) and isinstance(action, int)
assert action == int(torch.argmax(scores).item())
"""},
        {"name": "exact PUCT reference", "code": """import torch
import math
q = torch.tensor([0.3, -0.1, 0.4], dtype=torch.double)
p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.double)
n = torch.tensor([2, 1, 4])
got, selected = {fn}(q, p, n, torch.tensor(9.0), c_puct=2.0)
ref = q + 2.0 * p * (math.sqrt(9.0) / (1.0 + n.float()))
assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12) and selected == ref.argmax().item()
"""},
        {"name": "zero parent visits", "code": """import torch
q = torch.tensor([-1.0, 2.0])
scores, selected = {fn}(q, torch.tensor([0.9, 0.1]), torch.tensor([0, 0]), 0.0)
assert torch.equal(scores, q)
assert selected == 1
"""},
        {"name": "score gradient flow", "code": """import torch
q = torch.randn(4, dtype=torch.double, requires_grad=True)
p = torch.rand(4, dtype=torch.double, requires_grad=True)
scores, selected = {fn}(q, p, torch.arange(4), 10.0)
scores.sum().backward()
assert q.grad is not None and p.grad is not None
assert q.grad.shape == q.shape and torch.isfinite(p.grad).all() and 0 <= selected < 4
"""},
    ],
    "solution": '''import torch
import math

def puct_scores(q_values, priors, visit_counts, parent_visits, c_puct=1.5):
    bonus = c_puct * priors * (
        math.sqrt(float(parent_visits)) / (1.0 + visit_counts.float())
    )
    scores = q_values + bonus
    return scores, int(torch.argmax(scores).item())
''',
    "demo": '''q = torch.tensor([0.2, 0.3, 0.1])
priors = torch.tensor([0.5, 0.3, 0.2])
print(puct_scores(q, priors, torch.tensor([3, 1, 0]), 5.0))
''',
    "explanation": "The exploration term is zero before the parent has visits and decays separately for each child. Selection itself is the nondifferentiable argmax, while scores retain gradients.",
}
