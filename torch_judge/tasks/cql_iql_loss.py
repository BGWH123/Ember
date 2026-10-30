"""CQL and IQL offline reinforcement-learning losses."""

TASK = {
    "title": "Offline RL: CQL & IQL Expectile Regression Loss",
    "title_zh": "离线强化学习：CQL 与 IQL 期望分位数损失",
    "difficulty": "Hard",
    "category": "强化学习",
    "function_name": "offline_rl_losses",
    "starter_code": '''import torch

def offline_rl_losses(q_all, q_data, bellman_target, v_pred,
                      alpha=1.0, tau=0.7):
    pass
''',
    "hint": "Add detached-target Bellman MSE to alpha times logsumexp(all actions)-dataset Q. For IQL, detach dataset Q, choose tau or 1-tau by residual sign, and weight its square.",
    "hint_zh": "将使用分离目标的 Bellman MSE 与 alpha 倍的 logsumexp(全部动作)-数据动作 Q 相加。IQL 中分离数据 Q，按残差符号选择 tau 或 1-tau 并加权平方。",
    "description_en": "Compute a discrete-action Conservative Q-Learning objective and an Implicit Q-Learning expectile value objective from an offline batch.",
    "description_zh": "根据离线批次计算离散动作 Conservative Q-Learning 目标和 Implicit Q-Learning 的期望分位值目标。",
    "theory_en": "CQL penalizes Q-values spread over unsupported actions while fitting Bellman targets. IQL fits V to an asymmetric expectile of fixed dataset-action Q-values.",
    "theory_zh": "CQL 在拟合 Bellman 目标的同时惩罚未被数据支持动作的高 Q 值。IQL 将 V 拟合到固定数据动作 Q 值的非对称期望分位数。",
    "diagram_en": "```mermaid\nflowchart LR\nQA[all-action Q] --> C[logsumexp penalty]\nQD[data-action Q] --> C\nQD --> B[Bellman MSE]\nT[detached target] --> B\nC --> CL[CQL loss]\nB --> CL\nQD --> U[detached Q - V]\nV[value] --> U\nU --> I[expectile loss]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nQA[全部动作 Q] --> C[logsumexp 惩罚]\nQD[数据动作 Q] --> C\nQD --> B[Bellman MSE]\nT[分离目标] --> B\nC --> CL[CQL 损失]\nB --> CL\nQD --> U[分离 Q - V]\nV[状态值] --> U\nU --> I[期望分位损失]\n```",
    "tests": [
        {"name": "two scalar losses", "code": """import torch
q = torch.randn(4, 3)
qd = torch.randn(4)
cql, iql = {fn}(q, qd, torch.randn(4), torch.randn(4))
assert cql.shape == torch.Size([]) and iql.shape == torch.Size([])
assert torch.isfinite(cql) and torch.isfinite(iql)
"""},
        {"name": "exact CQL and IQL reference", "code": """import torch
qa = torch.tensor([[1., 2.], [-1., .5]], dtype=torch.double)
qd = torch.tensor([1.5, -.2], dtype=torch.double)
t = torch.tensor([1., .3], dtype=torch.double)
v = torch.tensor([1.2, .1], dtype=torch.double)
cql, iql = {fn}(qa, qd, t, v, alpha=.4, tau=.8)
ref_c = ((qd-t)**2).mean()+.4*(torch.logsumexp(qa,-1)-qd).mean()
u=qd-v; ref_i=(torch.where(u>=0,.8,.2)*u.square()).mean()
assert torch.allclose(cql, ref_c, atol=1e-12, rtol=1e-12) and torch.allclose(iql, ref_i, atol=1e-12, rtol=1e-12)
"""},
        {"name": "zero residual edge", "code": """import torch
qd = torch.tensor([1.0, -1.0])
cql, iql = {fn}(torch.stack((qd, qd), dim=1), qd, qd, qd.clone(), alpha=0.0)
assert torch.equal(cql, torch.tensor(0.0))
assert torch.equal(iql, torch.tensor(0.0)) and cql.shape == torch.Size([])
"""},
        {"name": "gradient detach semantics", "code": """import torch
qa = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
qd = torch.randn(3, dtype=torch.double, requires_grad=True)
t = torch.randn(3, dtype=torch.double, requires_grad=True)
v = torch.randn(3, dtype=torch.double, requires_grad=True)
cql, iql = {fn}(qa, qd, t, v); (cql+iql).backward()
assert qa.grad is not None and qd.grad is not None and v.grad is not None
assert t.grad is None and qa.grad.shape == qa.shape and torch.isfinite(v.grad).all()
"""},
    ],
    "solution": '''import torch

def offline_rl_losses(q_all, q_data, bellman_target, v_pred,
                      alpha=1.0, tau=0.7):
    bellman_mse = ((q_data - bellman_target.detach()) ** 2).mean()
    conservative_penalty = (
        torch.logsumexp(q_all, dim=-1) - q_data
    ).mean()
    cql_loss = bellman_mse + alpha * conservative_penalty
    u = q_data.detach() - v_pred
    weight = torch.where(u >= 0, tau, 1.0 - tau)
    iql_loss = (weight * (u ** 2)).mean()
    return cql_loss, iql_loss
''',
    "demo": '''q_all = torch.randn(5, 4)
q_data = torch.randn(5)
print(offline_rl_losses(q_all, q_data, torch.randn(5), torch.randn(5)))
''',
    "explanation": "Only the Bellman target and IQL's Q baseline are detached. CQL still trains q_data, while IQL trains only the value prediction.",
}
