"""Offline RL: CQL & IQL Expectile Regression Loss task."""

TASK = {
    'title': 'Offline RL: CQL & IQL Expectile Regression Loss',
    'title_zh': '离线强化学习：CQL 与 IQL 期望分位数损失',
    'difficulty': 'Hard',
    'category': '强化学习',
    'function_name': 'offline_rl_losses',
    'starter_code': """def offline_rl_losses(q_all, q_data, bellman_target, v_pred, alpha=1.0, tau=0.7):
    pass""",
    'hint': """Add detached-target Bellman MSE to alpha times logsumexp(all actions)-dataset Q. For IQL, detach dataset Q, choose tau or 1-tau by residual sign, and weight its square.""",
    'hint_zh': '将使用分离目标的 Bellman MSE 与 alpha 倍的 logsumexp(全部动作)-数据动作 Q 相加。IQL 中分离数据 Q，按残差符号选择 tau 或 1-tau 并加权平方。',
    'description_en': """Compute a discrete-action Conservative Q-Learning objective and an Implicit Q-Learning expectile value objective from an offline batch.""",
    'description_zh': '根据离线批次计算离散动作 Conservative Q-Learning 目标和 Implicit Q-Learning 的期望分位值目标。',
    'theory_en': """CQL penalizes Q-values spread over unsupported actions while fitting Bellman targets. IQL fits V to an asymmetric expectile of fixed dataset-action Q-values.""",
    'theory_zh': 'CQL 在拟合 Bellman 目标的同时惩罚未被数据支持动作的高 Q 值。IQL 将 V 拟合到固定数据动作 Q 值的非对称期望分位数。',
    'diagram_en': """```mermaid
flowchart LR
QA[all-action Q] --> C[logsumexp penalty]
QD[data-action Q] --> C
QD --> B[Bellman MSE]
T[detached target] --> B
C --> CL[CQL loss]
B --> CL
QD --> U[detached Q - V]
V[value] --> U
U --> I[expectile loss]
```""",
    'diagram_zh': """```mermaid
flowchart LR
QA[全部动作 Q] --> C[logsumexp 惩罚]
QD[数据动作 Q] --> C
QD --> B[Bellman MSE]
T[分离目标] --> B
C --> CL[CQL 损失]
B --> CL
QD --> U[分离 Q - V]
V[状态值] --> U
U --> I[期望分位损失]
```""",
    "tests": [
        {
            "name": 'two scalar losses',
            "code": """import torch
q = torch.randn(4, 3)
qd = torch.randn(4)
cql, iql = {fn}(q, qd, torch.randn(4), torch.randn(4))
assert cql.shape == torch.Size([]) and iql.shape == torch.Size([])
assert torch.isfinite(cql) and torch.isfinite(iql)""",
        },
        {
            "name": 'exact CQL and IQL reference',
            "code": """import torch
qa = torch.tensor([[1., 2.], [-1., .5]], dtype=torch.double)
qd = torch.tensor([1.5, -.2], dtype=torch.double)
t = torch.tensor([1., .3], dtype=torch.double)
v = torch.tensor([1.2, .1], dtype=torch.double)
cql, iql = {fn}(qa, qd, t, v, alpha=.4, tau=.8)
ref_c = ((qd-t)**2).mean()+.4*(torch.logsumexp(qa,-1)-qd).mean()
u=qd-v; ref_i=(torch.where(u>=0,.8,.2)*u.square()).mean()
assert torch.allclose(cql, ref_c, atol=1e-12, rtol=1e-12) and torch.allclose(iql, ref_i, atol=1e-12, rtol=1e-12)""",
        },
        {
            "name": 'zero residual edge',
            "code": """import torch
qd = torch.tensor([1.0, -1.0])
cql, iql = {fn}(torch.stack((qd, qd), dim=1), qd, qd, qd.clone(), alpha=0.0)
assert torch.equal(cql, torch.tensor(0.0))
assert torch.equal(iql, torch.tensor(0.0)) and cql.shape == torch.Size([])""",
        },
        {
            "name": 'gradient detach semantics',
            "code": """import torch
qa = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
qd = torch.randn(3, dtype=torch.double, requires_grad=True)
t = torch.randn(3, dtype=torch.double, requires_grad=True)
v = torch.randn(3, dtype=torch.double, requires_grad=True)
cql, iql = {fn}(qa, qd, t, v); (cql+iql).backward()
assert qa.grad is not None and qd.grad is not None and v.grad is not None
assert t.grad is None and qa.grad.shape == qa.shape and torch.isfinite(v.grad).all()""",
        },
    ],
    'solution': '''def offline_rl_losses(q_all, q_data, bellman_target, v_pred, alpha=1.0, tau=0.7):
    """计算保守 Q 学习与隐式 Q 学习损失 (CQL and IQL Losses)。"""
    # Step 1: 拟合停止梯度的 Bellman 目标，并加入保守正则项
    bellman_loss = ((q_data - bellman_target.detach()) ** 2).mean()
    conservative_penalty = (torch.logsumexp(q_all, dim=-1) - q_data).mean()
    cql_loss = bellman_loss + alpha * conservative_penalty

    # Step 2: 对固定数据 Q 值执行非对称期望分位回归
    residual = q_data.detach() - v_pred
    weights = torch.where(residual >= 0, tau, 1.0 - tau)
    iql_loss = (weights * residual.square()).mean()
    return cql_loss, iql_loss''',
    'demo': """torch.manual_seed(42)
q_all = torch.randn(5, 4)
q_data = torch.randn(5)
cql_loss, iql_loss = offline_rl_losses(q_all, q_data, torch.randn(5), torch.randn(5))
print("CQL loss:", cql_loss.item())
print("IQL loss:", iql_loss.item())""",
    'explanation': r"""### 核心思路
CQL 用 $\operatorname{MSE}(Q_{data}, y)+\alpha(\log\sum_a e^{Q_a}-Q_{data})$ 抑制数据外动作的高估；IQL 则对固定的 `q_data` 与 `v_pred` 之差执行非对称期望分位回归。

### 代码步骤拆解
- **Step 1（Bellman 拟合）**：将 `bellman_target` 停止梯度后与 `q_data` 计算均方误差，输出标量 `bellman_loss`。
- **Step 2（保守正则）**：沿动作维对 `q_all` 计算 `logsumexp`，减去数据动作价值后取批次均值。
- **Step 3（期望分位回归）**：根据 `residual` 的符号选择 `tau` 或 $1-\tau$，加权平方后得到 `iql_loss`。

### 重点与易错点分析
- **注意点 1（梯度边界）**：`bellman_target.detach()` 与 `q_data.detach()` 的位置不同；CQL 仍需更新 `q_data`，IQL 只通过该项更新 `v_pred`。
- **注意点 2（数值稳定性）**：必须使用 `torch.logsumexp`，直接先指数再取对数会在大 Q 值时溢出。
- **注意点 3（残差符号）**：零残差归入 `tau` 分支但平方仍为零，不改变边界结果。

### 复杂度分析
- **时间复杂度**：$O(BA)$，主要计算瓶颈是全部动作上的 `logsumexp`。
- **空间复杂度**：$O(B)$，除输入外仅保存逐样本残差与权重。""",
}
