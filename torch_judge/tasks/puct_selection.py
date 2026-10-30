"""AlphaZero / Reasoning MCTS PUCT Selection task."""

TASK = {
    'title': 'AlphaZero / Reasoning MCTS PUCT Selection',
    'title_zh': 'PUCT 树搜索节点选择评分',
    'difficulty': 'Easy',
    'category': '采样与解码',
    'function_name': 'puct_scores',
    'starter_code': """def puct_scores(q_values, priors, visit_counts, parent_visits, c_puct=1.5):
    pass""",
    'hint': """Add Q-values to a prior-weighted exploration bonus proportional to sqrt(parent visits) and inversely proportional to one plus child visits; then take argmax.""",
    'hint_zh': '将 Q 值与探索奖励相加：奖励正比于先验概率和父节点访问次数平方根，反比于子节点访问次数加一；最后取最大值索引。',
    'description_en': """Compute PUCT scores for all child actions and return both the score vector and selected action index.""",
    'description_zh': '计算所有子动作的 PUCT 分数，并返回分数向量和所选动作索引。',
    'theory_en': """PUCT balances exploitation through Q with exploration through a policy prior. Frequently visited children receive a smaller exploration bonus.""",
    'theory_zh': 'PUCT 通过 Q 值利用已知收益，并通过策略先验探索；访问越频繁的子节点，其探索奖励越小。',
    'diagram_en': """```mermaid
flowchart LR
Q[Q values] --> S[PUCT scores]
P[priors] --> B[exploration bonus]
N[parent/child visits] --> B
B --> S
S --> A[argmax action]
```""",
    'diagram_zh': """```mermaid
flowchart LR
Q[Q 值] --> S[PUCT 分数]
P[先验] --> B[探索奖励]
N[父/子访问次数] --> B
B --> S
S --> A[最大分数动作]
```""",
    "tests": [
        {
            "name": 'shape and index',
            "code": """import torch
scores, action = {fn}(torch.tensor([0.1, 0.2, 0.0]), torch.tensor([0.2, 0.7, 0.1]), torch.tensor([1, 0, 3]), 4.0)
assert scores.shape == (3,) and isinstance(action, int)
assert action == int(torch.argmax(scores).item())""",
        },
        {
            "name": 'exact PUCT reference',
            "code": """import torch
import math
q = torch.tensor([0.3, -0.1, 0.4], dtype=torch.double)
p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.double)
n = torch.tensor([2, 1, 4])
got, selected = {fn}(q, p, n, torch.tensor(9.0), c_puct=2.0)
ref = q + 2.0 * p * (math.sqrt(9.0) / (1.0 + n.float()))
assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12) and selected == ref.argmax().item()""",
        },
        {
            "name": 'zero parent visits',
            "code": """import torch
q = torch.tensor([-1.0, 2.0])
scores, selected = {fn}(q, torch.tensor([0.9, 0.1]), torch.tensor([0, 0]), 0.0)
assert torch.equal(scores, q)
assert selected == 1""",
        },
        {
            "name": 'score gradient flow',
            "code": """import torch
q = torch.randn(4, dtype=torch.double, requires_grad=True)
p = torch.rand(4, dtype=torch.double, requires_grad=True)
scores, selected = {fn}(q, p, torch.arange(4), 10.0)
scores.sum().backward()
assert q.grad is not None and p.grad is not None
assert q.grad.shape == q.shape and torch.isfinite(p.grad).all() and 0 <= selected < 4""",
        },
    ],
    'solution': '''def puct_scores(q_values, priors, visit_counts, parent_visits, c_puct=1.5):
    """PUCT 树搜索节点选择评分 (AlphaZero / Reasoning MCTS PUCT Selection)。"""
    # Step 1: 用父节点访问量与策略先验计算每个动作的探索奖励
    bonus = c_puct * priors * (math.sqrt(float(parent_visits)) / (1.0 + visit_counts.float()))
    # Step 2: 将探索奖励叠加到动作价值 Q，平衡利用与探索
    scores = q_values + bonus
    # Step 3: 选择 PUCT 分数最大的子动作继续树搜索
    return (scores, int(torch.argmax(scores).item()))''',
    'demo': """torch.manual_seed(42)
q = torch.tensor([0.2, 0.3, 0.1])
priors = torch.tensor([0.5, 0.3, 0.2])
visits = torch.tensor([8, 1, 0])
scores, selected = puct_scores(q, priors, visits, parent_visits=12, c_puct=1.5)
print('Q values:', q)
print('PUCT scores:', scores)
print('Selected action:', selected)""",
    'explanation': r"""### 核心思路
PUCT 为动作 $a$ 定义 $S(a)=Q(a)+c_{\mathrm{puct}}P(a)\sqrt{N_{\mathrm{parent}}}/(1+N(a))$。第一项利用当前价值估计，第二项优先探索策略先验高且访问少的子节点；随着 $N(a)$ 增长，该动作的探索奖励衰减，搜索预算逐步分配到其他分支。

### 代码步骤拆解
- **Step 1（计算探索奖励）**：`priors` 逐动作乘 `sqrt(parent_visits)`，再除以 `1+visit_counts.float()`；加 1 让尚未访问的子节点分母合法。
- **Step 2（融合价值与先验）**：`scores=q_values+bonus` 对每个动作形成统一选择尺度，`c_puct` 控制策略先验相对 Q 值的影响。
- **Step 3（贪心树策略）**：`torch.argmax(scores)` 返回本轮扩展或模拟的动作，`.item()` 转成接口要求的 Python `int`。

### 重点与易错点分析
- **注意点 1（父子访问量位置）**：分子是父节点总访问量的平方根，分母是各子节点访问量；交换二者会让访问越多的动作反而获得更高探索奖励。
- **注意点 2（零访问根节点）**：`parent_visits=0` 时全部探索奖励为零，选择完全由 `q_values` 决定；实际 MCTS 也可能通过根节点噪声或初始化访问量启动探索。
- **注意点 3（与 UCB1 的差别）**：PUCT 以神经策略 `priors` 调制探索，而 UCB1 常用 $\sqrt{\log N/n_a}$；`argmax` 的动作选择不可微，但返回的 `scores` 对 `q_values` 和 `priors` 可微。

### 复杂度分析
- **时间复杂度**：对 $A$ 个合法动作计算分数并执行一次最大值扫描，为 $O(A)$。
- **空间复杂度**：探索奖励与分数各是长度 $A$ 的向量，为 $O(A)$；所选索引为 $O(1)$。""",
}
