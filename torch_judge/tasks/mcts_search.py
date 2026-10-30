"""MCTS for Reasoning task."""

TASK = {
    'title': 'MCTS for Reasoning',
    'title_zh': '推理蒙特卡洛树搜索',
    'difficulty': 'Hard',
    'category': '采样与解码',
    'function_name': 'mcts_step',
    'starter_code': """def mcts_step(values, visit_counts, parent_visits, c_puct=1.414):
    pass""",
    'hint': """1. UCB(i) = values[i] + c_puct·√(log(parent_visits+1) / (visit_counts[i]+1))
2. selected = argmax(UCB)
3. rollout = torch.rand(1).item()  (random value in [0,1])
4. Q_new = (Q_old·n + rollout) / (n+1);  visit_counts[selected] += 1""",
    'hint_zh': """1. UCB(i) = values[i] + c_puct·√(log(parent_visits+1) / (visit_counts[i]+1))
2. selected = argmax(UCB)
3. rollout = torch.rand(1).item()（[0,1] 内随机值）
4. Q_new = (Q_old·n + rollout) / (n+1);  visit_counts[selected] += 1""",
    'description_en': """Implement one step of Monte Carlo Tree Search (MCTS) — the core algorithm behind o1/AlphaProof-style LLM reasoning.

**Signature:** `mcts_step(values, visit_counts, parent_visits, c_puct=1.414) -> (int, Tensor, Tensor)`

**Parameters:**
- `values` — current value estimates for N child nodes (N,)
- `visit_counts` — visit counts for each child (N,)
- `parent_visits` — total visits to the parent node (int or scalar)
- `c_puct` — exploration constant (default 1.414)

**Returns:** `(selected_idx, updated_values, updated_visits)`

**Algorithm:**
1. **Selection** — compute UCB1 scores and pick `argmax`:
   `UCB(i) = Q(i) + c_puct * sqrt(log(parent_visits + 1) / (visit_counts[i] + 1))`
2. **Rollout** — simulate a random value in [0, 1] for the selected node
3. **Backpropagation** — update with running mean:
   `Q_new = (Q_old * n + rollout_value) / (n + 1)` where `n` is the old visit count
4. Increment `visit_counts[selected_idx]` by 1""",
    'description_zh': """实现蒙特卡洛树搜索（MCTS）的单步操作——o1/AlphaProof 风格 LLM 推理的核心算法。

**签名:** `mcts_step(values, visit_counts, parent_visits, c_puct=1.414) -> (int, Tensor, Tensor)`

**参数:**
- `values` — N 个子节点的当前价值估计 (N,)
- `visit_counts` — 每个子节点的访问次数 (N,)
- `parent_visits` — 父节点的总访问次数（int 或标量）
- `c_puct` — 探索常数（默认 1.414）

**返回:** `(selected_idx, updated_values, updated_visits)`

**算法:**
1. **选择** — 计算 UCB1 分数并取 `argmax`：
   `UCB(i) = Q(i) + c_puct * sqrt(log(parent_visits + 1) / (visit_counts[i] + 1))`
2. **模拟** — 为选中节点模拟 [0, 1] 内的随机价值
3. **反向传播** — 用滑动均值更新：
   `Q_new = (Q_old * n + rollout_value) / (n + 1)`，其中 `n` 为旧访问次数
4. 将 `visit_counts[selected_idx]` 加 1""",
    "tests": [
        {
            "name": 'Return types and shapes',
            "code": """import torch
torch.manual_seed(0)
values = torch.rand(5)
visits = torch.zeros(5, dtype=torch.float32)
idx, new_vals, new_visits = {fn}(values, visits, parent_visits=10)
assert isinstance(idx, int), f'selected_idx must be int, got {type(idx)}'
assert new_vals.shape == values.shape, f'values shape mismatch: {new_vals.shape}'
assert new_visits.shape == visits.shape, f'visits shape mismatch: {new_visits.shape}'""",
        },
        {
            "name": 'Unvisited nodes preferred',
            "code": """import torch
# All nodes visited except node 2 — UCB should strongly prefer node 2
values = torch.zeros(4)
visits = torch.tensor([10.0, 10.0, 0.0, 10.0])
idx, _, _ = {fn}(values, visits, parent_visits=30)
assert idx == 2, f'Expected unvisited node 2 to be selected, got {idx}'""",
        },
        {
            "name": 'Visit count incremented',
            "code": """import torch
torch.manual_seed(42)
values = torch.rand(6)
visits = torch.ones(6)
idx, _, new_visits = {fn}(values, visits, parent_visits=6)
assert new_visits[idx] == visits[idx] + 1, 'visit_counts[selected_idx] must increase by 1'
# Other counts unchanged
for i in range(6):
    if i != idx:
        assert new_visits[i] == visits[i], f'visit_counts[{i}] should not change'""",
        },
        {
            "name": 'Value updated via running mean',
            "code": """import torch
torch.manual_seed(7)
values = torch.tensor([0.5, 0.5, 0.5])
visits = torch.tensor([2.0, 2.0, 2.0])
idx, new_vals, _ = {fn}(values, visits, parent_visits=6)
# new value must differ from old (running mean with random rollout)
assert new_vals[idx] != values[idx], 'Value at selected_idx should be updated'
# Other values unchanged
for i in range(3):
    if i != idx:
        assert new_vals[i] == values[i], f'values[{i}] should not change'""",
        },
        {
            "name": 'Pure exploitation with c_puct=0',
            "code": """import torch
values = torch.tensor([0.2, 0.9, 0.5, 0.1])
visits = torch.tensor([1.0, 1.0, 1.0, 1.0])
idx, _, _ = {fn}(values, visits, parent_visits=4, c_puct=0.0)
assert idx == 1, f'With c_puct=0, should select highest-value node (1), got {idx}'""",
        },
    ],
    'solution': '''def mcts_step(values, visit_counts, parent_visits, c_puct=1.414):
    """执行一次蒙特卡洛树搜索更新 (MCTS Step)。"""
    # Step 1: 计算价值与探索奖励之和并选择最大节点
    parent = torch.as_tensor(parent_visits + 1, dtype=values.dtype, device=values.device)
    exploration = c_puct * torch.sqrt(
        torch.log(parent) / (visit_counts.to(values.dtype) + 1.0)
    )
    selected_idx = int((values + exploration).argmax().item())

    # Step 2: 为选中节点采样一次随机 rollout 价值
    rollout_value = torch.rand((), dtype=values.dtype, device=values.device)
    old_count = visit_counts[selected_idx].to(values.dtype)
    new_value = (values[selected_idx] * old_count + rollout_value) / (old_count + 1.0)

    # Step 3: 克隆输入并更新运行均值与访问次数
    updated_values = values.clone()
    updated_visits = visit_counts.clone()
    updated_values[selected_idx] = new_value
    updated_visits[selected_idx] += 1
    return selected_idx, updated_values, updated_visits''',
    'demo': """torch.manual_seed(42)
values = torch.zeros(5)
visits = torch.zeros(5)
selected, updated_values, updated_visits = mcts_step(values, visits, parent_visits=0)
print("Selected index:", selected)
print("Updated values:", updated_values)
print("Updated visits:", updated_visits)""",
    'explanation': r"""### 核心思路
MCTS 单步以 $Q_i+c\sqrt{\log(N+1)/(n_i+1)}$ 平衡利用与探索，选中节点后采样 rollout，并用运行均值更新其价值和访问次数。未选节点保持不变。

### 代码步骤拆解
- **Step 1（UCB 选择）**：对所有子节点向量化计算 `exploration` 并取 argmax。
- **Step 2（随机模拟）**：在输入相同 dtype 和 device 上采样标量 `rollout_value`。
- **Step 3（反向更新）**：克隆两个输入，仅更新选中索引。

### 重点与易错点分析
- **注意点 1（不修改输入）**：返回克隆结果可避免调用方的树状态被意外原地覆盖。
- **注意点 2（计数加一）**：公式分母使用旧计数 `old_count + 1`，随后访问次数才加一。
- **注意点 3（纯利用边界）**：`c_puct=0` 时探索项消失，应严格选择最大 `values`。

### 复杂度分析
- **时间复杂度**：$O(N)$，计算所有子节点 UCB 并查找最大值。
- **空间复杂度**：$O(N)$，为保持输入不变需克隆价值和访问数组。""",
}
