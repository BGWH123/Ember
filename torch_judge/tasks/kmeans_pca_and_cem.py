"""PCA via SVD & Cross-Entropy Method (CEM) Trajectory Planner task."""

TASK = {
    'title': 'PCA via SVD & Cross-Entropy Method (CEM) Trajectory Planner',
    'title_zh': 'SVD 主成分分析（PCA）与 CEM 交叉熵规划器',
    'difficulty': 'Medium',
    'category': '具身智能与控制',
    'function_name': 'pca_and_cem_plan',
    'starter_code': """def pca_and_cem_plan(X, k_pca, candidate_actions, costs, num_elites):
    pass""",
    'hint': """Center X and take the leading rows of Vh as principal axes. Negate costs before top-k so that the selected trajectories have the smallest original costs, then fit population mean and standard deviation.""",
    'hint_zh': '中心化 X，并取 Vh 的前几行作为主轴。top-k 前对代价取负以选择原始代价最小的轨迹，再拟合总体均值和标准差。',
    'description_en': """Project data X onto k_pca principal components obtained by compact SVD, and perform one CEM refitting step on the num_elites lowest-cost candidate action sequences.""",
    'description_zh': '使用紧凑 SVD 得到的 k_pca 个主成分投影数据 X，并在代价最低的 num_elites 条候选动作序列上执行一次 CEM 重拟合。',
    'theory_en': """PCA finds orthogonal directions of maximal centered-data variation. CEM turns optimization into distribution fitting: retain low-cost elites and update each trajectory coordinate's Gaussian mean and scale.""",
    'theory_zh': 'PCA 寻找中心化数据方差最大的正交方向。CEM 将优化转化为分布拟合：保留低代价精英，并更新轨迹各坐标的高斯均值与尺度。',
    'diagram_en': """```text
X -> center -> SVD -> leading Vh -> projected scores
candidate trajectories + costs -> lowest-cost elites -> mean,std
```""",
    'diagram_zh': """```text
X -> 中心化 -> SVD -> Vh 前部 -> 投影分数
候选轨迹 + 代价 -> 最低代价精英 -> 均值、标准差
```""",
    "tests": [
        {
            "name": 'return shapes',
            "code": """import torch
X=torch.randn(8,5); actions=torch.randn(10,4,3); costs=torch.randn(10)
p,c,m,s={fn}(X,2,actions,costs,3)
assert p.shape==(8,2) and c.shape==(2,5) and m.shape==(4,3) and s.shape==(4,3)
assert torch.isfinite(p).all() and torch.isfinite(s).all()""",
        },
        {
            "name": 'exact SVD PCA reference',
            "code": """import torch
X=torch.tensor([[1.,2.,0.],[2.,0.,1.],[4.,1.,3.],[0.,3.,2.]])
a=torch.arange(24.).reshape(4,3,2); costs=torch.tensor([3.,1.,4.,2.])
p,c,m,s={fn}(X,2,a,costs,2)
xc=X-X.mean(0,keepdim=True); U,S,Vh=torch.linalg.svd(xc,full_matrices=False); rc=Vh[:2]; rp=xc@rc.T
assert torch.equal(c,rc) and torch.equal(p,rp)""",
        },
        {
            "name": 'lowest-cost elites and one-elite edge',
            "code": """import torch
X=torch.tensor([[0.,0.],[1.,1.],[2.,2.]])
a=torch.tensor([[[0.,0.]],[[10.,20.]],[[2.,4.]],[[6.,8.]]]); costs=torch.tensor([5.,-1.,2.,0.])
p,c,m,s={fn}(X,1,a,costs,2); elites=a[torch.tensor([1,3])]
assert torch.equal(m,elites.mean(0)) and torch.equal(s,elites.std(0,unbiased=False))
p,c,m1,s1={fn}(X,1,a,costs,1)
assert torch.equal(m1,a[1]) and torch.equal(s1,torch.zeros_like(a[1]))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
X=torch.randn(7,4,requires_grad=True); a=torch.randn(6,3,2,requires_grad=True); costs=torch.tensor([3.,1.,5.,0.,2.,4.])
p,c,m,s={fn}(X,2,a,costs,3); (p.square().sum()+c.square().sum()+m.sum()+s.sum()).backward()
assert X.grad is not None and a.grad is not None
assert torch.isfinite(X.grad).all() and torch.isfinite(a.grad).all()""",
        },
    ],
    'solution': '''def pca_and_cem_plan(X, k_pca, candidate_actions, costs, num_elites):
    """执行 SVD 主成分分析与 CEM 重拟合 (PCA and CEM)。"""
    # Step 1: 中心化数据并提取主成分
    centered = X - X.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:k_pca]
    projected = centered @ components.T

    # Step 2: 选择代价最低的精英轨迹
    elite_indices = torch.topk(-costs, k=num_elites).indices
    elites = candidate_actions[elite_indices]

    # Step 3: 以总体统计量重拟合动作分布
    elite_mean = elites.mean(dim=0)
    elite_std = elites.std(dim=0, unbiased=False)
    return projected, components, elite_mean, elite_std''',
    'demo': """torch.manual_seed(42)
X = torch.randn(20, 6)
candidate_actions = torch.randn(32, 5, 3)
costs = torch.randn(32)
projected, components, elite_mean, elite_std = pca_and_cem_plan(X, 2, candidate_actions, costs, 8)
print("Projection shape:", projected.shape)
print("Component shape:", components.shape)
print("Elite statistic shapes:", elite_mean.shape, elite_std.shape)""",
    'explanation': r"""### 核心思路
PCA 对中心化数据做紧凑 SVD 并投影到前 $k$ 个右奇异向量；CEM 选择最低代价轨迹并用其均值和总体标准差重拟合分布。

### 代码步骤拆解
- **Step 1（PCA 投影）**：`components` 为 `(k_pca, D)`，`projected` 为 `(N, k_pca)`。
- **Step 2（精英选择）**：对 `-costs` 做 `topk` 等价于选择原代价最小项。
- **Step 3（分布重拟合）**：沿候选维归约，均值与标准差保留动作序列形状。

### 重点与易错点分析
- **注意点 1（中心化）**：SVD 前必须减特征均值，否则首主成分会受数据偏置影响。
- **注意点 2（Vh 方向）**：主轴是 `Vh` 的前几行，投影时使用 `components.T`。
- **注意点 3（总体标准差）**：`unbiased=False` 使单精英场景返回零而非 NaN。

### 复杂度分析
- **时间复杂度**：$O(ND\min(N,D)+C\log k)$，SVD 通常是主要瓶颈。
- **空间复杂度**：$O(ND+kD+CTA)$，保存中心化数据、主成分和精英轨迹。""",
}
