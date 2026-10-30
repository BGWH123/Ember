"""RLOO Leave-One-Out Advantage & PRM Step-Score Aggregation task."""

TASK = {
    'title': 'RLOO Leave-One-Out Advantage & PRM Step-Score Aggregation',
    'title_zh': 'RLOO 留一法优势估计与 PRM 步骤评分聚合',
    'difficulty': 'Medium',
    'category': '强化学习',
    'function_name': 'rloo_and_prm_vote',
    'starter_code': """def rloo_and_prm_vote(group_rewards, step_probs, answer_ids, eps=1e-08):
    pass""",
    'hint': """Compute all leave-one-out baselines from one row sum. For voting, use `torch.unique(..., return_inverse=True)` and scatter-add minimum trace scores into unique-answer bins.""",
    'hint_zh': '由一次行求和计算全部留一基线；投票时使用 `torch.unique(..., return_inverse=True)`，并将轨迹最小分数 scatter-add 到唯一答案桶中。',
    'description_en': """Compute grouped RLOO advantages, two process-reward trace aggregations, and a minimum-score-weighted Best-of-N answer vote.""",
    'description_zh': '计算分组 RLOO 优势、两种过程奖励轨迹聚合，以及按最小步骤分数加权的 Best-of-N 答案投票。',
    'theory_en': """RLOO removes each sample from its own baseline to avoid self-influence. PRM minimum aggregation captures the weakest reasoning step, while the product captures cumulative confidence. Weighted voting combines evidence across traces sharing an answer.""",
    'theory_zh': 'RLOO 从每个样本自身的基线中排除该样本，避免自影响。PRM 最小聚合捕捉最弱推理步骤，乘积聚合反映累积置信度；加权投票汇总相同答案的多条轨迹证据。',
    'diagram_en': """```text
group rewards -> row sum minus self -> LOO baseline -> advantage
step probabilities -> min & exp(sum(log))
trace min + answer id -> weighted bins -> argmax answer
```""",
    'diagram_zh': """```text
组奖励 -> 行和减自身 -> 留一基线 -> 优势
步骤概率 -> min 与 exp(sum(log))
轨迹最小值 + 答案 ID -> 加权桶 -> 最大权重答案
```""",
    "tests": [
        {
            "name": 'all output shapes and Python answer',
            "code": """import torch
r = torch.randn(3,4); p = torch.rand(5,6); a = torch.tensor([1,2,1,3,2])
adv,mn,prod,best = {fn}(r,p,a)
assert adv.shape == (3,4) and mn.shape == prod.shape == (5,)
assert isinstance(best,int)""",
        },
        {
            "name": 'exact RLOO and PRM reference',
            "code": """import torch
r=torch.tensor([[1.,2.,4.],[0.,-1.,3.]],dtype=torch.double)
p=torch.tensor([[.9,.8],[.4,.5],[.7,.6]],dtype=torch.double)
a=torch.tensor([5,7,5])
adv,mn,prod,best={fn}(r,p,a,eps=1e-12)
base=(r.sum(-1,keepdim=True)-r)/2; ra=r-base
rm=p.amin(-1); rp=torch.exp(torch.log(p.clamp(min=1e-12)).sum(-1))
assert torch.equal(adv,ra) and torch.equal(mn,rm) and torch.equal(prod,rp)
assert best == 5""",
        },
        {
            "name": 'zero probability and tie edge',
            "code": """import torch
r=torch.tensor([[2.,4.]])
p=torch.tensor([[0.,.5],[.5,.5],[.5,.5],[.5,.5]])
a=torch.tensor([9,3,9,3])
adv,mn,prod,best={fn}(r,p,a,eps=1e-4)
assert torch.equal(adv,torch.tensor([[-2.,2.]]))
assert mn[0] == 0 and torch.all(prod > 0)
assert best == 3""",
        },
        {
            "name": 'gradient through rewards and trace scores',
            "code": """import torch
r=torch.randn(2,3,requires_grad=True)
p=(torch.rand(4,5)+.1).requires_grad_()
adv,mn,prod,best={fn}(r,p,torch.tensor([1,2,1,3]))
(adv.square().sum()+mn.sum()+prod.sum()).backward()
assert isinstance(best,int) and adv.requires_grad and prod.requires_grad
assert r.grad is not None and p.grad is not None
assert torch.isfinite(r.grad).all() and torch.isfinite(p.grad).all()""",
        },
    ],
    'solution': '''def rloo_and_prm_vote(group_rewards, step_probs, answer_ids, eps=1e-08):
    """RLOO 留一法优势估计与 PRM 步骤评分聚合 (RLOO Leave-One-Out Advantage & PRM Step-Score Aggregation)。"""
    # Step 1: 由组内总奖励减去自身，构造每条轨迹的留一基线与优势
    K = group_rewards.shape[-1]
    sum_r = group_rewards.sum(dim=-1, keepdim=True)
    baseline = (sum_r - group_rewards) / (K - 1)
    rloo_adv = group_rewards - baseline
    # Step 2: 用最弱步骤与对数域乘积形成两种 PRM 轨迹分数
    prm_min = step_probs.amin(dim=-1)
    prm_prod = torch.exp(torch.log(step_probs.clamp(min=eps)).sum(dim=-1))
    # Step 3: 按答案 ID 累加最小步骤分数，选出总证据最大的答案
    unique_answers, inverse = torch.unique(answer_ids, sorted=True, return_inverse=True)
    vote_weights = torch.zeros(unique_answers.shape[0], device=prm_min.device, dtype=prm_min.dtype)
    vote_weights = vote_weights.scatter_add(0, inverse, prm_min)
    best_answer = int(unique_answers[torch.argmax(vote_weights)].item())
    return (rloo_adv, prm_min, prm_prod, best_answer)''',
    'demo': """torch.manual_seed(42)
rewards = torch.randn(2, 4)
probs = torch.rand(6, 8)
answers = torch.tensor([10, 11, 10, 12, 11, 10])
adv, mn, prod, best = rloo_and_prm_vote(rewards, probs, answers)
print('RLOO advantages shape / row sums:', adv.shape, adv.sum(dim=-1))
print('PRM min vs product (first trace):', mn[0].item(), prod[0].item())
print('Weighted-vote answer:', best)""",
    'explanation': r"""### 核心思路
RLOO 对组内第 $i$ 条轨迹使用不含自身的基线 $b_i=(\sum_jr_j-r_i)/(K-1)$，优势 $A_i=r_i-b_i$ 避免样本奖励污染自己的控制变量。PRM 对每条推理轨迹同时计算最弱步骤 $\min_s p_s$ 与联合式分数 $\prod_s p_s$；最终按 `answer_ids` 汇总最弱步骤分数，让多条同答案轨迹共同投票。

### 代码步骤拆解
- **Step 1（留一优势）**：`sum_r` 只做一次组内求和，广播计算所有 `baseline`，再由 `group_rewards-baseline` 得到同形状 `(B,K)` 的 `rloo_adv`。
- **Step 2（PRM 轨迹聚合）**：`amin` 捕获每条轨迹的瓶颈步骤；乘积通过 `exp(sum(log(clamp(step_probs))))` 在对数域计算，输出均为 `(T,)`。
- **Step 3（答案级加权投票）**：`torch.unique(...,return_inverse=True)` 把每条轨迹映射到唯一答案桶，`scatter_add` 累加 `prm_min`，最大桶对应 `best_answer`。

### 重点与易错点分析
- **注意点 1（基线必须排除自身）**：若直接减组均值，当前样本会影响自己的基线并引入偏差；RLOO 要除以 `K-1`，因此每组至少需要两条轨迹。
- **注意点 2（两种 PRM 语义不同）**：最小值只关注最薄弱推理环节，乘积则随步骤数增长快速衰减并惩罚多处不确定；`eps` 仅保护对数，所以原始零概率的 `prm_min` 仍保持为零。
- **注意点 3（投票并列规则与梯度）**：唯一答案已排序，`argmax` 并列时选择最小答案 ID；投票索引和最终 Python 整数不可微，但 `rloo_adv`、`prm_min`、`prm_prod` 的连续输出保留梯度。

### 复杂度分析
- **时间复杂度**：奖励部分为 $O(BK)$，$T$ 条、每条 $S$ 步的 PRM 聚合为 $O(TS)$；唯一化通常为 $O(T\log T)$，总计 $O(BK+TS+T\log T)$。
- **空间复杂度**：优势张量占 $O(BK)$，轨迹分数、逆映射和至多 $T$ 个投票桶占 $O(T)$，合计 $O(BK+T)$。""",
}
