"""RLOO advantage and PRM aggregation task."""

TASK = {
    "title": "RLOO Leave-One-Out Advantage & PRM Step-Score Aggregation",
    "title_zh": "RLOO 留一法优势估计与 PRM 步骤评分聚合",
    "difficulty": "Medium",
    "category": "强化学习",
    "function_name": "rloo_and_prm_vote",
    "starter_code": '''import torch

def rloo_and_prm_vote(group_rewards, step_probs, answer_ids, eps=1e-8):
    pass
''',
    "hint": "Compute all leave-one-out baselines from one row sum. For voting, use `torch.unique(..., return_inverse=True)` and scatter-add minimum trace scores into unique-answer bins.",
    "hint_zh": "由一次行求和计算全部留一基线；投票时使用 `torch.unique(..., return_inverse=True)`，并将轨迹最小分数 scatter-add 到唯一答案桶中。",
    "description_en": "Compute grouped RLOO advantages, two process-reward trace aggregations, and a minimum-score-weighted Best-of-N answer vote.",
    "description_zh": "计算分组 RLOO 优势、两种过程奖励轨迹聚合，以及按最小步骤分数加权的 Best-of-N 答案投票。",
    "theory_en": "RLOO removes each sample from its own baseline to avoid self-influence. PRM minimum aggregation captures the weakest reasoning step, while the product captures cumulative confidence. Weighted voting combines evidence across traces sharing an answer.",
    "theory_zh": "RLOO 从每个样本自身的基线中排除该样本，避免自影响。PRM 最小聚合捕捉最弱推理步骤，乘积聚合反映累积置信度；加权投票汇总相同答案的多条轨迹证据。",
    "diagram_en": "```text\ngroup rewards -> row sum minus self -> LOO baseline -> advantage\nstep probabilities -> min & exp(sum(log))\ntrace min + answer id -> weighted bins -> argmax answer\n```",
    "diagram_zh": "```text\n组奖励 -> 行和减自身 -> 留一基线 -> 优势\n步骤概率 -> min 与 exp(sum(log))\n轨迹最小值 + 答案 ID -> 加权桶 -> 最大权重答案\n```",
    "tests": [
        {
            "name": "all output shapes and Python answer",
            "code": '''import torch
r = torch.randn(3,4); p = torch.rand(5,6); a = torch.tensor([1,2,1,3,2])
adv,mn,prod,best = {fn}(r,p,a)
assert adv.shape == (3,4) and mn.shape == prod.shape == (5,)
assert isinstance(best,int)
''',
        },
        {
            "name": "exact RLOO and PRM reference",
            "code": '''import torch
r=torch.tensor([[1.,2.,4.],[0.,-1.,3.]],dtype=torch.double)
p=torch.tensor([[.9,.8],[.4,.5],[.7,.6]],dtype=torch.double)
a=torch.tensor([5,7,5])
adv,mn,prod,best={fn}(r,p,a,eps=1e-12)
base=(r.sum(-1,keepdim=True)-r)/2; ra=r-base
rm=p.amin(-1); rp=torch.exp(torch.log(p.clamp(min=1e-12)).sum(-1))
assert torch.equal(adv,ra) and torch.equal(mn,rm) and torch.equal(prod,rp)
assert best == 5
''',
        },
        {
            "name": "zero probability and tie edge",
            "code": '''import torch
r=torch.tensor([[2.,4.]])
p=torch.tensor([[0.,.5],[.5,.5],[.5,.5],[.5,.5]])
a=torch.tensor([9,3,9,3])
adv,mn,prod,best={fn}(r,p,a,eps=1e-4)
assert torch.equal(adv,torch.tensor([[-2.,2.]]))
assert mn[0] == 0 and torch.all(prod > 0)
assert best == 3
''',
        },
        {
            "name": "gradient through rewards and trace scores",
            "code": '''import torch
r=torch.randn(2,3,requires_grad=True)
p=(torch.rand(4,5)+.1).requires_grad_()
adv,mn,prod,best={fn}(r,p,torch.tensor([1,2,1,3]))
(adv.square().sum()+mn.sum()+prod.sum()).backward()
assert isinstance(best,int) and adv.requires_grad and prod.requires_grad
assert r.grad is not None and p.grad is not None
assert torch.isfinite(r.grad).all() and torch.isfinite(p.grad).all()
''',
        },
    ],
    "solution": '''import torch

def rloo_and_prm_vote(group_rewards, step_probs, answer_ids, eps=1e-8):
    K = group_rewards.shape[-1]
    sum_r = group_rewards.sum(dim=-1, keepdim=True)
    baseline = (sum_r - group_rewards) / (K - 1)
    rloo_adv = group_rewards - baseline
    prm_min = step_probs.amin(dim=-1)
    prm_prod = torch.exp(torch.log(step_probs.clamp(min=eps)).sum(dim=-1))
    unique_answers, inverse = torch.unique(answer_ids, sorted=True, return_inverse=True)
    vote_weights = torch.zeros(unique_answers.shape[0], device=prm_min.device, dtype=prm_min.dtype)
    vote_weights = vote_weights.scatter_add(0, inverse, prm_min)
    best_answer = int(unique_answers[torch.argmax(vote_weights)].item())
    return rloo_adv, prm_min, prm_prod, best_answer
''',
    "demo": '''rewards=torch.randn(2,4); probs=torch.rand(6,8); answers=torch.tensor([10,11,10,12,11,10])
adv,mn,prod,best=rloo_and_prm_vote(rewards,probs,answers)
print(adv.shape,mn,prod,best)
''',
    "explanation": "RLOO uses K-1 peers for every sample. The vote sums weakest-link confidence across traces with the same final answer; sorted unique IDs make equal-weight ties deterministic in favor of the smaller ID.",
}
