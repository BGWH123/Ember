"""IPO (Identity Preference) & KTO (Prospect Theory) Alignment Losses task."""

TASK = {
    'title': 'IPO (Identity Preference) & KTO (Prospect Theory) Alignment Losses',
    'title_zh': 'IPO 与 KTO 偏好对齐损失',
    'difficulty': 'Medium',
    'category': '损失函数',
    'function_name': 'ipo_and_kto_losses',
    'starter_code': """def ipo_and_kto_losses(pi_chosen, pi_rejected, ref_chosen, ref_rejected, tau=0.1, beta_kto=0.1):
    pass""",
    'hint': """Detach both reference-model log-probabilities. Build IPO from the chosen-minus-rejected log-ratio margin, and detach the batch-average KTO reference point before applying sigmoids.""",
    'hint_zh': '分离两个参考模型对数概率；用选中减拒绝的对数比边际构造 IPO，并在 sigmoid 前分离 KTO 的批次平均参考点。',
    'description_en': """Compute IPO's squared preference-margin objective and KTO's prospect-theoretic desirable/undesirable objective from policy and reference sequence log-probabilities.""",
    'description_zh': '根据策略与参考模型的序列对数概率，计算 IPO 平方偏好边际目标和 KTO 前景理论中的期望/非期望目标。',
    'theory_en': """IPO regresses the implicit reward gap toward a finite target instead of driving it unbounded. KTO evaluates desirable and undesirable samples relative to a detached reference point, reflecting asymmetric prospect-style value.""",
    'theory_zh': 'IPO 将隐式奖励差回归到有限目标，而非无限推高。KTO 相对分离的参考点评估期望与非期望样本，体现非对称的前景价值。',
    'diagram_en': """```text
policy logp - stopgrad(reference logp) -> log ratios
ratios -> squared IPO margin
ratios -> stopgrad z0 -> desirable/undesirable sigmoids -> KTO
```""",
    'diagram_zh': """```text
策略 logp - stopgrad(参考 logp) -> 对数比
对数比 -> IPO 平方边际
对数比 -> stopgrad z0 -> 期望/非期望 sigmoid -> KTO
```""",
    "tests": [
        {
            "name": 'scalar loss shapes',
            "code": """import torch
x = [torch.randn(4) for _ in range(4)]
ipo,kto = {fn}(*x)
assert ipo.shape == torch.Size([]) and kto.shape == torch.Size([])
assert torch.isfinite(ipo) and torch.isfinite(kto)""",
        },
        {
            "name": 'exact IPO and KTO formulas',
            "code": """import torch
pc=torch.tensor([-.2,-.6],dtype=torch.double); pr=torch.tensor([-1.,-.8],dtype=torch.double)
rc=torch.tensor([-.4,-.5],dtype=torch.double); rr=torch.tensor([-.7,-1.1],dtype=torch.double)
ipo,kto={fn}(pc,pr,rc,rr,tau=.25,beta_kto=.3)
rw=pc-rc.detach(); rl=pr-rr.detach()
ri=((rw-rl-1/(2*.25))**2).mean()
z=.5*(rw.detach().mean()+rl.detach().mean())
vd=torch.sigmoid(.3*(rw-z)); vu=torch.sigmoid(.3*(z-rl))
rk=(1-.5*(vd+vu)).mean()
assert torch.allclose(ipo,ri,atol=1e-12,rtol=1e-12)
assert torch.allclose(kto,rk,atol=1e-12,rtol=1e-12)""",
        },
        {
            "name": 'symmetric-ratio edge case',
            "code": """import torch
pi = torch.tensor([-.5,-1.]); ref = torch.tensor([-.7,-1.2])
ipo,kto = {fn}(pi,pi,ref,ref,tau=.5,beta_kto=2.)
assert torch.allclose(ipo,torch.tensor(1.0))
assert torch.allclose(kto,torch.tensor(.5))""",
        },
        {
            "name": 'policy gradients and detached references',
            "code": """import torch
pc=torch.randn(5,requires_grad=True); pr=torch.randn(5,requires_grad=True)
rc=torch.randn(5,requires_grad=True); rr=torch.randn(5,requires_grad=True)
ipo,kto={fn}(pc,pr,rc,rr)
(ipo+kto).backward()
assert pc.grad is not None and pr.grad is not None
assert torch.isfinite(pc.grad).all() and torch.isfinite(pr.grad).all()
assert rc.grad is None and rr.grad is None""",
        },
    ],
    'solution': '''def ipo_and_kto_losses(pi_chosen, pi_rejected, ref_chosen, ref_rejected, tau=0.1, beta_kto=0.1):
    """计算 IPO 与 KTO 对齐损失 (IPO and KTO Losses)。"""
    # Step 1: 构造策略相对固定参考模型的对数比
    chosen_ratio = pi_chosen - ref_chosen.detach()
    rejected_ratio = pi_rejected - ref_rejected.detach()

    # Step 2: 回归 IPO 的有限偏好边际
    target_margin = 1.0 / (2.0 * tau)
    ipo_loss = ((chosen_ratio - rejected_ratio - target_margin) ** 2).mean()

    # Step 3: 以停梯度参考点计算 KTO 前景价值
    reference_point = 0.5 * (chosen_ratio.detach().mean() + rejected_ratio.detach().mean())
    desirable = torch.sigmoid(beta_kto * (chosen_ratio - reference_point))
    undesirable = torch.sigmoid(beta_kto * (reference_point - rejected_ratio))
    kto_loss = (1.0 - 0.5 * (desirable + undesirable)).mean()
    return ipo_loss, kto_loss''',
    'demo': """torch.manual_seed(42)
pi_chosen = torch.randn(4)
pi_rejected = torch.randn(4)
ref_chosen = torch.randn(4)
ref_rejected = torch.randn(4)
ipo_loss, kto_loss = ipo_and_kto_losses(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
print("IPO loss:", ipo_loss.item())
print("KTO loss:", kto_loss.item())
print("Output shapes:", ipo_loss.shape, kto_loss.shape)""",
    'explanation': r"""### 核心思路
IPO 将相对参考模型的偏好差回归到 $1/(2\tau)$；KTO 以分离的批次参考点为基准，用 sigmoid 衡量期望与非期望样本。

### 代码步骤拆解
- **Step 1（对数比）**：策略 log-prob 减去已分离的参考 log-prob。
- **Step 2（IPO 边际）**：选中与拒绝比值之差对有限目标做平方回归。
- **Step 3（KTO 价值）**：构造 `reference_point` 后组合两类 sigmoid 价值并取均值。

### 重点与易错点分析
- **注意点 1（参考模型停梯度）**：`ref_chosen` 和 `ref_rejected` 必须 `.detach()`。
- **注意点 2（参考点停梯度）**：批次均值 `reference_point` 也需分离，避免样本间额外梯度耦合。
- **注意点 3（边际系数）**：IPO 目标是 $1/(2\tau)$，遗漏因子 2 会改变最优点。

### 复杂度分析
- **时间复杂度**：$O(B)$，所有运算均为逐样本或批次归约。
- **空间复杂度**：$O(B)$，保存两个对数比和价值张量。""",
}
