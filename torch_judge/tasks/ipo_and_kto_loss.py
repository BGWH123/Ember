"""IPO and KTO alignment-loss task."""

TASK = {
    "title": "IPO (Identity Preference) & KTO (Prospect Theory) Alignment Losses",
    "title_zh": "IPO 与 KTO 偏好对齐损失",
    "difficulty": "Medium",
    "category": "损失函数",
    "function_name": "ipo_and_kto_losses",
    "starter_code": '''import torch

def ipo_and_kto_losses(pi_chosen, pi_rejected, ref_chosen, ref_rejected, tau=0.1, beta_kto=0.1):
    pass
''',
    "hint": "Detach both reference-model log-probabilities. Build IPO from the chosen-minus-rejected log-ratio margin, and detach the batch-average KTO reference point before applying sigmoids.",
    "hint_zh": "分离两个参考模型对数概率；用选中减拒绝的对数比边际构造 IPO，并在 sigmoid 前分离 KTO 的批次平均参考点。",
    "description_en": "Compute IPO's squared preference-margin objective and KTO's prospect-theoretic desirable/undesirable objective from policy and reference sequence log-probabilities.",
    "description_zh": "根据策略与参考模型的序列对数概率，计算 IPO 平方偏好边际目标和 KTO 前景理论中的期望/非期望目标。",
    "theory_en": "IPO regresses the implicit reward gap toward a finite target instead of driving it unbounded. KTO evaluates desirable and undesirable samples relative to a detached reference point, reflecting asymmetric prospect-style value.",
    "theory_zh": "IPO 将隐式奖励差回归到有限目标，而非无限推高。KTO 相对分离的参考点评估期望与非期望样本，体现非对称的前景价值。",
    "diagram_en": "```text\npolicy logp - stopgrad(reference logp) -> log ratios\nratios -> squared IPO margin\nratios -> stopgrad z0 -> desirable/undesirable sigmoids -> KTO\n```",
    "diagram_zh": "```text\n策略 logp - stopgrad(参考 logp) -> 对数比\n对数比 -> IPO 平方边际\n对数比 -> stopgrad z0 -> 期望/非期望 sigmoid -> KTO\n```",
    "tests": [
        {
            "name": "scalar loss shapes",
            "code": '''import torch
x = [torch.randn(4) for _ in range(4)]
ipo,kto = {fn}(*x)
assert ipo.shape == torch.Size([]) and kto.shape == torch.Size([])
assert torch.isfinite(ipo) and torch.isfinite(kto)
''',
        },
        {
            "name": "exact IPO and KTO formulas",
            "code": '''import torch
pc=torch.tensor([-.2,-.6],dtype=torch.double); pr=torch.tensor([-1.,-.8],dtype=torch.double)
rc=torch.tensor([-.4,-.5],dtype=torch.double); rr=torch.tensor([-.7,-1.1],dtype=torch.double)
ipo,kto={fn}(pc,pr,rc,rr,tau=.25,beta_kto=.3)
rw=pc-rc.detach(); rl=pr-rr.detach()
ri=((rw-rl-1/(2*.25))**2).mean()
z=.5*(rw.detach().mean()+rl.detach().mean())
vd=torch.sigmoid(.3*(rw-z)); vu=torch.sigmoid(.3*(z-rl))
rk=(1-.5*(vd+vu)).mean()
assert torch.allclose(ipo,ri,atol=1e-12,rtol=1e-12)
assert torch.allclose(kto,rk,atol=1e-12,rtol=1e-12)
''',
        },
        {
            "name": "symmetric-ratio edge case",
            "code": '''import torch
pi = torch.tensor([-.5,-1.]); ref = torch.tensor([-.7,-1.2])
ipo,kto = {fn}(pi,pi,ref,ref,tau=.5,beta_kto=2.)
assert torch.allclose(ipo,torch.tensor(1.0))
assert torch.allclose(kto,torch.tensor(.5))
''',
        },
        {
            "name": "policy gradients and detached references",
            "code": '''import torch
pc=torch.randn(5,requires_grad=True); pr=torch.randn(5,requires_grad=True)
rc=torch.randn(5,requires_grad=True); rr=torch.randn(5,requires_grad=True)
ipo,kto={fn}(pc,pr,rc,rr)
(ipo+kto).backward()
assert pc.grad is not None and pr.grad is not None
assert torch.isfinite(pc.grad).all() and torch.isfinite(pr.grad).all()
assert rc.grad is None and rr.grad is None
''',
        },
    ],
    "solution": '''import torch

def ipo_and_kto_losses(pi_chosen, pi_rejected, ref_chosen, ref_rejected, tau=0.1, beta_kto=0.1):
    log_rw = pi_chosen - ref_chosen.detach()
    log_rl = pi_rejected - ref_rejected.detach()
    ipo_loss = ((log_rw - log_rl - 1.0 / (2.0 * tau)) ** 2).mean()
    z0 = 0.5 * (log_rw.detach().mean() + log_rl.detach().mean())
    v_desirable = torch.sigmoid(beta_kto * (log_rw - z0))
    v_undesirable = torch.sigmoid(beta_kto * (z0 - log_rl))
    kto_loss = (1.0 - 0.5 * (v_desirable + v_undesirable)).mean()
    return ipo_loss, kto_loss
''',
    "demo": '''pc,pr = torch.tensor([-.3,-.4]),torch.tensor([-1.,-.8])
rc,rr = torch.tensor([-.5,-.6]),torch.tensor([-.7,-.7])
print(ipo_and_kto_losses(pc,pr,rc,rr))
''',
    "explanation": "Reference log-probabilities and z0 are stop-gradient quantities. Gradients therefore train only the policy ratios while IPO and KTO retain their prescribed batch means.",
}
