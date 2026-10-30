"""Diffusion-DPO and REPA representation alignment task."""

TASK = {
    "title": "Diffusion-DPO & REPA Representation Alignment Loss",
    "title_zh": "Diffusion-DPO 与 REPA 表征对齐损失",
    "difficulty": "Hard",
    "category": "扩散与流模型",
    "function_name": "diffusion_dpo_and_repa",
    "starter_code": '''def diffusion_dpo_and_repa(eps_theta_w, eps_ref_w, eps_w, eps_theta_l, eps_ref_l, eps_l, hidden_proj, ssl_target, beta=500.0, eps=1e-8):
    pass
''',
    "hint": "Compute four per-sample noise MSEs, compare policy-minus-reference errors for winner and loser through -logsigmoid, then add a separate negative cosine-similarity REPA calculation.",
    "hint_zh": "计算四个逐样本噪声 MSE，通过 -logsigmoid 比较胜者与败者的策略减参考误差，再单独计算负余弦相似度 REPA。",
    "description_en": "Compute Diffusion-DPO preference loss from winner/loser noise predictions of shape (B,D), and REPA alignment from projected and SSL features (B,N,C). Reference predictions and SSL targets are detached. Return (dpo_loss,repa_loss).",
    "description_zh": "从形状 (B,D) 的胜者/败者噪声预测计算 Diffusion-DPO 偏好损失，并从投影特征与 SSL 特征 (B,N,C) 计算 REPA 对齐。参考预测与 SSL 目标需停止梯度。返回 (dpo_loss,repa_loss)。",
    "theory_en": "Diffusion-DPO favors a policy whose denoising error improves over the reference more for preferred samples than rejected ones. REPA maximizes tokenwise cosine similarity to frozen self-supervised representations.",
    "theory_zh": "Diffusion-DPO 鼓励策略相对参考模型在偏好样本上的去噪误差改善大于拒绝样本。REPA 最大化 token 级特征与冻结自监督表征的余弦相似度。",
    "diagram_en": "```text\nwinner/loser policy and reference errors -> relative margin -> -log sigmoid -> DPO\nhidden projection + detached SSL target -> normalize -> negative cosine -> REPA\n```",
    "diagram_zh": "```text\n胜/负样本策略与参考误差 -> 相对间隔 -> -log sigmoid -> DPO\n隐藏投影 + 停止梯度的 SSL 目标 -> 归一化 -> 负余弦 -> REPA\n```",
    "tests": [
        {
            "name": "scalar output shapes",
            "code": '''import torch
noise=[torch.randn(3,5) for _ in range(6)]
h=torch.randn(3,4,7); z=torch.randn(3,4,7)
d,r={fn}(*noise,h,z)
assert d.shape==torch.Size([]) and r.shape==torch.Size([])
assert torch.isfinite(d) and torch.isfinite(r)
''',
        },
        {
            "name": "exact numerical reference",
            "code": '''import torch
import torch.nn.functional as F
torch.manual_seed(6)
tw,rw,w,tl,rl,l=[torch.randn(2,4) for _ in range(6)]
h=torch.randn(2,3,5); z=torch.randn(2,3,5)
d,r={fn}(tw,rw,w,tl,rl,l,h,z,beta=2.,eps=1e-6)
etw=((tw-w)**2).mean(-1); erw=((rw.detach()-w)**2).mean(-1)
etl=((tl-l)**2).mean(-1); erl=((rl.detach()-l)**2).mean(-1)
rd=-F.logsigmoid(-2.*((etw-erw)-(etl-erl))).mean()
rr=-((h/(h.norm(dim=-1,keepdim=True)+1e-6))*(z.detach()/(z.detach().norm(dim=-1,keepdim=True)+1e-6))).sum(-1).mean()
assert torch.equal(d,rd) and torch.equal(r,rr)
''',
        },
        {
            "name": "zero and identical feature edge cases",
            "code": '''import torch
q=torch.zeros(2,3); h=torch.ones(2,4,5); zero=torch.zeros_like(h)
d,r={fn}(q,q,q,q,q,q,h,zero,beta=500.)
assert torch.allclose(d,torch.tensor(0.69314718),atol=1e-7)
assert r.item()==0.0 and torch.isfinite(r)
''',
        },
        {
            "name": "gradient routing",
            "code": '''import torch
tw=torch.randn(2,3,requires_grad=True); rw=torch.randn(2,3,requires_grad=True); w=torch.randn(2,3)
tl=torch.randn(2,3,requires_grad=True); rl=torch.randn(2,3,requires_grad=True); l=torch.randn(2,3)
h=torch.randn(2,4,5,requires_grad=True); z=torch.randn(2,4,5,requires_grad=True)
d,r={fn}(tw,rw,w,tl,rl,l,h,z,beta=3.); (d+r).backward()
assert tw.grad is not None and tl.grad is not None and h.grad is not None
assert rw.grad is None and rl.grad is None and z.grad is None
''',
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F

def diffusion_dpo_and_repa(eps_theta_w, eps_ref_w, eps_w, eps_theta_l, eps_ref_l, eps_l, hidden_proj, ssl_target, beta=500.0, eps=1e-8):
    err_theta_w = ((eps_theta_w - eps_w) ** 2).mean(dim=-1)
    err_ref_w = ((eps_ref_w.detach() - eps_w) ** 2).mean(dim=-1)
    err_theta_l = ((eps_theta_l - eps_l) ** 2).mean(dim=-1)
    err_ref_l = ((eps_ref_l.detach() - eps_l) ** 2).mean(dim=-1)
    diff_w = err_theta_w - err_ref_w
    diff_l = err_theta_l - err_ref_l
    dpo_loss = -F.logsigmoid(-beta * (diff_w - diff_l)).mean()
    h_norm = hidden_proj / (hidden_proj.norm(dim=-1, keepdim=True) + eps)
    target = ssl_target.detach()
    z_norm = target / (target.norm(dim=-1, keepdim=True) + eps)
    repa_loss = -(h_norm * z_norm).sum(dim=-1).mean()
    return dpo_loss, repa_loss
''',
    "demo": '''noise = [torch.randn(4, 32) for _ in range(6)]
hidden = torch.randn(4, 16, 64)
target = torch.randn_like(hidden)
print(diffusion_dpo_and_repa(*noise, hidden, target))
''',
    "explanation": "Detaching reference and target tensors prevents these fixed teachers from receiving gradients. Both returned objectives are independent batch-mean scalars.",
}
