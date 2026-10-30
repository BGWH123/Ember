"""Flow-GRPO reverse SDE transition and KL task."""

TASK = {
    "title": "Flow-GRPO Reverse SDE Step & Velocity-Space KL",
    "title_zh": "Flow-GRPO 逆向 SDE 采样步与速度场 KL",
    "difficulty": "Hard",
    "category": "扩散与流模型",
    "function_name": "flow_grpo_sde_step",
    "starter_code": '''def flow_grpo_sde_step(x_t, v_theta, v_ref, t, dt, noise, sigma_a=0.7, eps=1e-6):
    pass
''',
    "hint": "Build policy and detached-reference reverse drifts with the same scalar variance. Sample from the policy mean, detach that sample only inside its transition log probability, and use the equal-covariance Gaussian KL.",
    "hint_zh": "用相同标量方差构造策略与分离参考速度的逆向漂移。从策略均值采样，仅在转移对数概率中分离样本，并使用同协方差高斯 KL。",
    "description_en": "Take one reverse SDE step from t to t-dt using policy velocity v_theta, evaluate its isotropic Gaussian transition log probability, and compute the closed-form KL to a transition induced by detached v_ref.",
    "description_zh": "使用策略速度 v_theta 从 t 到 t-dt 执行一步逆向 SDE，计算其各向同性高斯转移对数概率，并计算与分离 v_ref 所诱导转移之间的闭式 KL。",
    "theory_en": "The stochastic transition has covariance g(t)^2 dt I. Equal policy and reference covariance reduces KL to half the squared mean displacement divided by the scalar variance. Detaching the sampled target gives a score-function-style log-probability gradient.",
    "theory_zh": "随机转移协方差为 g(t)^2 dt I。策略与参考协方差相同时，KL 化为均值差平方除以标量方差的一半。分离采样目标可得到类似得分函数的对数概率梯度。",
    "diagram_en": "```text\n(x_t,v_theta) -> reverse drift -> mean + std*noise -> x_next -> log_prob\n(x_t,stopgrad(v_ref)) -> reference mean ---------> equal-variance KL\n```",
    "diagram_zh": "```text\n(x_t,v_theta) -> 逆漂移 -> 均值 + std*噪声 -> x_next -> log_prob\n(x_t,stopgrad(v_ref)) -> 参考均值 -------------> 同方差 KL\n```",
    "tests": [
        {
            "name": "return shapes",
            "code": '''import torch
x=torch.randn(3,5); vt=torch.randn(3,5); vr=torch.randn(3,5); n=torch.randn(3,5)
xn,lp,kl={fn}(x,vt,vr,.8,.1,n)
assert xn.shape==(3,5) and lp.shape==(3,) and kl.shape==(3,)
assert torch.isfinite(xn).all() and torch.isfinite(lp).all() and torch.isfinite(kl).all()
''',
        },
        {
            "name": "exact transition reference",
            "code": '''import torch, math
x=torch.tensor([[1.,-2.]]); vt=torch.tensor([[.3,.4]]); vr=torch.tensor([[-.2,.1]]); n=torch.tensor([[.5,-1.]])
t=.6; dt=.05; sa=.7; eps=1e-6
xn,lp,kl={fn}(x,vt,vr,t,dt,n,sa,eps)
g2=sa**2*t/(1-t+eps); drift=vt+g2/(2*t)*(x+(1-t)*vt); mean=x-dt*drift
ref=x-dt*(vr+g2/(2*t)*(x+(1-t)*vr))
rx=mean+math.sqrt(g2*dt)*n
rlp=-.5*(((rx.detach()-mean)**2)/(g2*dt+eps)+math.log(2*math.pi*g2*dt)).sum(-1)
rkl=(.5*(mean-ref)**2/(g2*dt+eps)).sum(-1)
assert torch.equal(xn,rx) and torch.equal(lp,rlp) and torch.equal(kl,rkl)
''',
        },
        {
            "name": "matching velocities zero KL edge",
            "code": '''import torch
x=torch.randn(2,4); v=torch.randn(2,4); noise=torch.zeros_like(x)
xn,lp,kl={fn}(x,v,v,.5,.01,noise)
assert torch.equal(kl,torch.zeros(2))
assert torch.isfinite(lp).all() and not torch.equal(xn,x)
''',
        },
        {
            "name": "policy gradient and detached reference",
            "code": '''import torch
x=torch.randn(2,3,requires_grad=True); vt=torch.randn(2,3,requires_grad=True)
vr=torch.randn(2,3,requires_grad=True); n=torch.randn(2,3)
xn,lp,kl={fn}(x,vt,vr,.7,.1,n); (xn.sum()+lp.sum()+kl.sum()).backward()
assert x.grad is not None and vt.grad is not None and torch.isfinite(vt.grad).all()
assert vr.grad is None
''',
        },
    ],
    "solution": '''import torch
import math

def flow_grpo_sde_step(x_t, v_theta, v_ref, t, dt, noise, sigma_a=0.7, eps=1e-6):
    g2 = (sigma_a ** 2) * (t / (1.0 - t + eps))
    drift = v_theta + (g2 / (2.0 * t)) * (
        x_t + (1.0 - t) * v_theta
    )
    mean = x_t - dt * drift
    std = math.sqrt(g2 * dt)
    x_next = mean + std * noise
    log_prob = -0.5 * (
        ((x_next.detach() - mean) ** 2) / (g2 * dt + eps)
        + math.log(2.0 * math.pi * g2 * dt)
    ).sum(dim=-1)
    v_ref_detached = v_ref.detach()
    drift_ref = v_ref_detached + (g2 / (2.0 * t)) * (
        x_t + (1.0 - t) * v_ref_detached
    )
    mean_ref = x_t - dt * drift_ref
    kl = (
        0.5 * ((mean - mean_ref) ** 2) / (g2 * dt + eps)
    ).sum(dim=-1)
    return x_next, log_prob, kl
''',
    "demo": '''x = torch.randn(4, 16)
velocity = torch.randn_like(x, requires_grad=True)
reference = torch.randn_like(x)
noise = torch.randn_like(x)
print([y.shape for y in flow_grpo_sde_step(x, velocity, reference, 0.8, 0.05, noise)])
''',
    "explanation": "The policy and reference differ only through their velocity-induced means. The returned sample remains reparameterized, while its occurrence as the target in log_prob is detached.",
}
