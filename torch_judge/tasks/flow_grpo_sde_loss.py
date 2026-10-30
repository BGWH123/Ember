"""Flow-GRPO Reverse SDE Step & Velocity-Space KL task."""

TASK = {
    'title': 'Flow-GRPO Reverse SDE Step & Velocity-Space KL',
    'title_zh': 'Flow-GRPO 逆向 SDE 采样步与速度场 KL',
    'difficulty': 'Hard',
    'category': '扩散与流模型',
    'function_name': 'flow_grpo_sde_step',
    'starter_code': """def flow_grpo_sde_step(x_t, v_theta, v_ref, t, dt, noise, sigma_a=0.7, eps=1e-6):
    pass""",
    'hint': """Build policy and detached-reference reverse drifts with the same scalar variance. Sample from the policy mean, detach that sample only inside its transition log probability, and use the equal-covariance Gaussian KL.""",
    'hint_zh': '用相同标量方差构造策略与分离参考速度的逆向漂移。从策略均值采样，仅在转移对数概率中分离样本，并使用同协方差高斯 KL。',
    'description_en': """Take one reverse SDE step from t to t-dt using policy velocity v_theta, evaluate its isotropic Gaussian transition log probability, and compute the closed-form KL to a transition induced by detached v_ref.""",
    'description_zh': '使用策略速度 v_theta 从 t 到 t-dt 执行一步逆向 SDE，计算其各向同性高斯转移对数概率，并计算与分离 v_ref 所诱导转移之间的闭式 KL。',
    'theory_en': """The stochastic transition has covariance g(t)^2 dt I. Equal policy and reference covariance reduces KL to half the squared mean displacement divided by the scalar variance. Detaching the sampled target gives a score-function-style log-probability gradient.""",
    'theory_zh': '随机转移协方差为 g(t)^2 dt I。策略与参考协方差相同时，KL 化为均值差平方除以标量方差的一半。分离采样目标可得到类似得分函数的对数概率梯度。',
    'diagram_en': """```text
(x_t,v_theta) -> reverse drift -> mean + std*noise -> x_next -> log_prob
(x_t,stopgrad(v_ref)) -> reference mean ---------> equal-variance KL
```""",
    'diagram_zh': """```text
(x_t,v_theta) -> 逆漂移 -> 均值 + std*噪声 -> x_next -> log_prob
(x_t,stopgrad(v_ref)) -> 参考均值 -------------> 同方差 KL
```""",
    "tests": [
        {
            "name": 'return shapes',
            "code": """import torch
x=torch.randn(3,5); vt=torch.randn(3,5); vr=torch.randn(3,5); n=torch.randn(3,5)
xn,lp,kl={fn}(x,vt,vr,.8,.1,n)
assert xn.shape==(3,5) and lp.shape==(3,) and kl.shape==(3,)
assert torch.isfinite(xn).all() and torch.isfinite(lp).all() and torch.isfinite(kl).all()""",
        },
        {
            "name": 'exact transition reference',
            "code": """import torch, math
x=torch.tensor([[1.,-2.]]); vt=torch.tensor([[.3,.4]]); vr=torch.tensor([[-.2,.1]]); n=torch.tensor([[.5,-1.]])
t=.6; dt=.05; sa=.7; eps=1e-6
xn,lp,kl={fn}(x,vt,vr,t,dt,n,sa,eps)
g2=sa**2*t/(1-t+eps); drift=vt+g2/(2*t)*(x+(1-t)*vt); mean=x-dt*drift
ref=x-dt*(vr+g2/(2*t)*(x+(1-t)*vr))
rx=mean+math.sqrt(g2*dt)*n
rlp=-.5*(((rx.detach()-mean)**2)/(g2*dt+eps)+math.log(2*math.pi*g2*dt)).sum(-1)
rkl=(.5*(mean-ref)**2/(g2*dt+eps)).sum(-1)
assert torch.equal(xn,rx) and torch.equal(lp,rlp) and torch.equal(kl,rkl)""",
        },
        {
            "name": 'matching velocities zero KL edge',
            "code": """import torch
x=torch.randn(2,4); v=torch.randn(2,4); noise=torch.zeros_like(x)
xn,lp,kl={fn}(x,v,v,.5,.01,noise)
assert torch.equal(kl,torch.zeros(2))
assert torch.isfinite(lp).all() and not torch.equal(xn,x)""",
        },
        {
            "name": 'policy gradient and detached reference',
            "code": """import torch
x=torch.randn(2,3,requires_grad=True); vt=torch.randn(2,3,requires_grad=True)
vr=torch.randn(2,3,requires_grad=True); n=torch.randn(2,3)
xn,lp,kl={fn}(x,vt,vr,.7,.1,n); (xn.sum()+lp.sum()+kl.sum()).backward()
assert x.grad is not None and vt.grad is not None and torch.isfinite(vt.grad).all()
assert vr.grad is None""",
        },
    ],
    'solution': '''def flow_grpo_sde_step(x_t, v_theta, v_ref, t, dt, noise, sigma_a=0.7, eps=1e-6):
    """执行 Flow-GRPO 逆向 SDE 步与速度空间 KL (Flow-GRPO SDE Step)。"""
    # Step 1: 由策略速度构造逆向漂移、均值并重参数化采样
    diffusion_var = sigma_a ** 2 * t / (1.0 - t + eps)
    policy_drift = v_theta + diffusion_var / (2.0 * t) * (x_t + (1.0 - t) * v_theta)
    policy_mean = x_t - dt * policy_drift
    transition_var = diffusion_var * dt
    x_next = policy_mean + math.sqrt(transition_var) * noise

    # Step 2: 将采样目标停止梯度后计算各向同性高斯转移对数概率
    log_prob = -0.5 * (
        (x_next.detach() - policy_mean).square() / (transition_var + eps)
        + math.log(2.0 * math.pi * transition_var)
    ).sum(dim=-1)

    # Step 3: 构造停止梯度的参考均值并计算同协方差高斯 KL
    reference_velocity = v_ref.detach()
    reference_drift = reference_velocity + diffusion_var / (2.0 * t) * (
        x_t + (1.0 - t) * reference_velocity
    )
    reference_mean = x_t - dt * reference_drift
    kl = (0.5 * (policy_mean - reference_mean).square() / (transition_var + eps)).sum(dim=-1)
    return x_next, log_prob, kl''',
    'demo': """torch.manual_seed(42)
x_t = torch.randn(4, 16)
velocity = torch.randn_like(x_t, requires_grad=True)
reference = torch.randn_like(x_t)
noise = torch.randn_like(x_t)
x_next, log_prob, kl = flow_grpo_sde_step(x_t, velocity, reference, 0.8, 0.05, noise)
print("Next state shape:", x_next.shape)
print("Log-probability shape:", log_prob.shape)
print("Mean KL:", kl.mean().item())""",
    'explanation': r"""### 核心思路
策略速度决定逆向 SDE 的高斯转移均值，方差为 $g(t)^2\Delta t$；从策略转移重参数化采样后，计算其对数概率及与固定参考速度转移的同协方差 KL。

### 代码步骤拆解
- **Step 1（策略转移）**：由 `(B, D)` 的 `v_theta` 计算 `policy_mean`，再加标准差乘 `noise` 得到 `x_next`。
- **Step 2（转移对数概率）**：将样本作为目标停止梯度，沿特征维求和得到 `(B,)` 的 `log_prob`。
- **Step 3（参考 KL）**：用停止梯度的 `v_ref` 构造 `reference_mean`，按均值差平方除共同方差计算逐样本 KL。

### 重点与易错点分析
- **注意点 1（梯度路径）**：`x_next` 本身保持重参数化梯度，但在 `log_prob` 的观测位置必须 `.detach()`；`v_ref` 全程固定。
- **注意点 2（时间边界）**：公式含 $1/t$ 与 $1/(1-t)$，实际采样应避开精确的 $t=0$、$t=1$ 退化端点。
- **注意点 3（同方差简化）**：KL 只需均值差项；若策略与参考方差不同，还必须加入方差比和对数行列式项。

### 复杂度分析
- **时间复杂度**：$O(BD)$，漂移、对数概率与 KL 都是逐元素计算和归约。
- **空间复杂度**：$O(BD)$，主要保存策略与参考均值。""",
}
