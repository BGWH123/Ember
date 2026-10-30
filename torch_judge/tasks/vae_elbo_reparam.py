"""VAE Reparameterization Trick & Beta-ELBO Loss task."""

TASK = {
    'title': 'VAE Reparameterization Trick & Beta-ELBO Loss',
    'title_zh': 'VAE 重参数化技巧与 Beta-ELBO 损失',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'vae_forward_loss',
    'starter_code': """def vae_forward_loss(x, recon_x, mu, logvar, eps_noise, beta=1.0):
    pass""",
    'hint': """Sample with mu + exp(0.5*logvar)*eps_noise. Sum reconstruction and KL terms per sample, then average over the batch.""",
    'hint_zh': '用 mu + exp(0.5*logvar)*eps_noise 采样。重建项和 KL 项先按样本求和，再对批次取平均。',
    'description_en': """Apply the VAE reparameterization trick and compute a beta-weighted ELBO loss. Data tensors are (B,D), latent tensors are (B,Z), and the function returns (z,total_loss,kl_loss).""",
    'description_zh': '应用 VAE 重参数化技巧并计算 beta 加权 ELBO 损失。数据张量为 (B,D)，潜变量张量为 (B,Z)，返回 (z,total_loss,kl_loss)。',
    'theory_en': """Writing z=mu+sigma*epsilon moves randomness into parameter-free epsilon, allowing gradients through mu and log-variance. The KL from a diagonal Gaussian to N(0,I) has a closed form.""",
    'theory_zh': '写成 z=mu+sigma*epsilon 可将随机性移到无参数的 epsilon，使梯度能通过 mu 和对数方差。对角高斯到 N(0,I) 的 KL 有闭式解。',
    'diagram_en': r"""```text
(mu, logvar, epsilon) -> reparameterize -> z
(x, reconstruction) -> MSE --\
Gaussian parameters -> KL -----+-> recon + beta*KL
```""",
    'diagram_zh': r"""```text
(mu, logvar, epsilon) -> 重参数化 -> z
(x, 重建结果) -> MSE --\
高斯参数 -> KL --------+-> 重建 + beta*KL
```""",
    "tests": [
        {
            "name": 'output shapes',
            "code": """import torch
x, r = torch.randn(4,6), torch.randn(4,6)
mu, lv, e = torch.randn(4,3), torch.randn(4,3), torch.randn(4,3)
z, total, kl = {fn}(x, r, mu, lv, e)
assert z.shape == (4,3) and total.shape == torch.Size([]) and kl.shape == torch.Size([])""",
        },
        {
            "name": 'exact numerical reference',
            "code": """import torch
torch.manual_seed(3)
x,r,mu,lv,e = [torch.randn(2,5) for _ in range(5)]
z,total,kl = {fn}(x,r,mu,lv,e,beta=.4)
rz = mu+torch.exp(.5*lv)*e
rr=((r-x)**2).sum(-1).mean()
rk=-.5*(1+lv-mu.pow(2)-lv.exp()).sum(-1).mean()
assert torch.equal(z,rz) and torch.equal(kl,rk) and torch.equal(total,rr+.4*rk)""",
        },
        {
            "name": 'standard normal edge case',
            "code": """import torch
x = torch.zeros(3,4); r=x.clone(); mu=torch.zeros(3,2); lv=torch.zeros_like(mu); e=torch.ones_like(mu)
z,total,kl = {fn}(x,r,mu,lv,e,beta=9.0)
assert torch.equal(z,e) and kl.item() == 0.0 and total.item() == 0.0""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x=torch.randn(2,4); r=torch.randn(2,4,requires_grad=True)
mu=torch.randn(2,3,requires_grad=True); lv=torch.randn(2,3,requires_grad=True); e=torch.randn(2,3)
z,total,kl={fn}(x,r,mu,lv,e)
(z.sum()+total+kl).backward()
assert r.grad is not None and mu.grad is not None and lv.grad is not None""",
        },
    ],
    'solution': '''def vae_forward_loss(x, recon_x, mu, logvar, eps_noise, beta=1.0):
    """实现VAE 重参数化技巧与 Beta-ELBO 损失（VAE Reparameterization Trick & Beta-ELBO Loss）。"""
    # Step 1: 由 log-variance 恢复标准差并重参数化采样，同时累计逐样本重建误差
    z = mu + torch.exp(0.5 * logvar) * eps_noise
    recon_loss = ((recon_x - x) ** 2).sum(dim=-1).mean()
    # Step 2: 用对角高斯闭式公式计算 q(z|x) 到标准正态先验的 KL
    kl_loss = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    total_loss = recon_loss + beta * kl_loss
    # Step 3: 以 beta 加权 KL 组成负 ELBO，返回潜样本和两个批均值标量
    return (z, total_loss, kl_loss)''',
    'demo': """torch.manual_seed(42)
x = torch.randn(8, 16)
mu = torch.randn(8, 4)
logvar = torch.randn(8, 4)
z, total_loss, kl_loss = vae_forward_loss(x, torch.randn_like(x), mu, logvar, torch.randn_like(mu))
print("Latent shape:", z.shape)
print("Total / KL loss:", total_loss.item(), kl_loss.item())""",
    'explanation': r"""### 核心思路
编码器给出 $q_\phi(z|x)=\mathcal N(\mu,\operatorname{diag}(e^{\text{logvar}}))$。重参数化 $z=\mu+e^{\frac12\text{logvar}}\odot\epsilon$ 把随机性移到无参数噪声 $\epsilon$，使路径可对 `mu`、`logvar` 求导。优化目标为 $\mathcal L=\|x-\hat x\|_2^2+\beta D_{KL}(q_\phi(z|x)\|\mathcal N(0,I))$。

### 代码步骤拆解
- **Step 1（可微潜变量采样）**：对 `(B,Z)` 的 `logvar` 取 `exp(0.5*logvar)` 得标准差，与 `eps_noise` 相乘后加 `mu`；重建误差先沿数据维 `D` 求和，再对批次平均。
- **Step 2（闭式 KL）**：逐潜维计算 $-\frac12(1+\log\sigma^2-\mu^2-\sigma^2)$，沿 `Z` 求和并对 `B` 取均值。
- **Step 3（Beta-ELBO 汇总）**：`total_loss=recon_loss+beta*kl_loss`，同时返回 `(B,Z)` 的 `z` 和两个零维损失张量。

### 重点与易错点分析
- **注意点 1（方差与标准差）**：`logvar` 是 $\log\sigma^2$，故标准差为 $\exp(0.5\,\text{logvar})$；遗漏 0.5 会把方差误当标准差。
- **注意点 2（归约约定）**：本题要求每个样本内求和、批间取均值；直接对所有元素 `.mean()` 会额外除以 $D$ 或 $Z$，改变重建项与 KL 的相对权重。
- **注意点 3（beta 的作用）**：$\beta>1$ 强化先验约束、常促进解耦但可能损害重建；当 `mu=0, logvar=0` 时 KL 必须精确为 0。

### 复杂度分析
- **时间复杂度**：重建与潜变量操作均为逐元素归约，总计 $O(B(D+Z))$。
- **空间复杂度**：返回的 `z` 为 $O(BZ)$；自动微分需保留重建和潜变量中间量，合计 $O(B(D+Z))$。""",
}
