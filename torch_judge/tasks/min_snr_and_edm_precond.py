"""Min-SNR weighting and EDM preconditioning task."""

TASK = {
    "title": "Diffusion Min-SNR-Gamma Weighting & EDM Preconditioning",
    "title_zh": "扩散模型 Min-SNR-γ 损失加权与 EDM 预处理",
    "difficulty": "Medium",
    "category": "扩散与流模型",
    "function_name": "min_snr_and_edm_precond",
    "starter_code": '''def min_snr_and_edm_precond(alpha_bar_t, sigma, gamma=5.0, sigma_data=0.5, eps=1e-8):
    pass
''',
    "hint": "Compute the schedule SNR first, cap it at gamma for the loss weight, and reuse sigma squared plus sigma_data squared for the EDM coefficients.",
    "hint_zh": "先计算调度信噪比并将损失权重中的信噪比截断到 gamma，再复用 sigma 平方与 sigma_data 平方之和计算 EDM 系数。",
    "description_en": "Given cumulative signal retention alpha_bar_t and EDM noise levels sigma, return the Min-SNR-gamma loss weight followed by c_skip, c_out, c_in, and c_noise. Both inputs have shape (B,).",
    "description_zh": "给定累计信号保留率 alpha_bar_t 和 EDM 噪声水平 sigma，返回 Min-SNR-γ 损失权重以及 c_skip、c_out、c_in、c_noise。两个输入形状均为 (B,)。",
    "theory_en": "Min-SNR weighting prevents high-SNR timesteps from dominating diffusion training. EDM rescales the noisy input, residual output, skip path, and noise embedding according to sigma and the data standard deviation.",
    "theory_zh": "Min-SNR 加权避免高信噪比时间步主导扩散训练。EDM 根据 sigma 与数据标准差缩放含噪输入、残差输出、跳连路径和噪声嵌入。",
    "diagram_en": "```text\nalpha_bar -> SNR -> min(SNR,gamma)/SNR\nsigma,sigma_data -> shared denominator -> c_skip,c_out,c_in,c_noise\n```",
    "diagram_zh": "```text\nalpha_bar -> SNR -> min(SNR,gamma)/SNR\nsigma、sigma_data -> 公共分母 -> c_skip,c_out,c_in,c_noise\n```",
    "tests": [
        {
            "name": "output shapes",
            "code": '''import torch
a=torch.tensor([.1,.5,.9]); s=torch.tensor([.2,.7,1.5])
out={fn}(a,s)
assert len(out)==5 and all(x.shape==(3,) for x in out)
assert all(torch.isfinite(x).all() for x in out)
''',
        },
        {
            "name": "exact reference math",
            "code": '''import torch
a=torch.tensor([.2,.8]); s=torch.tensor([.25,2.])
got={fn}(a,s,gamma=2.0,sigma_data=.5)
snr=a/(1-a).clamp(min=1e-8)
d=torch.sqrt(s**2+.5**2)
ref=(torch.minimum(snr,torch.full_like(snr,2.))/snr,.5**2/(s**2+.5**2),s*.5/d,1/d,.25*torch.log(s.clamp(min=1e-8)))
assert all(torch.equal(x,y) for x,y in zip(got,ref))
''',
        },
        {
            "name": "gamma cap edge cases",
            "code": '''import torch
a=torch.tensor([.5,.99]); s=torch.tensor([1e-12,1.])
w,cs,co,ci,cn={fn}(a,s,gamma=1.0,eps=1e-6)
assert torch.allclose(w,torch.tensor([1.,1/99.]),atol=1e-6)
assert torch.isfinite(cs).all() and torch.isfinite(co).all() and torch.isfinite(ci).all() and torch.isfinite(cn).all()
assert torch.equal(cn[:1],torch.tensor([.25*torch.log(torch.tensor(1e-6))]))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
a=torch.tensor([.2,.8],requires_grad=True); s=torch.tensor([.3,1.2],requires_grad=True)
out={fn}(a,s,gamma=3.0); sum(x.sum() for x in out).backward()
assert a.grad is not None and s.grad is not None
assert torch.isfinite(a.grad).all() and torch.isfinite(s.grad).all()
''',
        },
    ],
    "solution": '''import torch

def min_snr_and_edm_precond(alpha_bar_t, sigma, gamma=5.0, sigma_data=0.5, eps=1e-8):
    snr = alpha_bar_t / (1.0 - alpha_bar_t).clamp(min=eps)
    min_snr_weight = torch.minimum(
        snr, torch.full_like(snr, float(gamma))
    ) / snr
    denom_sq = sigma ** 2 + sigma_data ** 2
    denom = torch.sqrt(denom_sq)
    c_skip = (sigma_data ** 2) / denom_sq
    c_out = (sigma * sigma_data) / denom
    c_in = 1.0 / denom
    c_noise = 0.25 * torch.log(sigma.clamp(min=eps))
    return min_snr_weight, c_skip, c_out, c_in, c_noise
''',
    "demo": '''alpha_bar_t = torch.tensor([0.1, 0.5, 0.9])
sigma = torch.tensor([0.1, 1.0, 10.0])
print(min_snr_and_edm_precond(alpha_bar_t, sigma))
''',
    "explanation": "The SNR cap only changes high-SNR examples. All EDM coefficients are elementwise, with clamping used only for numerically safe logarithms and schedule division.",
}
