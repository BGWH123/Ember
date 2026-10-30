"""AWQ salience scaling and GPTQ compensation task."""

TASK = {
    "title": "AWQ Salience Scaling & GPTQ Hessian Error Compensation",
    "title_zh": "AWQ 激活显著度缩放与 GPTQ 黑塞误差补偿",
    "difficulty": "Hard",
    "category": "量化",
    "function_name": "awq_and_gptq_step",
    "starter_code": '''import torch

def awq_and_gptq_step(W, X, alpha=0.5, q_col=0, H_inv=None, scale_int=0.1, eps=1e-6):
    pass
''',
    "hint": "AWQ uses mean absolute activation salience. For GPTQ, quantize one original W column and distribute its Hessian-normalized error only into columns to its right.",
    "hint_zh": "AWQ 使用激活绝对值均值作为显著度；GPTQ 量化 W 的一个原始列，并仅将黑塞归一化误差分配到其右侧列。",
    "description_en": "Perform AWQ channel scaling and one GPTQ/OBS column quantization step with inverse-Hessian error compensation.",
    "description_zh": "执行 AWQ 通道缩放，以及带逆黑塞误差补偿的一步 GPTQ/OBS 列量化。",
    "theory_en": "AWQ protects input channels with salient activations through a function-preserving weight/activation rescaling. GPTQ uses second-order curvature to compensate future columns for the error introduced by quantizing the current column.",
    "theory_zh": "AWQ 通过保持函数不变的权重/激活反向缩放保护高显著度输入通道。GPTQ 使用二阶曲率，让后续列补偿当前列的量化误差。",
    "diagram_en": "```text\nX -> mean |.| -> s -> W*s and X/s\nW[:,q] -> round/clamp -> error/H_inv[q,q] -> compensate W[:,q+1:]\n```",
    "diagram_zh": "```text\nX -> |.| 均值 -> s -> W*s 与 X/s\nW[:,q] -> 舍入/截断 -> 误差/H_inv[q,q] -> 补偿 W[:,q+1:]\n```",
    "tests": [
        {
            "name": "AWQ and GPTQ shapes",
            "code": '''import torch
W = torch.randn(5,4); X = torch.randn(8,4)
s, wa, wg = {fn}(W,X,H_inv=torch.eye(4),q_col=1)
assert s.shape == (4,) and wa.shape == wg.shape == W.shape
''',
        },
        {
            "name": "exact salience and compensation",
            "code": '''import torch
W = torch.tensor([[.16,-.24,.7],[.35,.81,-.2]],dtype=torch.double)
X = torch.tensor([[1.,-2.,3.],[-3.,4.,1.]],dtype=torch.double)
H = torch.tensor([[2.,.1,.2],[.1,4.,.6],[.2,.6,3.]],dtype=torch.double)
s, wa, wg = {fn}(W,X,alpha=.5,q_col=1,H_inv=H,scale_int=.1,eps=1e-9)
rs = X.abs().mean(0).clamp(min=1e-9)**.5; rw = W*rs
hat = torch.clamp(torch.round(W[:,1]/.1),-8.,7.)*.1
err = (W[:,1]-hat)/H[1,1]
rg = W.clone(); rg[:,1]=hat; rg[:,2:]=W[:,2:]-err[:,None]*H[1,2:][None,:]
assert torch.allclose(s,rs,atol=1e-12,rtol=1e-12) and torch.allclose(wa,rw,atol=1e-12,rtol=1e-12)
assert torch.allclose(wg,rg,atol=1e-12,rtol=1e-12)
''',
        },
        {
            "name": "last-column and inferred Hessian edge",
            "code": '''import torch
W = torch.tensor([[.04,.76],[-.14,-.86]])
X = torch.tensor([[1.,0.],[0.,2.],[1.,1.]])
s, wa, wg = {fn}(W,X,q_col=1,H_inv=None,scale_int=.1)
assert torch.equal(wg[:,:1],W[:,:1])
assert torch.allclose(wg[:,1],torch.tensor([.7,-.8]))
assert torch.isfinite(s).all() and torch.isfinite(wa).all()
''',
        },
        {
            "name": "gradient through AWQ and compensation",
            "code": '''import torch
W = torch.randn(3,4,requires_grad=True); X = (torch.randn(6,4)+.2).requires_grad_()
H = torch.eye(4)
s, wa, wg = {fn}(W,X,q_col=1,H_inv=H)
(s.sum()+wa.square().sum()+wg[:,2:].square().sum()).backward()
assert s.requires_grad and wa.requires_grad and wg.requires_grad
assert W.grad is not None and X.grad is not None
assert torch.isfinite(W.grad).all() and torch.isfinite(X.grad).all()
''',
        },
    ],
    "solution": '''import torch

def awq_and_gptq_step(W, X, alpha=0.5, q_col=0, H_inv=None, scale_int=0.1, eps=1e-6):
    s = X.abs().mean(dim=0).clamp(min=eps) ** alpha
    W_awq = W * s.unsqueeze(0)
    X_awq = X / s.unsqueeze(0)
    if H_inv is None:
        identity = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        H_inv = torch.linalg.inv(X.T @ X + eps * identity)
    w_q = W[:, q_col]
    w_q_hat = torch.clamp(torch.round(w_q / scale_int), -8.0, 7.0) * scale_int
    err = (w_q - w_q_hat) / H_inv[q_col, q_col]
    W_gptq = W.clone()
    W_gptq[:, q_col] = w_q_hat
    W_gptq[:, q_col + 1:] = W[:, q_col + 1:] - err.unsqueeze(1) * H_inv[q_col, q_col + 1:].unsqueeze(0)
    return s, W_awq, W_gptq
''',
    "demo": '''W, X = torch.randn(64,128), torch.randn(256,128)
H_inv = torch.linalg.inv(X.T@X + 1e-4*torch.eye(128))
s, W_awq, W_gptq = awq_and_gptq_step(W,X,q_col=0,H_inv=H_inv)
print(s.shape, W_awq.shape, W_gptq.shape)
''',
    "explanation": "X_awq is the inverse activation rescaling that makes W_awq function-preserving, though only the requested salience and weight results are returned. If H_inv is omitted, a ridge-stabilized calibration Hessian inverse is formed.",
}
