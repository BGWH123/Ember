"""Diffusion-DPO & REPA Representation Alignment Loss task."""

TASK = {
    'title': 'Diffusion-DPO & REPA Representation Alignment Loss',
    'title_zh': 'Diffusion-DPO 与 REPA 表征对齐损失',
    'difficulty': 'Hard',
    'category': '扩散与流模型',
    'function_name': 'diffusion_dpo_and_repa',
    'starter_code': """def diffusion_dpo_and_repa(eps_theta_w, eps_ref_w, eps_w, eps_theta_l, eps_ref_l, eps_l, hidden_proj, ssl_target, beta=500.0, eps=1e-8):
    pass""",
    'hint': """Compute four per-sample noise MSEs, compare policy-minus-reference errors for winner and loser through -logsigmoid, then add a separate negative cosine-similarity REPA calculation.""",
    'hint_zh': '计算四个逐样本噪声 MSE，通过 -logsigmoid 比较胜者与败者的策略减参考误差，再单独计算负余弦相似度 REPA。',
    'description_en': """Compute Diffusion-DPO preference loss from winner/loser noise predictions of shape (B,D), and REPA alignment from projected and SSL features (B,N,C). Reference predictions and SSL targets are detached. Return (dpo_loss,repa_loss).""",
    'description_zh': """从形状 (B,D) 的胜者/败者噪声预测计算 Diffusion-DPO 偏好损失，并从投影特征与 SSL 特征 (B,N,C) 计算 REPA 对齐。参考预测与 SSL 目标需停止梯度。返回 (dpo_loss,repa_loss)。""",
    'theory_en': """Diffusion-DPO favors a policy whose denoising error improves over the reference more for preferred samples than rejected ones. REPA maximizes tokenwise cosine similarity to frozen self-supervised representations.""",
    'theory_zh': 'Diffusion-DPO 鼓励策略相对参考模型在偏好样本上的去噪误差改善大于拒绝样本。REPA 最大化 token 级特征与冻结自监督表征的余弦相似度。',
    'diagram_en': """```text
winner/loser policy and reference errors -> relative margin -> -log sigmoid -> DPO
hidden projection + detached SSL target -> normalize -> negative cosine -> REPA
```""",
    'diagram_zh': """```text
胜/负样本策略与参考误差 -> 相对间隔 -> -log sigmoid -> DPO
隐藏投影 + 停止梯度的 SSL 目标 -> 归一化 -> 负余弦 -> REPA
```""",
    "tests": [
        {
            "name": 'scalar output shapes',
            "code": """import torch
noise=[torch.randn(3,5) for _ in range(6)]
h=torch.randn(3,4,7); z=torch.randn(3,4,7)
d,r={fn}(*noise,h,z)
assert d.shape==torch.Size([]) and r.shape==torch.Size([])
assert torch.isfinite(d) and torch.isfinite(r)""",
        },
        {
            "name": 'exact numerical reference',
            "code": """import torch
import torch.nn.functional as F
torch.manual_seed(6)
tw,rw,w,tl,rl,l=[torch.randn(2,4) for _ in range(6)]
h=torch.randn(2,3,5); z=torch.randn(2,3,5)
d,r={fn}(tw,rw,w,tl,rl,l,h,z,beta=2.,eps=1e-6)
etw=((tw-w)**2).mean(-1); erw=((rw.detach()-w)**2).mean(-1)
etl=((tl-l)**2).mean(-1); erl=((rl.detach()-l)**2).mean(-1)
rd=-F.logsigmoid(-2.*((etw-erw)-(etl-erl))).mean()
rr=-((h/(h.norm(dim=-1,keepdim=True)+1e-6))*(z.detach()/(z.detach().norm(dim=-1,keepdim=True)+1e-6))).sum(-1).mean()
assert torch.equal(d,rd) and torch.equal(r,rr)""",
        },
        {
            "name": 'zero and identical feature edge cases',
            "code": """import torch
q=torch.zeros(2,3); h=torch.ones(2,4,5); zero=torch.zeros_like(h)
d,r={fn}(q,q,q,q,q,q,h,zero,beta=500.)
assert torch.allclose(d,torch.tensor(0.69314718),atol=1e-7)
assert r.item()==0.0 and torch.isfinite(r)""",
        },
        {
            "name": 'gradient routing',
            "code": """import torch
tw=torch.randn(2,3,requires_grad=True); rw=torch.randn(2,3,requires_grad=True); w=torch.randn(2,3)
tl=torch.randn(2,3,requires_grad=True); rl=torch.randn(2,3,requires_grad=True); l=torch.randn(2,3)
h=torch.randn(2,4,5,requires_grad=True); z=torch.randn(2,4,5,requires_grad=True)
d,r={fn}(tw,rw,w,tl,rl,l,h,z,beta=3.); (d+r).backward()
assert tw.grad is not None and tl.grad is not None and h.grad is not None
assert rw.grad is None and rl.grad is None and z.grad is None""",
        },
    ],
    'solution': '''def diffusion_dpo_and_repa(eps_theta_w, eps_ref_w, eps_w, eps_theta_l, eps_ref_l, eps_l, hidden_proj, ssl_target, beta=500.0, eps=1e-8):
    """计算扩散 DPO 与表征对齐损失 (Diffusion-DPO and REPA Loss)。"""
    # Step 1: 计算胜负样本中策略与参考模型的逐样本去噪误差
    policy_error_w = (eps_theta_w - eps_w).square().mean(dim=-1)
    reference_error_w = (eps_ref_w.detach() - eps_w).square().mean(dim=-1)
    policy_error_l = (eps_theta_l - eps_l).square().mean(dim=-1)
    reference_error_l = (eps_ref_l.detach() - eps_l).square().mean(dim=-1)

    # Step 2: 用相对误差间隔构造稳定的偏好逻辑损失
    relative_w = policy_error_w - reference_error_w
    relative_l = policy_error_l - reference_error_l
    dpo_loss = -F.logsigmoid(-beta * (relative_w - relative_l)).mean()

    # Step 3: 归一化投影与固定 SSL 目标并计算负余弦相似度
    hidden_norm = hidden_proj / (hidden_proj.norm(dim=-1, keepdim=True) + eps)
    target_fixed = ssl_target.detach()
    target_norm = target_fixed / (target_fixed.norm(dim=-1, keepdim=True) + eps)
    repa_loss = -(hidden_norm * target_norm).sum(dim=-1).mean()
    return dpo_loss, repa_loss''',
    'demo': """torch.manual_seed(42)
noise_tensors = [torch.randn(4, 32) for _ in range(6)]
hidden_proj = torch.randn(4, 16, 64)
ssl_target = torch.randn_like(hidden_proj)
dpo_loss, repa_loss = diffusion_dpo_and_repa(*noise_tensors, hidden_proj, ssl_target)
print("DPO loss:", dpo_loss.item())
print("REPA loss:", repa_loss.item())""",
    'explanation': """### 核心思路
Diffusion-DPO 比较策略相对参考模型在胜者和败者上的去噪误差改善，并通过 `logsigmoid` 优化偏好间隔；REPA 最大化投影特征与冻结 SSL 特征的 token 级余弦相似度。

### 代码步骤拆解
- **Step 1（去噪误差）**：对四组 `(B, D)` 预测沿特征维计算 MSE，得到逐样本误差向量。
- **Step 2（偏好目标）**：比较 `relative_w` 与 `relative_l`，经缩放和 `logsigmoid` 得到标量 `dpo_loss`。
- **Step 3（表征对齐）**：将 `(B, N, C)` 特征沿 `C` 维归一化，负余弦相似度均值构成 `repa_loss`。

### 重点与易错点分析
- **注意点 1（梯度截断）**：参考噪声预测和 `ssl_target` 必须 `.detach()`，固定教师不应接收梯度。
- **注意点 2（数值稳定性）**：使用 `F.logsigmoid` 比 `log(sigmoid(...))` 更稳定；向量范数分母加 `eps` 处理零特征。
- **注意点 3（符号方向）**：DPO 的负号与胜负差顺序共同决定优化方向，颠倒会鼓励败者误差改善更多。

### 复杂度分析
- **时间复杂度**：$O(BD+BNC)$，去噪误差和特征归一化均为线性扫描。
- **空间复杂度**：$O(B+BNC)$，归一化后的两组表征占主要显存。""",
}
