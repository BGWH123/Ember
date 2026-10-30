"""JEPA Latent Prediction & VICReg Anti-Collapse Regularization task."""

TASK = {
    'title': 'JEPA Latent Prediction & VICReg Anti-Collapse Regularization',
    'title_zh': 'JEPA 潜空间预测与 VICReg 防坍缩正则损失',
    'difficulty': 'Hard',
    'category': '世界模型',
    'function_name': 'jepa_vicreg_loss',
    'starter_code': """def jepa_vicreg_loss(pred_z, target_z, sim_coef=25.0, std_coef=25.0, cov_coef=1.0, gamma=1.0, eps=1e-4):
    pass""",
    'hint': """Detach the target for invariance. Compute population standard deviation for the variance hinge, but divide the centered Gram matrix by B-1 for covariance and remove its diagonal.""",
    'hint_zh': '在不变性项中分离目标。方差铰链使用总体标准差，但协方差用中心化 Gram 矩阵除以 B-1，并移除对角线。',
    'description_en': """Combine detached-target JEPA prediction MSE with VICReg variance and covariance anti-collapse penalties for latent batches pred_z and target_z of shape (B,D), with B at least two.""",
    'description_zh': '对形状为 (B,D)、B 至少为 2 的潜变量批次 pred_z 和 target_z，将分离目标的 JEPA 预测 MSE 与 VICReg 方差、协方差防坍缩惩罚结合。',
    'theory_en': """Invariance aligns predictions to target representations. A standard-deviation hinge prevents constant features, while squared off-diagonal covariance discourages redundant dimensions without forcing individual samples apart.""",
    'theory_zh': '不变性项将预测对齐目标表示；标准差铰链防止恒定特征，非对角协方差平方则抑制冗余维度，而无需强制分离单个样本。',
    'diagram_en': """```text
pred,target(stopgrad) -> invariance MSE
pred -> per-dim std hinge + off-diagonal covariance -> total weighted loss
```""",
    'diagram_zh': """```text
预测、目标(stopgrad) -> 不变性 MSE
预测 -> 逐维标准差铰链 + 非对角协方差 -> 加权总损失
```""",
    "tests": [
        {
            "name": 'scalar output shapes',
            "code": """import torch
p=torch.randn(5,7); t=torch.randn(5,7)
out={fn}(p,t)
assert len(out)==4 and all(x.shape==torch.Size([]) for x in out)
assert all(torch.isfinite(x) for x in out)""",
        },
        {
            "name": 'exact VICReg reference',
            "code": """import torch
import torch.nn.functional as F
p=torch.tensor([[1.,2.],[3.,0.],[2.,4.]]); t=torch.tensor([[0.,1.],[2.,1.],[2.,3.]])
got={fn}(p,t,2.,3.,4.,.8,1e-4)
inv=((p-t)**2).mean(); std=torch.sqrt(p.var(0,unbiased=False)+1e-4); var=F.relu(.8-std).mean()
zc=p-p.mean(0,keepdim=True); cov=zc.T@zc/(p.size(0)-1); off=cov-torch.diag(torch.diag(cov)); cv=(off**2).sum()/p.size(1)
ref=(2*inv+3*var+4*cv,inv,var,cv)
assert all(torch.equal(x,y) for x,y in zip(got,ref))""",
        },
        {
            "name": 'collapsed representation edge',
            "code": """import torch
p=torch.zeros(4,3); t=torch.zeros_like(p)
total,inv,var,cov={fn}(p,t,sim_coef=1.,std_coef=1.,cov_coef=1.,gamma=1.,eps=1e-4)
assert inv.item()==0.0 and cov.item()==0.0
assert torch.allclose(var,torch.tensor(.99)) and torch.equal(total,var)""",
        },
        {
            "name": 'prediction-only gradient flow',
            "code": """import torch
p=torch.randn(6,4,requires_grad=True); t=torch.randn(6,4,requires_grad=True)
total,inv,var,cov={fn}(p,t); total.backward()
assert p.grad is not None and torch.isfinite(p.grad).all()
assert t.grad is None""",
        },
    ],
    'solution': '''def jepa_vicreg_loss(pred_z, target_z, sim_coef=25.0, std_coef=25.0, cov_coef=1.0, gamma=1.0, eps=1e-4):
    """计算 JEPA 预测与 VICReg 防坍缩损失 (JEPA-VICReg Loss)。"""
    # Step 1: 对齐预测与固定目标表示
    inv_loss = ((pred_z - target_z.detach()) ** 2).mean()

    # Step 2: 用标准差铰链避免特征坍缩
    feature_std = torch.sqrt(pred_z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - feature_std).mean()

    # Step 3: 惩罚特征间的非对角协方差
    B, D = pred_z.shape
    centered = pred_z - pred_z.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (B - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    cov_loss = off_diagonal.square().sum() / D
    total_loss = sim_coef * inv_loss + std_coef * var_loss + cov_coef * cov_loss
    return total_loss, inv_loss, var_loss, cov_loss''',
    'demo': """torch.manual_seed(42)
B, D = 8, 16
pred_z = torch.randn(B, D, requires_grad=True)
target_z = torch.randn(B, D)
losses = jepa_vicreg_loss(pred_z, target_z)
print("Output shapes:", [loss.shape for loss in losses])
print("Total loss:", losses[0].item())
print("Loss terms:", [loss.item() for loss in losses[1:]])""",
    'explanation': """### 核心思路
总目标由不变性 MSE、标准差下界铰链和非对角协方差惩罚组成，分别实现目标对齐、防止常量坍缩和减少特征冗余。

### 代码步骤拆解
- **Step 1（不变性）**：`pred_z` 与分离后的 `target_z` 在 `(B, D)` 上计算均方误差。
- **Step 2（方差约束）**：逐特征总体标准差低于 `gamma` 时产生惩罚。
- **Step 3（协方差约束）**：中心化 Gram 矩阵除以 `B-1`，移除对角后平方求和。

### 重点与易错点分析
- **注意点 1（目标停梯度）**：JEPA 目标分支必须 `.detach()`，只训练预测分支。
- **注意点 2（两种方差约定）**：标准差用 `unbiased=False`，协方差却按题意除以 `B-1`。
- **注意点 3（批量边界）**：协方差分母要求 `B >= 2`，单样本批次会除零。

### 复杂度分析
- **时间复杂度**：$O(BD^2)$，协方差矩阵乘法占主导。
- **空间复杂度**：$O(D^2+BD)$，主要保存协方差与中心化表示。""",
}
