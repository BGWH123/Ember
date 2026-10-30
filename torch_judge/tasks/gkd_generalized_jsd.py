"""On-Policy Distillation: Generalized JSD (GKD) Loss task."""

TASK = {
    'title': 'On-Policy Distillation: Generalized JSD (GKD) Loss',
    'title_zh': '在线策略蒸馏：广义 JS 散度（GKD）损失',
    'difficulty': 'Medium',
    'category': '损失函数',
    'function_name': 'gkd_jsd_loss',
    'starter_code': """def gkd_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0, eps=1e-8):
    pass""",
    'hint': """Detach teacher logits, temperature-scale both softmaxes, construct the beta mixture, compute both KL divergences along vocabulary, and average.""",
    'hint_zh': '分离教师 logits，对两者进行温度缩放 softmax，构造 beta 混合分布，沿词表维计算两个 KL 散度并取均值。',
    'description_en': """Implement generalized Jensen-Shannon divergence for on-policy knowledge distillation over batched token logits. Teacher logits are fixed targets.""",
    'description_zh': '针对批量 token logits 实现在线知识蒸馏的广义 Jensen-Shannon 散度。教师 logits 必须作为固定目标。',
    'theory_en': """The mixture m=beta*p_t+(1-beta)*p_s gives a finite interpolation target. Weighted forward KL terms yield generalized JSD and coincide at zero when distributions match.""",
    'theory_zh': '混合分布 m=beta*p_t+(1-beta)*p_s 提供有限的插值目标。加权前向 KL 构成广义 JSD，两分布相同时为零。',
    'diagram_en': """```mermaid
flowchart LR
S[student logits / T] --> PS[softmax]
T[detached teacher logits / T] --> PT[softmax]
PS --> M[beta mixture]
PT --> M
M --> J[weighted KL mean]
PS --> J
PT --> J
```""",
    'diagram_zh': """```mermaid
flowchart LR
S[学生 logits / T] --> PS[softmax]
T[分离教师 logits / T] --> PT[softmax]
PS --> M[beta 混合]
PT --> M
M --> J[加权 KL 均值]
PS --> J
PT --> J
```""",
    "tests": [
        {
            "name": 'scalar nonnegative output',
            "code": """import torch
s = torch.randn(2, 3, 5)
t = torch.randn(2, 3, 5)
loss = {fn}(s, t)
assert loss.shape == torch.Size([])
assert torch.isfinite(loss) and loss >= -1e-7""",
        },
        {
            "name": 'exact generalized JSD',
            "code": """import torch
s = torch.tensor([[[1.0, -1.0, 0.5]]], dtype=torch.double)
t = torch.tensor([[[0.0, 2.0, -0.5]]], dtype=torch.double)
got = {fn}(s, t, beta=.3, temperature=.7, eps=1e-12)
ps, pt = torch.softmax(s/.7, -1), torch.softmax(t/.7, -1)
m = .3*pt + .7*ps
klt = (pt*(torch.log(pt.clamp(min=1e-12))-torch.log(m.clamp(min=1e-12)))).sum(-1)
kls = (ps*(torch.log(ps.clamp(min=1e-12))-torch.log(m.clamp(min=1e-12)))).sum(-1)
assert torch.allclose(got, (.3*klt+.7*kls).mean(), atol=1e-12, rtol=1e-12)""",
        },
        {
            "name": 'identical distributions edge',
            "code": """import torch
x = torch.randn(1, 2, 4)
loss = {fn}(x, x.clone(), beta=0.8, temperature=2.0)
assert loss.shape == torch.Size([])
assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)""",
        },
        {
            "name": 'teacher detached',
            "code": """import torch
s = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
t = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
{fn}(s, t, beta=.4).backward()
assert s.grad is not None and s.grad.shape == s.shape
assert t.grad is None and torch.isfinite(s.grad).all()""",
        },
    ],
    'solution': '''def gkd_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0, eps=1e-8):
    """计算广义 Jensen-Shannon 蒸馏损失 (Generalized JSD)。"""
    # Step 1: 温度缩放并固定教师分布
    student_probs = torch.softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.softmax(teacher_logits.detach() / temperature, dim=-1)

    # Step 2: 构造混合分布及稳定对数
    mixture = beta * teacher_probs + (1.0 - beta) * student_probs
    log_mixture = torch.log(mixture.clamp(min=eps))

    # Step 3: 组合两个前向 KL 并取均值
    teacher_kl = (teacher_probs * (torch.log(teacher_probs.clamp(min=eps)) - log_mixture)).sum(dim=-1)
    student_kl = (student_probs * (torch.log(student_probs.clamp(min=eps)) - log_mixture)).sum(dim=-1)
    jsd = beta * teacher_kl + (1.0 - beta) * student_kl
    return jsd.mean()''',
    'demo': """torch.manual_seed(42)
student_logits = torch.randn(2, 4, 10, requires_grad=True)
teacher_logits = torch.randn(2, 4, 10)
loss = gkd_jsd_loss(student_logits, teacher_logits)
print("Output shape:", loss.shape)
print("Loss value:", loss.item())
print("Is finite:", torch.isfinite(loss).item())""",
    'explanation': r"""### 核心思路
先构造 $m=\beta p_t+(1-\beta)p_s$，再计算 $\beta KL(p_t\|m)+(1-\beta)KL(p_s\|m)$，并对批次与 token 取均值。

### 代码步骤拆解
- **Step 1（概率分布）**：在词表维对温度缩放 logits 做 softmax，教师 logits 先 `.detach()`。
- **Step 2（混合目标）**：`mixture` 与两分布同形，`clamp` 后取对数。
- **Step 3（散度归约）**：先沿词表求和，再对剩余维求均值形成标量。

### 重点与易错点分析
- **注意点 1（教师停梯度）**：分离必须发生在教师 softmax 前，确保完整教师分支无梯度。
- **注意点 2（数值稳定）**：`eps` 只用于对数下界，避免零概率导致 `log(0)`。
- **注意点 3（权重方向）**：教师项权重为 `beta`，学生项权重为 `1-beta`，混合分布也须一致。

### 复杂度分析
- **时间复杂度**：$O(BSV)$，softmax 与 KL 都遍历词表。
- **空间复杂度**：$O(BSV)$，保存两个概率分布和混合分布。""",
}
