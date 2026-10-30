"""Consistency Distillation & Pseudo-Huber Loss task."""

TASK = {
    'title': 'Consistency Distillation & Pseudo-Huber Loss',
    'title_zh': '一致性蒸馏（Consistency Distillation）与伪 Huber 损失',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'consistency_distillation_loss',
    'starter_code': """def consistency_distillation_loss(f_student, f_teacher, weights=None, c=0.001):
    pass""",
    'hint': """Detach the teacher before forming the squared Euclidean distance, apply the pseudo-Huber transform per sample, then optionally weight before averaging.""",
    'hint_zh': '计算平方欧氏距离前先分离教师输出，逐样本应用伪 Huber 变换，再选择性加权并取平均。',
    'description_en': """Compute consistency distillation between student and detached teacher outputs of shape (B,D). Return the mean optionally weighted objective and the unweighted per-sample pseudo-Huber values.""",
    'description_zh': '计算形状为 (B,D) 的学生输出与已分离教师输出之间的一致性蒸馏损失。返回可选加权目标的均值和未加权的逐样本伪 Huber 值。',
    'theory_en': """The pseudo-Huber form sqrt(||delta||^2+c^2)-c is smooth near zero and grows approximately linearly for large residuals. Stop-gradient makes the teacher a fixed regression target.""",
    'theory_zh': '伪 Huber 形式 sqrt(||delta||^2+c^2)-c 在零点附近光滑，对大残差近似线性增长。停止梯度使教师成为固定回归目标。',
    'diagram_en': """```text
student - stopgrad(teacher) -> squared L2 -> pseudo-Huber -> weight -> mean
```""",
    'diagram_zh': """```text
学生 - stopgrad(教师) -> 平方 L2 -> 伪 Huber -> 加权 -> 均值
```""",
    "tests": [
        {
            "name": 'shapes and no weights',
            "code": """import torch
s=torch.randn(4,6); t=torch.randn(4,6)
loss,per={fn}(s,t)
assert loss.shape==torch.Size([]) and per.shape==(4,)
assert torch.equal(loss,per.mean()) and (per>=0).all()""",
        },
        {
            "name": 'exact weighted reference',
            "code": """import torch
s=torch.tensor([[1.,2.],[-1.,3.]]); t=torch.tensor([[0.,2.],[2.,-1.]])
w=torch.tensor([.5,2.]); c=.2
loss,per={fn}(s,t,w,c)
ref=torch.sqrt(((s-t)**2).sum(-1)+c**2)-c
assert torch.equal(per,ref) and torch.equal(loss,(w*ref).mean())""",
        },
        {
            "name": 'identical outputs edge case',
            "code": """import torch
x=torch.randn(3,5)
loss,per={fn}(x,x,c=.01)
assert torch.equal(per,torch.zeros(3)) and loss.item()==0.0""",
        },
        {
            "name": 'student-only gradient flow',
            "code": """import torch
s=torch.randn(3,4,requires_grad=True); t=torch.randn(3,4,requires_grad=True)
w=torch.tensor([1.,2.,3.]); loss,per={fn}(s,t,w); (loss+per.sum()).backward()
assert s.grad is not None and torch.isfinite(s.grad).all()
assert t.grad is None""",
        },
    ],
    'solution': '''def consistency_distillation_loss(f_student, f_teacher, weights=None, c=0.001):
    """一致性蒸馏伪 Huber 损失 (Consistency Distillation Loss)。"""
    # Step 1: 分离教师分支并计算逐样本平方欧氏距离
    squared_distance = (
        (f_student - f_teacher.detach()) ** 2
    ).sum(dim=-1)

    # Step 2: 应用平滑的伪 Huber 变换
    per_sample_loss = torch.sqrt(squared_distance + c ** 2) - c

    # Step 3: 可选加权后取批次均值
    objective = (
        weights * per_sample_loss
        if weights is not None
        else per_sample_loss
    )
    return objective.mean(), per_sample_loss''',
    'demo': """torch.manual_seed(42)
B, D = 8, 64
student = torch.randn(B, D, requires_grad=True)
teacher = torch.randn(B, D)
loss, per_sample = consistency_distillation_loss(student, teacher)
print("Output shape:", per_sample.shape)
print("Loss:", loss.item())
print("Sample values:", per_sample[:3])""",
    'explanation': r"""### 核心思路
将教师输出停止梯度后，以 $\sqrt{\|f_s-f_t\|_2^2+c^2}-c$ 构造平滑且对大残差近似线性的逐样本目标。

### 代码步骤拆解
- **Step 1（固定教师）**：`f_teacher.detach()` 阻断教师梯度，沿特征维求平方距离。
- **Step 2（稳健变换）**：`per_sample_loss` 保持 `(B,)`，在零点附近平滑。
- **Step 3（加权归约）**：`weights` 仅影响标量训练目标，不改变返回的原始逐样本值。

### 重点与易错点分析
- **注意点 1（梯度截断）**：必须在做差前 `.detach()`，否则教师会被意外优化。
- **注意点 2（c 的位置）**：$c^2$ 位于平方根内，根外再减 $c$，保证相同输出损失严格为零。
- **注意点 3（权重语义）**：当前实现是加权值的算术均值，而不是除以权重和。

### 复杂度分析
- **时间复杂度**：$O(BD)$，主要是逐元素残差与特征归约。
- **空间复杂度**：$O(BD)$，主要保存残差及自动微分中间量。""",
}
