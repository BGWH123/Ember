"""Consistency distillation with pseudo-Huber loss task."""

TASK = {
    "title": "Consistency Distillation & Pseudo-Huber Loss",
    "title_zh": "一致性蒸馏（Consistency Distillation）与伪 Huber 损失",
    "difficulty": "Medium",
    "category": "扩散与流模型",
    "function_name": "consistency_distillation_loss",
    "starter_code": '''def consistency_distillation_loss(f_student, f_teacher, weights=None, c=0.001):
    pass
''',
    "hint": "Detach the teacher before forming the squared Euclidean distance, apply the pseudo-Huber transform per sample, then optionally weight before averaging.",
    "hint_zh": "计算平方欧氏距离前先分离教师输出，逐样本应用伪 Huber 变换，再选择性加权并取平均。",
    "description_en": "Compute consistency distillation between student and detached teacher outputs of shape (B,D). Return the mean optionally weighted objective and the unweighted per-sample pseudo-Huber values.",
    "description_zh": "计算形状为 (B,D) 的学生输出与已分离教师输出之间的一致性蒸馏损失。返回可选加权目标的均值和未加权的逐样本伪 Huber 值。",
    "theory_en": "The pseudo-Huber form sqrt(||delta||^2+c^2)-c is smooth near zero and grows approximately linearly for large residuals. Stop-gradient makes the teacher a fixed regression target.",
    "theory_zh": "伪 Huber 形式 sqrt(||delta||^2+c^2)-c 在零点附近光滑，对大残差近似线性增长。停止梯度使教师成为固定回归目标。",
    "diagram_en": "```text\nstudent - stopgrad(teacher) -> squared L2 -> pseudo-Huber -> weight -> mean\n```",
    "diagram_zh": "```text\n学生 - stopgrad(教师) -> 平方 L2 -> 伪 Huber -> 加权 -> 均值\n```",
    "tests": [
        {
            "name": "shapes and no weights",
            "code": '''import torch
s=torch.randn(4,6); t=torch.randn(4,6)
loss,per={fn}(s,t)
assert loss.shape==torch.Size([]) and per.shape==(4,)
assert torch.equal(loss,per.mean()) and (per>=0).all()
''',
        },
        {
            "name": "exact weighted reference",
            "code": '''import torch
s=torch.tensor([[1.,2.],[-1.,3.]]); t=torch.tensor([[0.,2.],[2.,-1.]])
w=torch.tensor([.5,2.]); c=.2
loss,per={fn}(s,t,w,c)
ref=torch.sqrt(((s-t)**2).sum(-1)+c**2)-c
assert torch.equal(per,ref) and torch.equal(loss,(w*ref).mean())
''',
        },
        {
            "name": "identical outputs edge case",
            "code": '''import torch
x=torch.randn(3,5)
loss,per={fn}(x,x,c=.01)
assert torch.equal(per,torch.zeros(3)) and loss.item()==0.0
''',
        },
        {
            "name": "student-only gradient flow",
            "code": '''import torch
s=torch.randn(3,4,requires_grad=True); t=torch.randn(3,4,requires_grad=True)
w=torch.tensor([1.,2.,3.]); loss,per={fn}(s,t,w); (loss+per.sum()).backward()
assert s.grad is not None and torch.isfinite(s.grad).all()
assert t.grad is None
''',
        },
    ],
    "solution": '''import torch

def consistency_distillation_loss(f_student, f_teacher, weights=None, c=0.001):
    diff_sq = ((f_student - f_teacher.detach()) ** 2).sum(dim=-1)
    phuber = torch.sqrt(diff_sq + c ** 2) - c
    if weights is not None:
        weighted = weights * phuber
    else:
        weighted = phuber
    return weighted.mean(), phuber
''',
    "demo": '''student = torch.randn(8, 64, requires_grad=True)
teacher = torch.randn(8, 64)
loss, per_sample = consistency_distillation_loss(student, teacher)
print(loss, per_sample.shape)
''',
    "explanation": "Weights affect only the reduced training objective; the second return value always exposes the original per-sample pseudo-Huber loss. Detaching the teacher prevents accidental teacher updates.",
}
