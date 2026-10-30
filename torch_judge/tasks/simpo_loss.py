"""SimPO preference loss task."""

TASK = {
    "title": "SimPO (Simple Preference Optimization) Loss",
    "title_zh": "SimPO（无参考模型偏好优化）损失",
    "difficulty": "Easy",
    "category": "损失函数",
    "function_name": "simpo_loss",
    "starter_code": '''import torch
import torch.nn.functional as F

def simpo_loss(chosen_logps, rejected_logps, chosen_lengths, rejected_lengths,
               beta=2.0, gamma=0.5):
    pass
''',
    "hint": "Divide each sequence log-probability by its own floating-point length, scale by beta, subtract the rejected score and target margin, then use negative log-sigmoid.",
    "hint_zh": "将每个序列的对数概率除以对应的浮点长度，乘以 beta，减去拒绝样本分数和目标间隔，再计算负 log-sigmoid。",
    "description_en": "Implement reference-free SimPO from sums of response-token log-probabilities and response lengths. The chosen normalized score should exceed the rejected score by gamma.",
    "description_zh": "根据回复 token 对数概率之和及回复长度实现无参考模型的 SimPO。选中回复的归一化分数应比拒绝回复高 gamma。",
    "theory_en": "Length normalization removes preference for short responses. A logistic loss trains the beta-scaled chosen-minus-rejected score to clear a fixed margin gamma.",
    "theory_zh": "长度归一化消除对短回复的偏好；逻辑损失促使经 beta 缩放的选中与拒绝分数之差超过固定间隔 gamma。",
    "diagram_en": "```mermaid\nflowchart LR\nC[chosen sum / length] --> D[beta score difference]\nR[rejected sum / length] --> D\nG[gamma] --> D\nD --> L[-log sigmoid mean]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nC[选中总分/长度] --> D[beta 分数差]\nR[拒绝总分/长度] --> D\nG[gamma] --> D\nD --> L[负 log-sigmoid 均值]\n```",
    "tests": [
        {"name": "scalar shape", "code": """import torch
c = torch.tensor([-2.0, -4.0])
r = torch.tensor([-3.0, -6.0])
loss = {fn}(c, r, torch.tensor([2, 4]), torch.tensor([3, 6]))
assert loss.shape == torch.Size([]) and torch.isfinite(loss)
"""},
        {"name": "exact reference", "code": """import torch
import torch.nn.functional as F
c = torch.tensor([-2.0, -9.0], dtype=torch.double)
r = torch.tensor([-4.0, -8.0], dtype=torch.double)
cl, rl = torch.tensor([2, 3]), torch.tensor([4, 2])
got = {fn}(c, r, cl, rl, beta=1.7, gamma=0.3)
logits = 1.7 / cl.float() * c - 1.7 / rl.float() * r - 0.3
assert torch.allclose(got, -F.logsigmoid(logits).mean(), atol=1e-12, rtol=1e-12)
"""},
        {"name": "single item edge case", "code": """import torch
import torch.nn.functional as F
got = {fn}(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([1]), torch.tensor([1]), gamma=0.0)
assert got.shape == torch.Size([])
assert torch.equal(got, torch.tensor(float(torch.log(torch.tensor(2.0)))))
"""},
        {"name": "gradient flow", "code": """import torch
c = torch.randn(4, dtype=torch.double, requires_grad=True)
r = torch.randn(4, dtype=torch.double, requires_grad=True)
loss = {fn}(c, r, torch.tensor([1, 2, 3, 4]), torch.tensor([4, 3, 2, 1]))
loss.backward()
assert c.grad is not None and r.grad is not None
assert c.grad.shape == c.shape and torch.isfinite(r.grad).all()
"""},
    ],
    "solution": '''import torch
import torch.nn.functional as F

def simpo_loss(chosen_logps, rejected_logps, chosen_lengths, rejected_lengths,
               beta=2.0, gamma=0.5):
    logits = (
        (beta / chosen_lengths.float()) * chosen_logps
        - (beta / rejected_lengths.float()) * rejected_logps
        - gamma
    )
    return -F.logsigmoid(logits).mean()
''',
    "demo": '''chosen = torch.tensor([-4.0, -6.0])
rejected = torch.tensor([-7.0, -8.0])
print(simpo_loss(chosen, rejected, torch.tensor([4, 5]), torch.tensor([4, 5])))
''',
    "explanation": "Each summed log-probability is normalized independently before comparison. F.logsigmoid computes the logistic objective stably.",
}
