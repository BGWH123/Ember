"""Bradley-Terry Reward Model Loss task."""

TASK = {
    "title": "Bradley-Terry Reward Model Loss",
    "title_zh": "Bradley-Terry 奖励模型",
    "difficulty": "Medium",
    "category": "强化学习",
    "description_en": (
        "Implement the canonical RLHF reward model training loss based on the Bradley-Terry preference model.\n\nGiven hidden states for chosen and rejected responses, project them to scalar rewards and compute the preference loss.\n\n**Signature:** `reward_model_loss(chosen_hidden, rejected_hidden, reward_head) -> Tensor`\n\n**Parameters:**\n- `chosen_hidden` — last-token hidden states for chosen responses (B, D)\n- `rejected_hidden` — last-token hidden states for rejected responses (B, D)\n- `reward_head` — linear projection to scalar reward (D, 1)\n\n**Returns:** scalar loss tensor\n\n**Formula:**\n1. `r_chosen = chosen_hidden @ reward_head` — shape (B, 1)\n2. `r_rejected = rejected_hidden @ reward_head` — shape (B, 1)\n3. `loss = -mean(log(sigmoid(r_chosen - r_rejected)))`\n\n**Constraint:** implement sigmoid manually — `σ(x) = 1 / (1 + exp(-x))` — do not use library sigmoid functions."
    ),
    "description_zh": (
        "实现基于 Bradley-Terry 偏好模型的标准 RLHF 奖励模型训练损失。\n\n给定选中和拒绝响应的隐藏状态，将其投影为标量奖励并计算偏好损失。\n\n**签名:** `reward_model_loss(chosen_hidden, rejected_hidden, reward_head) -> Tensor`\n\n**参数:**\n- `chosen_hidden` — 选中响应的最后一个 token 隐藏状态 (B, D)\n- `rejected_hidden` — 拒绝响应的最后一个 token 隐藏状态 (B, D)\n- `reward_head` — 投影到标量奖励的线性层权重 (D, 1)\n\n**返回:** 标量损失张量\n\n**公式:**\n1. `r_chosen = chosen_hidden @ reward_head` — 形状 (B, 1)\n2. `r_rejected = rejected_hidden @ reward_head` — 形状 (B, 1)\n3. `loss = -mean(log(sigmoid(r_chosen - r_rejected)))`\n\n**约束:** 手动实现 sigmoid — `σ(x) = 1 / (1 + exp(-x))` — 不得使用库函数。"
    ),
    "function_name": "reward_model_loss",
    "hint": (
        "1. r_chosen = (chosen_hidden @ reward_head).squeeze(-1)  → (B,)\n2. r_rejected = (rejected_hidden @ reward_head).squeeze(-1)\n3. margin = r_chosen - r_rejected\n4. loss = -log(1 / (1 + exp(-margin))).mean()  (manual sigmoid)"
    ),
    "hint_zh": (
        "1. r_chosen = (chosen_hidden @ reward_head).squeeze(-1)  → (B,)\n2. r_rejected = (rejected_hidden @ reward_head).squeeze(-1)\n3. margin = r_chosen - r_rejected\n4. loss = -log(1 / (1 + exp(-margin))).mean()  (手动 sigmoid)"
    ),
    "tests": [
        {
            "name": "Returns scalar",
            "code": """









import torch
torch.manual_seed(0)
B, D = 4, 16
chosen = torch.randn(B, D)
rejected = torch.randn(B, D)
head = torch.randn(D, 1)
loss = {fn}(chosen, rejected, head)
assert loss.shape == torch.Size([]), f'Expected scalar, got shape {loss.shape}'
assert loss.item() > 0, 'Loss should be positive'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Clear preference -> low loss",
            "code": """









import torch
torch.manual_seed(1)
B, D = 8, 32
head = torch.ones(D, 1) / D
# chosen hidden states much larger than rejected
chosen = torch.ones(B, D) * 10.0
rejected = torch.ones(B, D) * -10.0
loss = {fn}(chosen, rejected, head)
assert loss.item() < 0.01, f'Loss should be near 0 for clear preference, got {loss.item():.4f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Equal hidden states -> loss near log(2)",
            "code": """









import torch
import math
torch.manual_seed(2)
B, D = 16, 64
chosen = torch.randn(B, D)
rejected = chosen.clone()  # identical
head = torch.randn(D, 1)
loss = {fn}(chosen, rejected, head)
assert abs(loss.item() - math.log(2)) < 0.01, f'Expected log(2)={math.log(2):.4f}, got {loss.item():.4f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flows through all inputs",
            "code": """









import torch
torch.manual_seed(3)
B, D = 4, 8
chosen = torch.randn(B, D, requires_grad=True)
rejected = torch.randn(B, D, requires_grad=True)
head = torch.randn(D, 1, requires_grad=True)
loss = {fn}(chosen, rejected, head)
loss.backward()
assert chosen.grad is not None, 'No gradient for chosen_hidden'
assert rejected.grad is not None, 'No gradient for rejected_hidden'
assert head.grad is not None, 'No gradient for reward_head'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Larger margin -> lower loss",
            "code": """









import torch
torch.manual_seed(4)
B, D = 8, 16
head = torch.ones(D, 1) / D
base = torch.zeros(B, D)
# small margin
chosen_small = base + 0.5
rejected_small = base - 0.5
loss_small = {fn}(chosen_small, rejected_small, head)
# large margin
chosen_large = base + 5.0
rejected_large = base - 5.0
loss_large = {fn}(chosen_large, rejected_large, head)
assert loss_large.item() < loss_small.item(), 'Larger margin should yield lower loss'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








def reward_model_loss(chosen_hidden, rejected_hidden, reward_head):
    r_chosen = (chosen_hidden @ reward_head).squeeze(-1)     # (B,)
    r_rejected = (rejected_hidden @ reward_head).squeeze(-1) # (B,)
    margin = r_chosen - r_rejected
    # manual log-sigmoid: log(1/(1+exp(-x))) = -log(1+exp(-x))
    loss = -torch.log(1.0 / (1.0 + torch.exp(-margin))).mean()  # 指数运算
    return loss
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








torch.manual_seed(0)

B, D = 8, 64
reward_head = torch.randn(D, 1)

h = torch.randn(B, D)
loss_equal = reward_model_loss(h, h, reward_head)
print(f"Equal hidden states => loss = {loss_equal.item():.4f}  (expected ≈ {math.log(2):.4f})")

chosen  = torch.ones(B, D)
rejected = -torch.ones(B, D)
loss_good = reward_model_loss(chosen, rejected, reward_head)
print(f"Chosen >> rejected  => loss = {loss_good.item():.4f}  (expected small)")
    
    
    
    
    
    
    
    
    ''',
}
