"""Mixture of Experts (MoE) task."""

TASK = {
    "title": "Mixture of Experts (MoE)",
    "title_zh": "混合专家（MoE）",
    "difficulty": "Hard",
    "category": "混合专家模型",
    "description_en": (
        "Implement a Mixture of Experts (MoE) layer as an nn.Module.\n\nMoE routes each token to the top-k experts via a learned router, combining their outputs with softmax-normalized weights for conditional computation.\n\n**Signature:** `MixtureOfExperts(d_model, d_ff, num_experts, top_k=2)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (B, S, d_model)\n\n**Returns:** output tensor (B, S, d_model)\n\n**Constraints:**\n- Router: `nn.Linear(d_model, num_experts)` -> topk -> softmax\n- Each expert: Linear -> ReLU -> Linear\n- Store experts in `self.experts` (nn.ModuleList)"
    ),
    "description_zh": (
        "实现混合专家（MoE）层（nn.Module）。\n\nMoE 通过学习的路由器将每个 token 路由到 top-k 个专家，用 softmax 归一化的权重组合它们的输出，实现条件计算。\n\n**签名:** `MixtureOfExperts(d_model, d_ff, num_experts, top_k=2)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (B, S, d_model)\n\n**返回:** 输出张量 (B, S, d_model)\n\n**约束:**\n- 路由器：`nn.Linear(d_model, num_experts)` -> topk -> softmax\n- 每个专家：Linear -> ReLU -> Linear\n- 专家存储在 `self.experts`（nn.ModuleList）中"
    ),
    "function_name": "MixtureOfExperts",
    "hint": "`router`: `Linear(d, num_experts)` → `topk` → `softmax` weights. Each expert: `Linear → ReLU → Linear`. Weighted sum of top-k expert outputs per token.",
    "hint_zh": "`router`：`Linear(d, num_experts)` → `topk` → `softmax` 权重。每个专家：`Linear → ReLU → Linear`。对每个 token 的 top-k 专家输出加权求和。",
    "tests": [
        {
            "name": "Output shape",
            "code": """









import torch, torch.nn as nn
moe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2)
assert isinstance(moe, nn.Module)
out = moe(torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Has router and experts",
            "code": """









import torch, torch.nn as nn
moe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2)
assert hasattr(moe, 'router'), 'Need self.router'
assert hasattr(moe, 'experts'), 'Need self.experts'
assert len(moe.experts) == 4, f'Expected 4 experts, got {len(moe.experts)}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Router logits shape",
            "code": """









import torch
moe = {fn}(d_model=16, d_ff=32, num_experts=8, top_k=2)
logits = moe.router(torch.randn(4, 16))
assert logits.shape == (4, 8), f'Router output: {logits.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
moe = {fn}(d_model=16, d_ff=32, num_experts=4, top_k=2)
x = torch.randn(1, 4, 16, requires_grad=True)
moe(x).sum().backward()
assert x.grad is not None, 'x.grad is None'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Routing to single expert matches manual computation",
            "code": """









import torch, torch.nn as nn
torch.manual_seed(0)
d_model, d_ff, num_experts, top_k = 8, 16, 4, 1
moe = {fn}(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k)
# Force router to always select expert 0 via large bias
with torch.no_grad():
    moe.router.weight.zero_()
    if moe.router.bias is not None:
        moe.router.bias.zero_()
        moe.router.bias[0] = 1e6  # expert 0 always wins
    else:
        # No bias: set weight row 0 to a large constant vector
        moe.router.weight[0] = 1e6
torch.manual_seed(1)
x = torch.randn(2, 3, d_model)
out = moe(x)
# With top_k=1 and expert 0 always winning, softmax of single value = 1.0
# Expected: expert 0 applied to every token with weight 1.0
x_flat = x.reshape(-1, d_model)
expert0 = moe.experts[0]
expected = expert0(x_flat).reshape(2, 3, d_model)
assert torch.allclose(out, expected, atol=1e-5), f'Max diff: {(out - expected).abs().max().item()}'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








class _ManualReLU(nn.Module):  # 继承 nn.Module，注册为可训练模块
    """
    Mixture of Experts (MoE) module.
    """
    def forward(self, x):  # 前向传播: 定义数据流
        # Compute forward pass
        return x.clamp(min=0)

class MixtureOfExperts(nn.Module):  # 继承 nn.Module，注册为可训练模块
    """
    Mixture of Experts (MoE) module.
    """
    def __init__(self, d_model, d_ff, num_experts, top_k=2):  # 初始化: 定义模型结构和参数
        # Initialize layers and parameters
        super().__init__()  # 调用父类 nn.Module 初始化，注册所有子模块
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)  # Linear projection
        self.experts = nn.ModuleList([  # 模块列表: PyTorch 自动追踪其中的子模块
            nn.Sequential(nn.Linear(d_model, d_ff), _ManualReLU(), nn.Linear(d_ff, d_model))  # Linear projection
            for _ in range(num_experts)
        ])

    def forward(self, x):  # 前向传播: 定义数据流
        # Compute forward pass
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(-1, D)
        else:
            x_flat = x
        logits = self.router(x_flat)
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)  # Apply softmax to get attention weights
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(len(self.experts)):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    output[mask] += weights[mask, k:k+1] * self.experts[e](x_flat[mask])
        return output.reshape(orig_shape)
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








moe = MixtureOfExperts(32, 64, num_experts=4, top_k=2)
x = torch.randn(2, 8, 32)
print('Output:', moe(x).shape)
print('Params:', sum(p.numel() for p in moe.parameters()))
    
    
    
    
    
    
    
    
    ''',
}
