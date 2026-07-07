"""LoRA (Low-Rank Adaptation) task."""

TASK = {
    "title": "LoRA (Low-Rank Adaptation)",
    "title_zh": "LoRA（低秩适配）",
    "difficulty": "Medium",
    "category": "参数高效微调",
    "description_en": (
        "Implement LoRA (Low-Rank Adaptation) for a linear layer.\n\nLoRA freezes the base weights and adds trainable low-rank matrices A and B, enabling efficient fine-tuning with far fewer parameters.\n\n**Signature:** `LoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (*, in_features)\n\n**Returns:** `linear(x) + (x @ A^T @ B^T) * (alpha/rank)`\n\n**Constraints:**\n- Base `nn.Linear` weights must be frozen (requires_grad=False)\n- `lora_A`: (rank, in_features), `lora_B`: (out_features, rank) initialized to zeros\n- Only LoRA params should receive gradients"
    ),
    "description_zh": (
        "实现线性层的 LoRA（低秩适配）。\n\nLoRA 冻结基础权重并添加可训练的低秩矩阵 A 和 B，以极少的参数实现高效微调。\n\n**签名:** `LoRALinear(in_features, out_features, rank, alpha=1.0)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (*, in_features)\n\n**返回:** `linear(x) + (x @ A^T @ B^T) * (alpha/rank)`\n\n**约束:**\n- 基础 `nn.Linear` 权重必须冻结（requires_grad=False）\n- `lora_A`：(rank, in_features)，`lora_B`：(out_features, rank) 初始化为零\n- 只有 LoRA 参数应接收梯度"
    ),
    "function_name": "LoRALinear",
    "hint": (
        "1. Freeze `linear.weight/bias` with `requires_grad_(False)`\n2. `lora_A`: `(rank, in)`, `lora_B`: `(out, rank)` init zeros\n3. Forward: `linear(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)`"
    ),
    "hint_zh": (
        "1. `linear.weight/bias` 设 `requires_grad_(False)` 冻结\n2. `lora_A`: `(rank, in)`，`lora_B`: `(out, rank)` 初始化为零\n3. 前向：`linear(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)`"
    ),
    "tests": [
        {
            "name": "Base weights frozen",
            "code": """









import torch, torch.nn as nn
layer = {fn}(in_features=16, out_features=8, rank=4)
assert isinstance(layer, nn.Module)
assert not layer.linear.weight.requires_grad, 'Base weight must be frozen'
assert not layer.linear.bias.requires_grad, 'Base bias must be frozen'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "LoRA parameter shapes",
            "code": """









import torch
layer = {fn}(in_features=16, out_features=8, rank=4)
assert layer.lora_A.shape == (4, 16), f'lora_A: {layer.lora_A.shape}'
assert layer.lora_B.shape == (8, 4), f'lora_B: {layer.lora_B.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "B=0 means output equals base",
            "code": """









import torch
torch.manual_seed(0)
layer = {fn}(in_features=8, out_features=4, rank=2)
x = torch.randn(2, 8)
assert torch.allclose(layer(x), layer.linear(x), atol=1e-5), 'With B=0, should equal base'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Only LoRA params get gradients",
            "code": """









import torch
layer = {fn}(in_features=8, out_features=4, rank=2)
layer(torch.randn(2, 8)).sum().backward()
assert layer.lora_A.grad is not None, 'lora_A.grad is None'
assert layer.lora_B.grad is not None, 'lora_B.grad is None'
assert layer.linear.weight.grad is None, 'Base weight should not have grad'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Forward computation",
            "code": """









import torch
torch.manual_seed(0)
layer = {fn}(in_features=8, out_features=4, rank=2, alpha=2.0)
layer.lora_B.data.normal_()
x = torch.randn(3, 8)
ref = layer.linear(x) + (x @ layer.lora_A.T @ layer.lora_B.T) * (2.0 / 2)
assert torch.allclose(layer(x), ref, atol=1e-5), 'Forward mismatch'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








class LoRALinear(nn.Module):  # 继承 nn.Module，注册为可训练模块
    """
    LoRA (Low-Rank Adaptation) module.
    """
    def __init__(self, in_features, out_features, rank, alpha=1.0):  # 初始化: 定义模型结构和参数
        # Initialize layers and parameters
        super().__init__()  # 调用父类 nn.Module 初始化，注册所有子模块
        self.linear = nn.Linear(in_features, out_features)  # Linear projection
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):  # 前向传播: 定义数据流
        # Compute forward pass
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








layer = LoRALinear(16, 8, rank=4)
x = torch.randn(2, 16)
print('Output:', layer(x).shape)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f'Trainable: {trainable}/{total} ({100*trainable/total:.1f}%)')
    
    
    
    
    
    
    
    
    ''',
}
