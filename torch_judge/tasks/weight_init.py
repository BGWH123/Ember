"""Kaiming Initialization task."""

TASK = {
    "title": "Kaiming Initialization",
    "title_zh": "Kaiming 初始化",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "description_en": (
        "Implement Kaiming (He) weight initialization.\n\nKaiming initialization sets weight variance based on fan-in to preserve signal magnitude through ReLU networks, preventing vanishing/exploding activations.\n\n**Signature:** `kaiming_init(weight) -> Tensor`\n\n**Parameters:**\n- `weight` — tensor to initialize in-place (out_features, in_features)\n\n**Returns:** the same tensor (in-place operation)\n\n**Constraints:**\n- `std = sqrt(2 / fan_in)` where `fan_in = weight.shape[1]`\n- Fill with `normal(0, std)`\n- Smaller fan_in should give larger std"
    ),
    "description_zh": (
        "实现 Kaiming（He）权重初始化。\n\nKaiming 初始化根据 fan-in 设置权重方差，以在 ReLU 网络中保持信号幅度，防止激活值消失或爆炸。\n\n**签名:** `kaiming_init(weight) -> Tensor`\n\n**参数:**\n- `weight` — 需要原地初始化的张量 (out_features, in_features)\n\n**返回:** 同一张量（原地操作）\n\n**约束:**\n- `std = sqrt(2 / fan_in)`，其中 `fan_in = weight.shape[1]`\n- 用 `normal(0, std)` 填充\n- 更小的 fan_in 应产生更大的 std"
    ),
    "function_name": "kaiming_init",
    "hint": (
        "`fan_in = weight.shape[1]` → `std = sqrt(2 / fan_in)`\n`weight.normal_(0, std)` in-place → return `weight`"
    ),
    "hint_zh": (
        "`fan_in = weight.shape[1]` → `std = sqrt(2 / fan_in)`\n`weight.normal_(0, std)` 原地操作 → 返回 `weight`"
    ),
    "tests": [
        {
            "name": "Mean approximately 0",
            "code": """









import torch
torch.manual_seed(0)
w = torch.empty(256, 512)
{fn}(w)
assert abs(w.mean().item()) < 0.02, f'Mean too far from 0: {w.mean().item():.4f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Std matches sqrt(2/fan_in)",
            "code": """









import torch, math
torch.manual_seed(0)
fan_in = 1024
w = torch.empty(256, fan_in)
{fn}(w)
expected = math.sqrt(2.0 / fan_in)
assert abs(w.std().item() - expected) < 0.005, f'Std {w.std().item():.4f} vs expected {expected:.4f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Returns same tensor (in-place)",
            "code": """









import torch
w = torch.empty(64, 32)
out = {fn}(w)
assert out is w, 'Should return the same tensor'
assert out.shape == (64, 32), 'Shape should be unchanged'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Smaller fan_in gives larger std",
            "code": """









import torch
w1 = torch.empty(64, 16)
w2 = torch.empty(64, 256)
{fn}(w1)
{fn}(w2)
assert w1.std().item() > w2.std().item(), 'Smaller fan_in should give larger std'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








def kaiming_init(weight):
    fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        weight.normal_(0, std)
    return weight
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f} (expect ~0)')
print(f'Std:  {w.std():.4f} (expect {math.sqrt(2/512):.4f})')
    
    
    
    
    
    
    
    
    ''',
}
