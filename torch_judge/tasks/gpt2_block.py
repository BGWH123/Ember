"""GPT-2 Transformer Block task."""

TASK = {
    "title": "GPT-2 Transformer Block",
    "title_zh": "GPT-2 Transformer Block",
    "difficulty": "Hard",
    "category": "Transformer组件",
    "description_en": (
        "Implement a GPT-2 transformer block as an nn.Module.\n\nA GPT-2 block uses pre-norm architecture: LayerNorm before causal self-attention and MLP, with residual connections around both.\n\n**Signature:** `GPT2Block(d_model, num_heads)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (B, S, d_model)\n\n**Returns:** output tensor (B, S, d_model)\n\n**Constraints:**\n- Pre-norm: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`\n- MLP: Linear(d, 4d) -> GELU -> Linear(4d, d)\n- Attention must be causal (future tokens cannot affect past)"
    ),
    "description_zh": (
        "实现 GPT-2 Transformer 块（nn.Module）。\n\nGPT-2 块使用 pre-norm 架构：在因果自注意力和 MLP 之前进行 LayerNorm，两者都有残差连接。\n\n**签名:** `GPT2Block(d_model, num_heads)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (B, S, d_model)\n\n**返回:** 输出张量 (B, S, d_model)\n\n**约束:**\n- Pre-norm：`x = x + attn(ln1(x))`，`x = x + mlp(ln2(x))`\n- MLP：Linear(d, 4d) -> GELU -> Linear(4d, d)\n- 注意力必须是因果的（未来 token 不能影响过去）"
    ),
    "function_name": "GPT2Block",
    "hint": "Pre-norm residual: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`. MLP: `Linear(d,4d) → GELU → Linear(4d,d)`. Attention must be causal (mask future with `-inf`).",
    "hint_zh": "Pre-norm 残差：`x = x + attn(ln1(x))`，`x = x + mlp(ln2(x))`。MLP：`Linear(d,4d) → GELU → Linear(4d,d)`。注意力必须是因果的（用 `-inf` 遮蔽未来）。",
    "theory_en": (
        "GPT-2 uses a **pre-norm** transformer block: LayerNorm is applied *before* attention and MLP, with residual connections around both.\n\n**Architecture:**\n$$x = x + \\text{Attention}(\\text{LN}_1(x))$$\n$$x = x + \\text{MLP}(\\text{LN}_2(x))$$\n\n**Pre-norm vs Post-norm:**\n- Post-norm (original Transformer): $x = \\text{LN}(x + \\text{Attention}(x))$ — harder to train deep\n- Pre-norm: more stable gradients, allows training deeper networks (GPT-2, GPT-3, LLaMA)\n\n**MLP Structure:**\n$$\\text{MLP}(x) = W_2 \\cdot \\text{GELU}(W_1 x)$$\nwhere hidden dim = $4 \\times d_{model}$ (GPT-2 convention)"
    ),
    "theory_zh": (
        "GPT-2 使用 **pre-norm** Transformer 块：LayerNorm 在注意力和 MLP *之前* 应用，两者都有残差连接。\n\n**架构：**\n$$x = x + \\text{Attention}(\\text{LN}_1(x))$$\n$$x = x + \\text{MLP}(\\text{LN}_2(x))$$\n\n**Pre-norm vs Post-norm：**\n- Post-norm（原始 Transformer）：$x = \\text{LN}(x + \\text{Attention}(x))$ —— 深层训练困难\n- Pre-norm：梯度更稳定，允许训练更深网络（GPT-2、GPT-3、LLaMA）\n\n**MLP 结构：**\n$$\\text{MLP}(x) = W_2 \\cdot \\text{GELU}(W_1 x)$$\n其中隐藏维度 = $4 \\times d_{model}$（GPT-2 惯例）"
    ),
    "diagram_en": (
        "```mermaid\nflowchart TD\n    IN[Input x] --> LN1[LayerNorm]\n    LN1 --> ATT[Masked Self-Attention]\n    ATT --> ADD1[x + Attention]\n    IN --> ADD1\n    ADD1 --> LN2[LayerNorm]\n    LN2 --> MLP[MLP<br/>Linear -> GELU -> Linear]\n    MLP --> ADD2[x + MLP]\n    ADD1 --> ADD2\n    ADD2 --> OUT[Output]\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart TD\n    IN[输入 x] --> LN1[LayerNorm]\n    LN1 --> ATT[掩码自注意力]\n    ATT --> ADD1[x + Attention]\n    IN --> ADD1\n    ADD1 --> LN2[LayerNorm]\n    LN2 --> MLP[MLP<br/>Linear -> GELU -> Linear]\n    MLP --> ADD2[x + MLP]\n    ADD1 --> ADD2\n    ADD2 --> OUT[输出]\n```"
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """









import torch, torch.nn as nn
torch.manual_seed(0)
block = {fn}(d_model=64, num_heads=4)
assert isinstance(block, nn.Module), 'GPT2Block should inherit from nn.Module'
out = block(torch.randn(2, 8, 64))
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Has LayerNorm (pre-norm architecture)",
            "code": """









import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'ln1') and isinstance(block.ln1, nn.LayerNorm), 'Need self.ln1 = nn.LayerNorm'
assert hasattr(block, 'ln2') and isinstance(block.ln2, nn.LayerNorm), 'Need self.ln2 = nn.LayerNorm'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "MLP has 4x expansion with GELU",
            "code": """









import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'mlp'), 'Need self.mlp'
linears = [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]
assert len(linears) >= 2, f'MLP needs >= 2 Linear layers, got {len(linears)}'
assert linears[0].weight.shape == (128, 32), f'MLP first layer: {linears[0].weight.shape}, expected (128, 32)'
assert linears[-1].weight.shape == (32, 128), f'MLP last layer: {linears[-1].weight.shape}, expected (32, 128)'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Causal masking — future doesn't affect past",
            "code": """









import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 8, 32)
out1 = block(x)
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 4, 32)
out2 = block(x2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Future tokens affected past — not causal'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow to all parameters",
            "code": """









import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 4, 32, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in block.parameters())
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


class GPT2Block(nn.Module):
    """
    GPT-2 Transformer Block (Pre-Norm architecture).
    先对输入做 LayerNorm，再经过 Attention/MLP，最后残差连接。
    Pre-Norm 比 Post-Norm 更稳定，允许训练更深的网络。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()                               # nn.Module 初始化
        self.ln1 = nn.LayerNorm(d_model)                 # 注意力前的 LayerNorm
        self.ln2 = nn.LayerNorm(d_model)                 # MLP 前的 LayerNorm
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # 多头自注意力
        self.mlp = nn.Sequential(                        # MLP: 扩张 -> 激活 -> 收缩
            nn.Linear(d_model, 4 * d_model),             # 扩张到 4*d_model (GPT-2 惯例)
            nn.GELU(),                                    # GELU 激活
            nn.Linear(4 * d_model, d_model),             # 收缩回 d_model
        )
        self.dropout = nn.Dropout(dropout)               # Dropout 正则化

    def forward(self, x):
        # Pre-Norm Attention Block:
        # 1. LayerNorm -> Attention -> Dropout -> 残差连接
        # 残差连接缓解梯度消失，允许深层网络训练
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.dropout(attn_out)                   # 残差: x + Dropout(Attn(LN(x)))

        # Pre-Norm MLP Block:
        # 2. LayerNorm -> MLP -> Dropout -> 残差连接
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_out)                    # 残差: x + Dropout(MLP(LN(x)))
        return x                                         # (B, S, d_model)

    
    
    ''',
    "demo": '''








block = GPT2Block(64, 4)
print('Output:', block(torch.randn(2, 8, 64)).shape)
print('Params:', sum(p.numel() for p in block.parameters()))
    
    
    
    
    
    
    
    
    ''',
}
