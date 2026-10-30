"""GPT-2 Transformer Block task."""

TASK = {
    'title': 'GPT-2 Transformer Block',
    'title_zh': 'GPT-2 Transformer Block',
    'difficulty': 'Hard',
    'category': 'Transformer组件',
    'function_name': 'GPT2Block',
    'starter_code': """class GPT2Block(nn.Module):
    pass""",
    'hint': """Pre-norm residual: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`. MLP: `Linear(d,4d) → GELU → Linear(4d,d)`. Attention must be causal (mask future with `-inf`).""",
    'hint_zh': """Pre-norm 残差：`x = x + attn(ln1(x))`，`x = x + mlp(ln2(x))`。MLP：`Linear(d,4d) → GELU → Linear(4d,d)`。注意力必须是因果的（用 `-inf` 遮蔽未来）。""",
    'description_en': """Implement a GPT-2 transformer block as an nn.Module.

A GPT-2 block uses pre-norm architecture: LayerNorm before causal self-attention and MLP, with residual connections around both.

**Signature:** `GPT2Block(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, S, d_model)

**Returns:** output tensor (B, S, d_model)

**Constraints:**
- Pre-norm: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`
- MLP: Linear(d, 4d) -> GELU -> Linear(4d, d)
- Attention must be causal (future tokens cannot affect past)""",
    'description_zh': """实现 GPT-2 Transformer 块（nn.Module）。

GPT-2 块使用 pre-norm 架构：在因果自注意力和 MLP 之前进行 LayerNorm，两者都有残差连接。

**签名:** `GPT2Block(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, S, d_model)

**返回:** 输出张量 (B, S, d_model)

**约束:**
- Pre-norm：`x = x + attn(ln1(x))`，`x = x + mlp(ln2(x))`
- MLP：Linear(d, 4d) -> GELU -> Linear(4d, d)
- 注意力必须是因果的（未来 token 不能影响过去）""",
    'theory_en': r"""GPT-2 uses a **pre-norm** transformer block: LayerNorm is applied *before* attention and MLP, with residual connections around both.

**Architecture:**
$$x = x + \text{Attention}(\text{LN}_1(x))$$
$$x = x + \text{MLP}(\text{LN}_2(x))$$

**Pre-norm vs Post-norm:**
- Post-norm (original Transformer): $x = \text{LN}(x + \text{Attention}(x))$ — harder to train deep
- Pre-norm: more stable gradients, allows training deeper networks (GPT-2, GPT-3, LLaMA)

**MLP Structure:**
$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x)$$
where hidden dim = $4 \times d_{model}$ (GPT-2 convention)""",
    'theory_zh': r"""GPT-2 使用 **pre-norm** Transformer 块：LayerNorm 在注意力和 MLP *之前* 应用，两者都有残差连接。

**架构：**
$$x = x + \text{Attention}(\text{LN}_1(x))$$
$$x = x + \text{MLP}(\text{LN}_2(x))$$

**Pre-norm vs Post-norm：**
- Post-norm（原始 Transformer）：$x = \text{LN}(x + \text{Attention}(x))$ —— 深层训练困难
- Pre-norm：梯度更稳定，允许训练更深网络（GPT-2、GPT-3、LLaMA）

**MLP 结构：**
$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x)$$
其中隐藏维度 = $4 \times d_{model}$（GPT-2 惯例）""",
    'diagram_en': """```mermaid
flowchart TD
    IN[Input x] --> LN1[LayerNorm]
    LN1 --> ATT[Masked Self-Attention]
    ATT --> ADD1[x + Attention]
    IN --> ADD1
    ADD1 --> LN2[LayerNorm]
    LN2 --> MLP[MLP<br/>Linear -> GELU -> Linear]
    MLP --> ADD2[x + MLP]
    ADD1 --> ADD2
    ADD2 --> OUT[Output]
```""",
    'diagram_zh': """```mermaid
flowchart TD
    IN[输入 x] --> LN1[LayerNorm]
    LN1 --> ATT[掩码自注意力]
    ATT --> ADD1[x + Attention]
    IN --> ADD1
    ADD1 --> LN2[LayerNorm]
    LN2 --> MLP[MLP<br/>Linear -> GELU -> Linear]
    MLP --> ADD2[x + MLP]
    ADD1 --> ADD2
    ADD2 --> OUT[输出]
```""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
block = {fn}(d_model=64, num_heads=4)
assert isinstance(block, nn.Module), 'GPT2Block should inherit from nn.Module'
out = block(torch.randn(2, 8, 64))
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": 'Has LayerNorm (pre-norm architecture)',
            "code": """import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'ln1') and isinstance(block.ln1, nn.LayerNorm), 'Need self.ln1 = nn.LayerNorm'
assert hasattr(block, 'ln2') and isinstance(block.ln2, nn.LayerNorm), 'Need self.ln2 = nn.LayerNorm'""",
        },
        {
            "name": 'MLP has 4x expansion with GELU',
            "code": """import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'mlp'), 'Need self.mlp'
linears = [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]
assert len(linears) >= 2, f'MLP needs >= 2 Linear layers, got {len(linears)}'
assert linears[0].weight.shape == (128, 32), f'MLP first layer: {linears[0].weight.shape}, expected (128, 32)'
assert linears[-1].weight.shape == (32, 128), f'MLP last layer: {linears[-1].weight.shape}, expected (32, 128)'""",
        },
        {
            "name": "Causal masking — future doesn't affect past",
            "code": """import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 8, 32)
out1 = block(x)
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 4, 32)
out2 = block(x2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Future tokens affected past — not causal'""",
        },
        {
            "name": 'Gradient flow to all parameters',
            "code": """import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 4, 32, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in block.parameters())
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'""",
        },
    ],
    'solution': '''class GPT2Block(nn.Module):
    """实现预归一化 GPT-2 Transformer 块 (GPT-2 Block)。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        # Step 1: 定义两个预归一化子层
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Step 2: 执行带因果掩码的预归一化自注意力
        S = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        normalized = self.ln1(x)
        attn_out, _ = self.attn(
            normalized, normalized, normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Step 3: 执行预归一化 MLP 与残差连接
        return x + self.mlp(self.ln2(x))''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 8, 64
block = GPT2Block(D, 4)
out = block(torch.randn(B, S, D))
print("Output shape:", out.shape)
print("Parameter count:", sum(param.numel() for param in block.parameters()))
print("Sample values:", out[0, 0, :4])""",
    'explanation': r"""### 核心思路
GPT-2 块采用 $x\leftarrow x+\operatorname{Attn}(LN(x))$ 与 $x\leftarrow x+\operatorname{MLP}(LN(x))$，并用上三角掩码禁止访问未来 token。

### 代码步骤拆解
- **Step 1（模块构造）**：定义两个 `LayerNorm`、多头注意力和 $D\to4D\to D$ 的 MLP。
- **Step 2（因果注意力）**：`causal_mask` 为 `(S, S)`，注意力输出保持 `(B, S, D)`。
- **Step 3（MLP 残差）**：归一化后的表示经过 GELU MLP，再与注意力残差结果相加。

### 重点与易错点分析
- **注意点 1（因果掩码）**：掩码应遮住严格上三角，当前 token 可以关注自身和历史。
- **注意点 2（Pre-Norm）**：LayerNorm 位于子层之前，不应放在残差相加之后。
- **注意点 3（确定性测试）**：实现不引入默认 dropout，避免两次前向的随机差异干扰因果性检查。

### 复杂度分析
- **时间复杂度**：$O(BS^2D+BSD^2)$，注意力和 MLP 是主要开销。
- **空间复杂度**：$O(BHS^2+BSD)$，注意力矩阵占主要激活空间。""",
}
