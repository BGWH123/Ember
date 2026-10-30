"""Complete Transformer Encoder task."""

TASK = {
    'title': 'Complete Transformer Encoder',
    'title_zh': '完整 Transformer 编码器',
    'difficulty': 'Hard',
    'category': 'Transformer组件',
    'function_name': 'TransformerEncoder',
    'starter_code': """class TransformerEncoder(nn.Module):
    pass""",
    'hint': """1. `self.embedding = nn.Embedding(vocab_size, d_model)`
2. `self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))`
3. For each layer: `self_attn -> ln -> ffn -> ln` with residuals
4. Use `nn.TransformerEncoderLayer` is NOT allowed — implement from scratch""",
    'hint_zh': """1. `self.embedding = nn.Embedding(vocab_size, d_model)`
2. `self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))`
3. 每层：`self_attn -> ln -> ffn -> ln`，带残差连接
4. 不允许使用 `nn.TransformerEncoderLayer` —— 必须从零实现""",
    'description_en': """Implement a complete Transformer Encoder from scratch.

A Transformer Encoder consists of a stack of N identical layers. Each layer has two sub-layers:
1. Multi-Head Self-Attention (with residual connection and LayerNorm)
2. Position-wise Feed-Forward Network (with residual connection and LayerNorm)

**Signature:** `TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input token indices (B, S)

**Returns:** encoded representations (B, S, d_model)

**Constraints:**
- Use token embedding + positional encoding
- Pre-norm architecture: LN before each sub-layer
- FFN: Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)
- Apply dropout after each sub-layer and on embeddings
- All intermediate outputs have shape (B, S, d_model)""",
    'description_zh': """从零实现完整的 Transformer 编码器。

Transformer 编码器由 N 个相同的层堆叠而成。每层包含两个子层：
1. 多头自注意力（带残差连接和 LayerNorm）
2. 逐位置前馈网络（带残差连接和 LayerNorm）

**签名:** `TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入 token 索引 (B, S)

**返回:** 编码表示 (B, S, d_model)

**约束:**
- 使用 token 嵌入 + 位置编码
- Pre-norm 架构：每个子层之前做 LN
- FFN：Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)
- 每个子层后和嵌入层上应用 dropout
- 所有中间输出形状为 (B, S, d_model)""",
    'theory_en': r"""The Transformer Encoder, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), revolutionized NLP by replacing recurrence with pure attention.

**Architecture:**
$$\text{Output} = \text{LayerNorm}(x + \text{SelfAttn}(\text{LayerNorm}(x)))$$
$$\text{Output} = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x)))$$

**Positional Encoding (learned):**
$$X_{emb} = \text{Embed}(x) + \text{PosEmbed}[0:S]$$

**FFN:**
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$
Typically $d_{ff} = 4 \times d_{model}$

**Why Pre-norm?**
Pre-norm places LayerNorm before the sub-layer, making training more stable for deep stacks. Post-norm (original) places it after, which can cause gradient vanishing in very deep networks.""",
    'theory_zh': r"""Transformer 编码器在《Attention Is All You Need》(Vaswani et al., 2017) 中提出，通过纯注意力机制替代循环，彻底改变了 NLP。

**架构：**
$$\text{Output} = \text{LayerNorm}(x + \text{SelfAttn}(\text{LayerNorm}(x)))$$
$$\text{Output} = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x)))$$

**位置编码（可学习）：**
$$X_{emb} = \text{Embed}(x) + \text{PosEmbed}[0:S]$$

**前馈网络：**
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$
通常 $d_{ff} = 4 \times d_{model}$

**为什么用 Pre-norm？**
Pre-norm 将 LayerNorm 放在子层之前，使深层堆叠训练更稳定。Post-norm（原始版本）放在之后，在极深网络中可能导致梯度消失。""",
    'diagram_en': """```mermaid
flowchart TD
    Input[Input tokens] --> Embed[Token Embedding + Positional Encoding]
    Embed --> Drop[Dropout]
    Drop --> Layer1[Encoder Layer 1]
    Layer1 --> Layer2[Encoder Layer 2]
    Layer2 --> LayerN[...]
    LayerN --> Output[Output representations]

    subgraph EncoderLayer
        direction TB
        X --> LN1[LayerNorm]
        LN1 --> MHA[Multi-Head Self-Attention]
        MHA --> ADD1[Residual: x + MHA]
        X --> ADD1
        ADD1 --> LN2[LayerNorm]
        LN2 --> FFN[FFN: Linear->ReLU->Linear]
        FFN --> ADD2[Residual: x + FFN]
        ADD1 --> ADD2
    end
```""",
    'diagram_zh': """```mermaid
flowchart TD
    Input[输入 token] --> Embed[Token 嵌入 + 位置编码]
    Embed --> Drop[Dropout]
    Drop --> Layer1[编码器层 1]
    Layer1 --> Layer2[编码器层 2]
    Layer2 --> LayerN[...]
    LayerN --> Output[输出表示]

    subgraph 编码器层
        direction TB
        X --> LN1[LayerNorm]
        LN1 --> MHA[多头自注意力]
        MHA --> ADD1[残差: x + MHA]
        X --> ADD1
        ADD1 --> LN2[LayerNorm]
        LN2 --> FFN[FFN: Linear->ReLU->Linear]
        FFN --> ADD2[残差: x + FFN]
        ADD1 --> ADD2
    end
```""",
    "tests": [
        {
            "name": 'Is nn.Module',
            "code": """import torch, torch.nn as nn
model = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
assert isinstance(model, nn.Module), 'Must inherit from nn.Module'""",
        },
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
model = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 10))
out = model(x)
assert out.shape == (2, 10, 64), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": 'Has embedding and positional encoding',
            "code": """import torch, torch.nn as nn
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
assert hasattr(model, 'token_embed') or hasattr(model, 'embedding'), 'Need token embedding'
has_pos = hasattr(model, 'pos_embed') or hasattr(model, 'pos_encoding')
assert has_pos, 'Need positional encoding'""",
        },
        {
            "name": 'Has correct number of layers',
            "code": """import torch
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=3)
layer_attr = getattr(model, 'layers', None) or getattr(model, 'encoder_layers', None)
assert layer_attr is not None, 'Need self.layers or self.encoder_layers'
assert len(layer_attr) == 3, f'Expected 3 layers, got {len(layer_attr)}'""",
        },
        {
            "name": 'Gradient flows to embedding',
            "code": """import torch
torch.manual_seed(0)
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=2)
x = torch.randint(0, 50, (2, 8))
out = model(x)
out.sum().backward()
embed = getattr(model, 'token_embed', None) or getattr(model, 'embedding')
assert embed.weight.grad is not None, 'Embedding gradients missing'""",
        },
        {
            "name": 'Causal mask NOT applied in encoder',
            "code": """import torch
torch.manual_seed(0)
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
model.eval()
x = torch.randint(0, 50, (1, 5))
out1 = model(x)
x2 = x.clone()
x2[:, 2:] = torch.randint(0, 50, (1, 3))
out2 = model(x2)
# If causal, earlier positions would be unchanged; encoder should change all
assert not torch.allclose(out1[:, :2], out2[:, :2], atol=1e-5), 'Encoder should not use causal mask'""",
        },
    ],
    'solution': '''class TransformerEncoderLayer(nn.Module):
    """实现完整 Transformer 编码器（Complete Transformer Encoder）。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)
        ffn_out = self.linear2(self.activation(self.linear1(self.norm2(x))))
        x = x + self.dropout(ffn_out)
        return x

class TransformerEncoder(nn.Module):
    """实现完整 Transformer 编码器（Complete Transformer Encoder）。"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        # Step 1: 构造缩放 token embedding、全序列位置编码与 N 个 pre-norm 编码层
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe.unsqueeze(0))
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Step 2: 让每层所有位置双向自注意，再经逐位置 FFN 更新残差流
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        # Step 3: 对层栈输出做最终 LayerNorm，返回每个 token 的双向上下文表示
        return self.norm(x)''',
    'demo': """torch.manual_seed(42)
model = TransformerEncoder(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 10))
out = model(x)
print('Input shape:', x.shape)
print('Output shape:', out.shape)
print('Parameters:', sum((p.numel() for p in model.parameters())))""",
    'explanation': r"""### 核心思路
编码器以双向多头自注意力混合整段序列，再用共享于各位置的 FFN 变换通道。每层采用 pre-norm：$Y=X+\operatorname{MHA}(\operatorname{LN}_1X)$，$X'=Y+W_2\operatorname{GELU}(W_1\operatorname{LN}_2Y)$；残差主干不经过归一化或非线性，便于深层梯度传播。

### 代码步骤拆解
- **Step 1（注入词义与位置）**：`embedding(x)*sqrt(d_model)` 产生 `(B,S,D)` token 表示，叠加 `pos_encoding[:,:S]` 以打破注意力的置换不变性。
- **Step 2（双向层栈编码）**：每个 `TransformerEncoderLayer` 在无因果 mask 下令所有位置互相注意，然后通过 `D→d_ff→D` 的 GELU-FFN 和两条残差更新表示。
- **Step 3（输出尺度校准）**：最终 `LayerNorm` 归一化最后一维，返回同为 `(B,S,D)` 的上下文化 token 序列。

### 重点与易错点分析
- **注意点 1（不能使用因果掩码）**：编码器位置 $i$ 应能依赖任意位置 $j$；加上上三角 mask 会把双向编码错误改成单向语言模型。
- **注意点 2（位置编码与设备）**：`pos_encoding` 注册为 buffer，既不参与优化，又会随 `.to(device)` 和 `state_dict` 移动；切片长度不得超过 `max_len`。
- **注意点 3（pre-norm 语义）**：归一化应作用于子层输入而非残差和；最终额外 LayerNorm 用于输出尺度稳定，这与原始 post-norm Transformer 的次序不同。

### 复杂度分析
- **时间复杂度**：$L$ 层总计 $O(LB(S^2D+SD^2+SDD_{ff}))$，长序列时注意力分数计算 $S^2D$ 主导。
- **空间复杂度**：训练时注意力权重与层激活约为 $O(LB(S^2+SD_{ff}))$；仅推理当前层可复用空间而降至 $O(B(S^2+SD_{ff}))$。""",
}
