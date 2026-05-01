"""Complete Transformer Encoder task."""

TASK = {
    "title": "Complete Transformer Encoder",
    "title_zh": "完整 Transformer 编码器",
    "difficulty": "Hard",
    "category": "Transformer组件",
    "description_en": (
        "Implement a complete Transformer Encoder from scratch.\n\nA Transformer Encoder consists of a stack of N identical layers. Each layer has two sub-layers:\n1. Multi-Head Self-Attention (with residual connection and LayerNorm)\n2. Position-wise Feed-Forward Network (with residual connection and LayerNorm)\n\n**Signature:** `TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input token indices (B, S)\n\n**Returns:** encoded representations (B, S, d_model)\n\n**Constraints:**\n- Use token embedding + positional encoding\n- Pre-norm architecture: LN before each sub-layer\n- FFN: Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)\n- Apply dropout after each sub-layer and on embeddings\n- All intermediate outputs have shape (B, S, d_model)"
    ),
    "description_zh": (
        "从零实现完整的 Transformer 编码器。\n\nTransformer 编码器由 N 个相同的层堆叠而成。每层包含两个子层：\n1. 多头自注意力（带残差连接和 LayerNorm）\n2. 逐位置前馈网络（带残差连接和 LayerNorm）\n\n**签名:** `TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入 token 索引 (B, S)\n\n**返回:** 编码表示 (B, S, d_model)\n\n**约束:**\n- 使用 token 嵌入 + 位置编码\n- Pre-norm 架构：每个子层之前做 LN\n- FFN：Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)\n- 每个子层后和嵌入层上应用 dropout\n- 所有中间输出形状为 (B, S, d_model)"
    ),
    "function_name": "TransformerEncoder",
    "hint": (
        "1. `self.embedding = nn.Embedding(vocab_size, d_model)`\n2. `self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))`\n3. For each layer: `self_attn -> ln -> ffn -> ln` with residuals\n4. Use `nn.TransformerEncoderLayer` is NOT allowed — implement from scratch"
    ),
    "hint_zh": (
        "1. `self.embedding = nn.Embedding(vocab_size, d_model)`\n2. `self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))`\n3. 每层：`self_attn -> ln -> ffn -> ln`，带残差连接\n4. 不允许使用 `nn.TransformerEncoderLayer` —— 必须从零实现"
    ),
    "theory_en": (
        "The Transformer Encoder, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), revolutionized NLP by replacing recurrence with pure attention.\n\n**Architecture:**\n$$\\text{Output} = \\text{LayerNorm}(x + \\text{SelfAttn}(\\text{LayerNorm}(x)))$$\n$$\\text{Output} = \\text{LayerNorm}(x + \\text{FFN}(\\text{LayerNorm}(x)))$$\n\n**Positional Encoding (learned):**\n$$X_{emb} = \\text{Embed}(x) + \\text{PosEmbed}[0:S]$$\n\n**FFN:**\n$$\\text{FFN}(x) = W_2 \\cdot \\text{ReLU}(W_1 x + b_1) + b_2$$\nTypically $d_{ff} = 4 \\times d_{model}$\n\n**Why Pre-norm?**\nPre-norm places LayerNorm before the sub-layer, making training more stable for deep stacks. Post-norm (original) places it after, which can cause gradient vanishing in very deep networks."
    ),
    "theory_zh": (
        "Transformer 编码器在《Attention Is All You Need》(Vaswani et al., 2017) 中提出，通过纯注意力机制替代循环，彻底改变了 NLP。\n\n**架构：**\n$$\\text{Output} = \\text{LayerNorm}(x + \\text{SelfAttn}(\\text{LayerNorm}(x)))$$\n$$\\text{Output} = \\text{LayerNorm}(x + \\text{FFN}(\\text{LayerNorm}(x)))$$\n\n**位置编码（可学习）：**\n$$X_{emb} = \\text{Embed}(x) + \\text{PosEmbed}[0:S]$$\n\n**前馈网络：**\n$$\\text{FFN}(x) = W_2 \\cdot \\text{ReLU}(W_1 x + b_1) + b_2$$\n通常 $d_{ff} = 4 \\times d_{model}$\n\n**为什么用 Pre-norm？**\nPre-norm 将 LayerNorm 放在子层之前，使深层堆叠训练更稳定。Post-norm（原始版本）放在之后，在极深网络中可能导致梯度消失。"
    ),
    "diagram_en": (
        "```mermaid\nflowchart TD\n    Input[Input tokens] --> Embed[Token Embedding + Positional Encoding]\n    Embed --> Drop[Dropout]\n    Drop --> Layer1[Encoder Layer 1]\n    Layer1 --> Layer2[Encoder Layer 2]\n    Layer2 --> LayerN[...]\n    LayerN --> Output[Output representations]\n\n    subgraph EncoderLayer\n        direction TB\n        X --> LN1[LayerNorm]\n        LN1 --> MHA[Multi-Head Self-Attention]\n        MHA --> ADD1[Residual: x + MHA]\n        X --> ADD1\n        ADD1 --> LN2[LayerNorm]\n        LN2 --> FFN[FFN: Linear->ReLU->Linear]\n        FFN --> ADD2[Residual: x + FFN]\n        ADD1 --> ADD2\n    end\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart TD\n    Input[输入 token] --> Embed[Token 嵌入 + 位置编码]\n    Embed --> Drop[Dropout]\n    Drop --> Layer1[编码器层 1]\n    Layer1 --> Layer2[编码器层 2]\n    Layer2 --> LayerN[...]\n    LayerN --> Output[输出表示]\n\n    subgraph 编码器层\n        direction TB\n        X --> LN1[LayerNorm]\n        LN1 --> MHA[多头自注意力]\n        MHA --> ADD1[残差: x + MHA]\n        X --> ADD1\n        ADD1 --> LN2[LayerNorm]\n        LN2 --> FFN[FFN: Linear->ReLU->Linear]\n        FFN --> ADD2[残差: x + FFN]\n        ADD1 --> ADD2\n    end\n```"
    ),
    "tests": [
        {
            "name": "Is nn.Module",
            "code": """








import torch, torch.nn as nn
model = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
assert isinstance(model, nn.Module), 'Must inherit from nn.Module'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Output shape",
            "code": """








import torch
torch.manual_seed(0)
model = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 10))
out = model(x)
assert out.shape == (2, 10, 64), f'Shape mismatch: {out.shape}'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Has embedding and positional encoding",
            "code": """








import torch, torch.nn as nn
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
assert hasattr(model, 'token_embed') or hasattr(model, 'embedding'), 'Need token embedding'
has_pos = hasattr(model, 'pos_embed') or hasattr(model, 'pos_encoding')
assert has_pos, 'Need positional encoding'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Has correct number of layers",
            "code": """








import torch
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=3)
layer_attr = getattr(model, 'layers', None) or getattr(model, 'encoder_layers', None)
assert layer_attr is not None, 'Need self.layers or self.encoder_layers'
assert len(layer_attr) == 3, f'Expected 3 layers, got {len(layer_attr)}'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flows to embedding",
            "code": """








import torch
torch.manual_seed(0)
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=2)
x = torch.randint(0, 50, (2, 8))
out = model(x)
out.sum().backward()
embed = getattr(model, 'token_embed', None) or getattr(model, 'embedding')
assert embed.weight.grad is not None, 'Embedding gradients missing'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Causal mask NOT applied in encoder",
            "code": """








import torch
torch.manual_seed(0)
model = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
model.eval()
x = torch.randint(0, 50, (1, 5))
out1 = model(x)
x2 = x.clone()
x2[:, 2:] = torch.randint(0, 50, (1, 3))
out2 = model(x2)
# If causal, earlier positions would be unchanged; encoder should change all
assert not torch.allclose(out1[:, :2], out2[:, :2], atol=1e-5), 'Encoder should not use causal mask'

            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''





class TransformerEncoderLayer(nn.Module):
    """
    单个 Transformer Encoder Layer (Pre-Norm + Residual).
    结构: LN -> Self-Attention -> Residual -> LN -> FFN -> Residual
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)          # FFN 第一层: d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)          # FFN 第二层: d_ff -> d_model
        self.norm1 = nn.LayerNorm(d_model)               # Attention 前的 LayerNorm
        self.norm2 = nn.LayerNorm(d_model)               # FFN 前的 LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()                      # FFN 激活函数

    def forward(self, x):
        # Pre-Norm Self-Attention + 残差
        # self_attn(Q,K,V) 输入均为 self.norm1(x)，因为是自注意力
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)                   # 残差连接

        # Pre-Norm FFN + 残差
        # FFN(x) = Linear2(Activation(Linear1(Norm2(x))))
        ffn_out = self.linear2(self.activation(self.linear1(self.norm2(x))))
        x = x + self.dropout(ffn_out)                    # 残差连接
        return x


class TransformerEncoder(nn.Module):
    """
    完整的 Transformer Encoder (堆叠 N 层)。
    包含词嵌入、正弦位置编码、N 个 Encoder Layer、最终 LayerNorm。
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 词嵌入层: 将整数 token ID 映射为 d_model 维连续向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 正弦位置编码: 为每个序列位置提供唯一的位置表示
        # 使用不同频率的正弦/余弦函数，使模型能学习相对位置关系
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)       # (max_len, 1)
        # div_term: 频率衰减项，使不同维度的波长呈指数增长
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)     # 偶数维: sin(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)     # 奇数维: cos(pos / 10000^(2i/d_model))
        self.register_buffer('pe', pe.unsqueeze(0))      # (1, max_len, d_model)，不参与梯度更新

        # 堆叠 N 个相同的 Encoder Layer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)                # 最终输出归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Step 1: 词嵌入 + 位置编码
        # 乘以 √(d_model) 缩放嵌入向量，使位置编码与嵌入向量量级相当
        x = self.embedding(x) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = x + self.pe[:, :x.size(1), :]                # 广播相加: (B,S,d_model) + (1,S,d_model)
        x = self.dropout(x)                              # 随机丢弃防止过拟合

        # Step 2: 逐层前向传播
        for layer in self.layers:
            x = layer(x)                                 # 每层保持 (B, S, d_model)

        # Step 3: 最终 LayerNorm
        return self.norm(x)                              # (B, S, d_model)

    
    
    
    
    
    ''',
    "demo": '''







torch.manual_seed(0)
model = TransformerEncoder(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 10))
out = model(x)
print("Input shape:", x.shape)
print("Output shape:", out.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))
    
    
    
    
    
    
    
    ''',
}
