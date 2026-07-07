"""Transformer Decoder with Cross-Attention task."""

TASK = {
    "title": "Transformer Decoder with Cross-Attention",
    "title_zh": "带交叉注意力的 Transformer 解码器",
    "difficulty": "Hard",
    "category": "Transformer组件",
    "description_en": (
        "Implement a complete Transformer Decoder from scratch.\n\nA Transformer Decoder consists of a stack of N identical layers. Each layer has three sub-layers:\n1. Masked Multi-Head Self-Attention (causal, with residual + LN)\n2. Multi-Head Cross-Attention (attends to encoder output, with residual + LN)\n3. Position-wise Feed-Forward Network (with residual + LN)\n\n**Signature:** `TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)` (nn.Module)\n\n**Forward:** `forward(x, encoder_output) -> Tensor`\n- `x` — decoder input token indices (B, S_dec)\n- `encoder_output` — encoder output (B, S_enc, d_model)\n\n**Returns:** decoded representations (B, S_dec, d_model)\n\n**Constraints:**\n- Causal mask in self-attention (future tokens cannot attend to past)\n- No causal mask in cross-attention\n- Pre-norm architecture\n- Use token embedding + positional encoding"
    ),
    "description_zh": (
        "从零实现完整的 Transformer 解码器。\n\nTransformer 解码器由 N 个相同的层堆叠而成。每层包含三个子层：\n1. 掩码多头自注意力（因果性，带残差 + LN）\n2. 多头交叉注意力（关注编码器输出，带残差 + LN）\n3. 逐位置前馈网络（带残差 + LN）\n\n**签名:** `TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)`（nn.Module）\n\n**前向传播:** `forward(x, encoder_output) -> Tensor`\n- `x` — 解码器输入 token 索引 (B, S_dec)\n- `encoder_output` — 编码器输出 (B, S_enc, d_model)\n\n**返回:** 解码表示 (B, S_dec, d_model)\n\n**约束:**\n- 自注意力中使用因果掩码（未来 token 不能关注过去）\n- 交叉注意力中不使用因果掩码\n- Pre-norm 架构\n- 使用 token 嵌入 + 位置编码"
    ),
    "function_name": "TransformerDecoder",
    "hint": (
        "1. `self.token_embed = nn.Embedding(vocab_size, d_model)`\n2. Causal mask: `torch.triu(torch.ones(S,S), diagonal=1).bool()` → masked_fill with -inf\n3. Cross-attention: Q from decoder, K/V from encoder_output\n4. Each layer: self_attn -> ln -> cross_attn -> ln -> ffn -> ln"
    ),
    "hint_zh": (
        "1. `self.token_embed = nn.Embedding(vocab_size, d_model)`\n2. 因果掩码：`torch.triu(torch.ones(S,S), diagonal=1).bool()` → 用 -inf 填充\n3. 交叉注意力：Q 来自解码器，K/V 来自 encoder_output\n4. 每层：self_attn -> ln -> cross_attn -> ln -> ffn -> ln"
    ),
    "theory_en": (
        "The Transformer Decoder extends the encoder with cross-attention to the encoder output and causal masking for autoregressive generation.\n\n**Architecture per layer:**\n$$x = \\text{LN}(x + \\text{MaskedSelfAttn}(\\text{LN}(x)))$$\n$$x = \\text{LN}(x + \\text{CrossAttn}(\\text{LN}(x), \\text{enc_out}))$$\n$$x = \\text{LN}(x + \\text{FFN}(\\text{LN}(x)))$$\n\n**Causal Mask:**\n$$M_{ij} = \\begin{cases} 0 & i \\le j \\\\ -\\infty & i > j \\end{cases}$$\nThis ensures position $i$ can only attend to positions $\\le i$.\n\n**Cross-Attention:**\nQ is projected from decoder hidden states, while K and V come from the encoder output. This allows the decoder to 'query' the source representation for relevant information."
    ),
    "theory_zh": (
        "Transformer 解码器通过交叉注意力扩展了编码器，并加入因果掩码实现自回归生成。\n\n**每层架构：**\n$$x = \\text{LN}(x + \\text{MaskedSelfAttn}(\\text{LN}(x)))$$\n$$x = \\text{LN}(x + \\text{CrossAttn}(\\text{LN}(x), \\text{enc_out}))$$\n$$x = \\text{LN}(x + \\text{FFN}(\\text{LN}(x)))$$\n\n**因果掩码：**\n$$M_{ij} = \\begin{cases} 0 & i \\le j \\\\ -\\infty & i > j \\end{cases}$$\n这确保位置 $i$ 只能关注位置 $\\le i$。\n\n**交叉注意力：**\nQ 从解码器隐藏状态投影，K 和 V 来自编码器输出。这使解码器能够向源表示'查询'相关信息。"
    ),
    "diagram_en": (
        "```mermaid\nflowchart TD\n    Input[Decoder Input] --> Embed[Token Embedding + Positional Encoding]\n    Embed --> Layer1[Decoder Layer 1]\n    Layer1 --> LayerN[...]\n    Enc[Encoder Output] --> Layer1\n    LayerN --> Output[Output]\n\n    subgraph DecoderLayer\n        direction TB\n        X --> LN1[LayerNorm]\n        LN1 --> MSA[Masked Self-Attention]\n        MSA --> ADD1[x + MSA]\n        X --> ADD1\n        ADD1 --> LN2[LayerNorm]\n        LN2 --> CA[Cross-Attention<br/>Q from decoder, KV from encoder]\n        EncOut[Encoder Output] --> CA\n        CA --> ADD2[x + CA]\n        ADD1 --> ADD2\n        ADD2 --> LN3[LayerNorm]\n        LN3 --> FFN[FFN]\n        FFN --> ADD3[x + FFN]\n        ADD2 --> ADD3\n    end\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart TD\n    Input[解码器输入] --> Embed[Token 嵌入 + 位置编码]\n    Embed --> Layer1[解码器层 1]\n    Layer1 --> LayerN[...]\n    Enc[编码器输出] --> Layer1\n    LayerN --> Output[输出]\n\n    subgraph 解码器层\n        direction TB\n        X --> LN1[LayerNorm]\n        LN1 --> MSA[掩码自注意力]\n        MSA --> ADD1[x + MSA]\n        X --> ADD1\n        ADD1 --> LN2[LayerNorm]\n        LN2 --> CA[交叉注意力<br/>Q 来自解码器, KV 来自编码器]\n        EncOut[编码器输出] --> CA\n        CA --> ADD2[x + CA]\n        ADD1 --> ADD2\n        ADD2 --> LN3[LayerNorm]\n        LN3 --> FFN[FFN]\n        FFN --> ADD3[x + FFN]\n        ADD2 --> ADD3\n    end\n```"
    ),
    "tests": [
        {
            "name": "Is nn.Module",
            "code": """








import torch, torch.nn as nn
dec = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
assert isinstance(dec, nn.Module), 'Must inherit from nn.Module'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Output shape",
            "code": """








import torch
torch.manual_seed(0)
dec = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 8))
enc_out = torch.randn(2, 12, 64)
out = dec(x, enc_out)
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Causal masking in self-attention",
            "code": """








import torch
torch.manual_seed(0)
dec = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
dec.eval()
x = torch.randint(0, 50, (1, 6))
enc_out = torch.randn(1, 10, 32)
out1 = dec(x, enc_out)
x2 = x.clone()
x2[:, 3:] = torch.randint(0, 50, (1, 3))
out2 = dec(x2, enc_out)
# Earlier positions should be unchanged due to causal mask
assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5), 'Causal mask violated: future changes affected past'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Cross-attention uses encoder output",
            "code": """








import torch
torch.manual_seed(0)
dec = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
dec.eval()
x = torch.randint(0, 50, (1, 4))
enc1 = torch.randn(1, 10, 32)
enc2 = torch.randn(1, 10, 32)
out1 = dec(x, enc1)
out2 = dec(x, enc2)
assert not torch.allclose(out1, out2, atol=1e-5), 'Changing encoder output should change decoder output'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Has correct number of layers",
            "code": """








import torch
dec = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=4)
layer_attr = getattr(dec, 'layers', None) or getattr(dec, 'decoder_layers', None)
assert layer_attr is not None, 'Need self.layers or self.decoder_layers'
assert len(layer_attr) == 4, f'Expected 4 layers, got {len(layer_attr)}'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """








import torch
torch.manual_seed(0)
dec = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
x = torch.randint(0, 50, (2, 6))
enc_out = torch.randn(2, 10, 32, requires_grad=True)
out = dec(x, enc_out)
out.sum().backward()
assert enc_out.grad is not None, 'Gradient must flow to encoder_output'

            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


class TransformerDecoderLayer(nn.Module):
    """
    单个 Transformer Decoder Layer (Pre-Norm + 3 sub-layers).
    1. Masked Self-Attention (causal, 只能看已生成的 token)
    2. Cross-Attention (Q from decoder, K/V from encoder)
    3. Feed-Forward Network
    每层都有残差连接和 Dropout。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)               # Self-Attention 前
        self.norm2 = nn.LayerNorm(d_model)               # Cross-Attention 前
        self.norm3 = nn.LayerNorm(d_model)               # FFN 前
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, memory):
        # Sub-layer 1: Masked Self-Attention (causal)
        # attn_mask 保证每个位置只能关注自己和之前的位置
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        self_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                      attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(self_out)                   # 残差连接

        # Sub-layer 2: Cross-Attention (decoder attends to encoder output)
        # Q from decoder, K/V from encoder memory
        cross_out, _ = self.cross_attn(self.norm2(x), memory, memory, need_weights=False)
        x = x + self.dropout(cross_out)                  # 残差连接

        # Sub-layer 3: Feed-Forward Network
        ffn_out = self.linear2(self.activation(self.linear1(self.norm3(x))))
        x = x + self.dropout(ffn_out)                    # 残差连接
        return x


class TransformerDecoder(nn.Module):
    """
    完整的 Transformer Decoder (用于 Seq2Seq 或自回归生成)。
    包含词嵌入、位置编码、N 个 Decoder Layer、最终投影到词表。
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # 正弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)     # 投影到词表维度，用于预测下一个 token
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # 词嵌入 + 位置编码 + Dropout
        x = self.embedding(x) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)

        # 逐层通过 Decoder
        for layer in self.layers:
            x = layer(x, memory)                         # (B, S, d_model)

        # 最终归一化 + 投影到词表
        x = self.norm(x)
        return self.fc_out(x)                            # (B, S, vocab_size)

    
    
    ''',
    "demo": '''







torch.manual_seed(0)
dec = TransformerDecoder(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 8))
enc_out = torch.randn(2, 12, 64)
out = dec(x, enc_out)
print("Decoder input:", x.shape)
print("Encoder output:", enc_out.shape)
print("Decoder output:", out.shape)
    
    
    
    
    
    
    
    ''',
}
