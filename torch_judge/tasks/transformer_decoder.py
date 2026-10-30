"""Transformer Decoder with Cross-Attention task."""

TASK = {
    'title': 'Transformer Decoder with Cross-Attention',
    'title_zh': '带交叉注意力的 Transformer 解码器',
    'difficulty': 'Hard',
    'category': 'Transformer组件',
    'function_name': 'TransformerDecoder',
    'starter_code': """class TransformerDecoder(nn.Module):
    pass""",
    'hint': """1. `self.token_embed = nn.Embedding(vocab_size, d_model)`
2. Causal mask: `torch.triu(torch.ones(S,S), diagonal=1).bool()` → masked_fill with -inf
3. Cross-attention: Q from decoder, K/V from encoder_output
4. Each layer: self_attn -> ln -> cross_attn -> ln -> ffn -> ln""",
    'hint_zh': """1. `self.token_embed = nn.Embedding(vocab_size, d_model)`
2. 因果掩码：`torch.triu(torch.ones(S,S), diagonal=1).bool()` → 用 -inf 填充
3. 交叉注意力：Q 来自解码器，K/V 来自 encoder_output
4. 每层：self_attn -> ln -> cross_attn -> ln -> ffn -> ln""",
    'description_en': """Implement a complete Transformer Decoder from scratch.

A Transformer Decoder consists of a stack of N identical layers. Each layer has three sub-layers:
1. Masked Multi-Head Self-Attention (causal, with residual + LN)
2. Multi-Head Cross-Attention (attends to encoder output, with residual + LN)
3. Position-wise Feed-Forward Network (with residual + LN)

**Signature:** `TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)` (nn.Module)

**Forward:** `forward(x, encoder_output) -> Tensor`
- `x` — decoder input token indices (B, S_dec)
- `encoder_output` — encoder output (B, S_enc, d_model)

**Returns:** decoded representations (B, S_dec, d_model)

**Constraints:**
- Causal mask in self-attention (future tokens cannot attend to past)
- No causal mask in cross-attention
- Pre-norm architecture
- Use token embedding + positional encoding""",
    'description_zh': """从零实现完整的 Transformer 解码器。

Transformer 解码器由 N 个相同的层堆叠而成。每层包含三个子层：
1. 掩码多头自注意力（因果性，带残差 + LN）
2. 多头交叉注意力（关注编码器输出，带残差 + LN）
3. 逐位置前馈网络（带残差 + LN）

**签名:** `TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1)`（nn.Module）

**前向传播:** `forward(x, encoder_output) -> Tensor`
- `x` — 解码器输入 token 索引 (B, S_dec)
- `encoder_output` — 编码器输出 (B, S_enc, d_model)

**返回:** 解码表示 (B, S_dec, d_model)

**约束:**
- 自注意力中使用因果掩码（未来 token 不能关注过去）
- 交叉注意力中不使用因果掩码
- Pre-norm 架构
- 使用 token 嵌入 + 位置编码""",
    'theory_en': r"""The Transformer Decoder extends the encoder with cross-attention to the encoder output and causal masking for autoregressive generation.

**Architecture per layer:**
$$x = \text{LN}(x + \text{MaskedSelfAttn}(\text{LN}(x)))$$
$$x = \text{LN}(x + \text{CrossAttn}(\text{LN}(x), \text{enc_out}))$$
$$x = \text{LN}(x + \text{FFN}(\text{LN}(x)))$$

**Causal Mask:**
$$M_{ij} = \begin{cases} 0 & i \le j \\ -\infty & i > j \end{cases}$$
This ensures position $i$ can only attend to positions $\le i$.

**Cross-Attention:**
Q is projected from decoder hidden states, while K and V come from the encoder output. This allows the decoder to 'query' the source representation for relevant information.""",
    'theory_zh': r"""Transformer 解码器通过交叉注意力扩展了编码器，并加入因果掩码实现自回归生成。

**每层架构：**
$$x = \text{LN}(x + \text{MaskedSelfAttn}(\text{LN}(x)))$$
$$x = \text{LN}(x + \text{CrossAttn}(\text{LN}(x), \text{enc_out}))$$
$$x = \text{LN}(x + \text{FFN}(\text{LN}(x)))$$

**因果掩码：**
$$M_{ij} = \begin{cases} 0 & i \le j \\ -\infty & i > j \end{cases}$$
这确保位置 $i$ 只能关注位置 $\le i$。

**交叉注意力：**
Q 从解码器隐藏状态投影，K 和 V 来自编码器输出。这使解码器能够向源表示'查询'相关信息。""",
    'diagram_en': """```mermaid
flowchart TD
    Input[Decoder Input] --> Embed[Token Embedding + Positional Encoding]
    Embed --> Layer1[Decoder Layer 1]
    Layer1 --> LayerN[...]
    Enc[Encoder Output] --> Layer1
    LayerN --> Output[Output]

    subgraph DecoderLayer
        direction TB
        X --> LN1[LayerNorm]
        LN1 --> MSA[Masked Self-Attention]
        MSA --> ADD1[x + MSA]
        X --> ADD1
        ADD1 --> LN2[LayerNorm]
        LN2 --> CA[Cross-Attention<br/>Q from decoder, KV from encoder]
        EncOut[Encoder Output] --> CA
        CA --> ADD2[x + CA]
        ADD1 --> ADD2
        ADD2 --> LN3[LayerNorm]
        LN3 --> FFN[FFN]
        FFN --> ADD3[x + FFN]
        ADD2 --> ADD3
    end
```""",
    'diagram_zh': """```mermaid
flowchart TD
    Input[解码器输入] --> Embed[Token 嵌入 + 位置编码]
    Embed --> Layer1[解码器层 1]
    Layer1 --> LayerN[...]
    Enc[编码器输出] --> Layer1
    LayerN --> Output[输出]

    subgraph 解码器层
        direction TB
        X --> LN1[LayerNorm]
        LN1 --> MSA[掩码自注意力]
        MSA --> ADD1[x + MSA]
        X --> ADD1
        ADD1 --> LN2[LayerNorm]
        LN2 --> CA[交叉注意力<br/>Q 来自解码器, KV 来自编码器]
        EncOut[编码器输出] --> CA
        CA --> ADD2[x + CA]
        ADD1 --> ADD2
        ADD2 --> LN3[LayerNorm]
        LN3 --> FFN[FFN]
        FFN --> ADD3[x + FFN]
        ADD2 --> ADD3
    end
```""",
    "tests": [
        {
            "name": 'Is nn.Module',
            "code": """import torch, torch.nn as nn
dec = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
assert isinstance(dec, nn.Module), 'Must inherit from nn.Module'""",
        },
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
dec = {fn}(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 8))
enc_out = torch.randn(2, 12, 64)
out = dec(x, enc_out)
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": 'Causal masking in self-attention',
            "code": """import torch
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
assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5), 'Causal mask violated: future changes affected past'""",
        },
        {
            "name": 'Cross-attention uses encoder output',
            "code": """import torch
torch.manual_seed(0)
dec = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
dec.eval()
x = torch.randint(0, 50, (1, 4))
enc1 = torch.randn(1, 10, 32)
enc2 = torch.randn(1, 10, 32)
out1 = dec(x, enc1)
out2 = dec(x, enc2)
assert not torch.allclose(out1, out2, atol=1e-5), 'Changing encoder output should change decoder output'""",
        },
        {
            "name": 'Has correct number of layers',
            "code": """import torch
dec = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=4)
layer_attr = getattr(dec, 'layers', None) or getattr(dec, 'decoder_layers', None)
assert layer_attr is not None, 'Need self.layers or self.decoder_layers'
assert len(layer_attr) == 4, f'Expected 4 layers, got {len(layer_attr)}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(0)
dec = {fn}(vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=1)
x = torch.randint(0, 50, (2, 6))
enc_out = torch.randn(2, 10, 32, requires_grad=True)
out = dec(x, enc_out)
out.sum().backward()
assert enc_out.grad is not None, 'Gradient must flow to encoder_output'""",
        },
    ],
    'solution': '''class TransformerDecoderLayer(nn.Module):
    """实现带交叉注意力的 Transformer 解码器（Transformer Decoder with Cross-Attention）。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, memory):
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        self_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(self_out)
        cross_out, _ = self.cross_attn(self.norm2(x), memory, memory, need_weights=False)
        x = x + self.dropout(cross_out)
        ffn_out = self.linear2(self.activation(self.linear1(self.norm3(x))))
        x = x + self.dropout(ffn_out)
        return x

class TransformerDecoder(nn.Module):
    """实现带交叉注意力的 Transformer 解码器（Transformer Decoder with Cross-Attention）。"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        # Step 1: 构造缩放 token embedding、正弦位置编码与 N 层 pre-norm 解码块
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # Step 2: 逐层执行因果自注意力、以 memory 为 K/V 的交叉注意力及 FFN
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory)
        # Step 3: 对最后一层残差流归一化，输出每个目标位置的上下文化表示
        return self.norm(x)''',
    'demo': """torch.manual_seed(42)
dec = TransformerDecoder(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
x = torch.randint(0, 100, (2, 8))
enc_out = torch.randn(2, 12, 64)
out = dec(x, enc_out)
print('Decoder input:', x.shape)
print('Encoder output:', enc_out.shape)
print('Decoder output:', out.shape)""",
    'explanation': r"""### 核心思路
每个 pre-norm 解码层包含三条残差更新：$X\leftarrow X+\operatorname{SelfAttn}(\operatorname{LN}_1X;M)$、$X\leftarrow X+\operatorname{CrossAttn}(\operatorname{LN}_2X,E,E)$、$X\leftarrow X+\operatorname{FFN}(\operatorname{LN}_3X)$。上三角因果掩码 $M_{ij}=1$（当 $j>i$）阻止目标位置读取未来 token，而交叉注意力允许每个目标位置读取全部编码器 memory。

### 代码步骤拆解
- **Step 1（嵌入与层栈构造）**：token embedding 乘 $\sqrt D$ 后叠加 `(1,S_dec,D)` 正弦位置编码，并建立 `num_layers` 个包含三子层的解码块。
- **Step 2（双源注意力更新）**：每层先以归一化 `x` 同时作 Q/K/V 执行因果自注意力，再以解码状态作 Q、`memory` 作 K/V 计算 `(B,S_dec,S_enc)` 交叉注意力，最后经过 GELU-FFN。
- **Step 3（最终归一化）**：全部残差更新后由 `self.norm` 稳定最后的特征尺度，返回 `(B,S_dec,D)` 而非词表 logits。

### 重点与易错点分析
- **注意点 1（掩码方向）**：`triu(..., diagonal=1)` 屏蔽行位置右侧的未来列；反向三角会错误地阻止读取历史。该 mask 只能传给 self-attention。
- **注意点 2（交叉注意力来源）**：Q 的长度是 `S_dec`，K/V 长度是 `S_enc`；把 K/V 也取自 `x` 会退化为第二个自注意力并切断源序列信息。
- **注意点 3（pre-norm 与自回归）**：LayerNorm 位于子层输入且残差主干保持恒等路径，有利于深层梯度；训练时整段计算仍须因果 mask，增量推理则可缓存历史 K/V。

### 复杂度分析
- **时间复杂度**：$L$ 层为 $O\!\left(LB(S_{dec}^2D+S_{dec}S_{enc}D+(S_{dec}+S_{enc})D^2+S_{dec}DD_{ff})\right)$，两类注意力矩阵与投影/FFN 为主耗时。
- **空间复杂度**：训练激活约为 $O\!\left(LB(S_{dec}^2+S_{dec}S_{enc}+S_{dec}D_{ff})\right)$，其中两张注意力图是序列长度相关的主要开销。""",
}
