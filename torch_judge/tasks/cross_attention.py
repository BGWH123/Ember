"""Multi-Head Cross-Attention task."""

TASK = {
    'title': 'Multi-Head Cross-Attention',
    'title_zh': '多头交叉注意力',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'MultiHeadCrossAttention',
    'starter_code': """class MultiHeadCrossAttention(nn.Module):
    pass""",
    'hint': """Q from `x_q`, K/V from `x_kv`. Project → reshape to `(B, H, S, d_k)` → scaled dot-product (no causal mask) → concat heads → `W_o`.""",
    'hint_zh': 'Q 来自 `x_q`，K/V 来自 `x_kv`。投影 → reshape 为 `(B, H, S, d_k)` → 缩放点积（无因果遮蔽）→ 拼接各头 → `W_o`。',
    'description_en': """Implement multi-head cross-attention as an nn.Module.

Cross-attention lets a decoder attend to encoder outputs: Q comes from one sequence, K/V from another. No causal mask is applied.

**Signature:** `MultiHeadCrossAttention(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x_q, x_kv) -> Tensor`
- `x_q` — query input (B, S_q, d_model)
- `x_kv` — key/value input (B, S_kv, d_model)

**Returns:** attention output (B, S_q, d_model)

**Constraints:**
- Use separate W_q, W_k, W_v, W_o linear projections
- Q and KV can have different sequence lengths""",
    'description_zh': """实现多头交叉注意力（nn.Module）。

交叉注意力让解码器关注编码器输出：Q 来自一个序列，K/V 来自另一个序列，不使用因果掩码。

**签名:** `MultiHeadCrossAttention(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x_q, x_kv) -> Tensor`
- `x_q` — 查询输入 (B, S_q, d_model)
- `x_kv` — 键/值输入 (B, S_kv, d_model)

**返回:** 注意力输出 (B, S_q, d_model)

**约束:**
- 使用独立的 W_q、W_k、W_v、W_o 线性投影
- Q 和 KV 可以有不同的序列长度""",
    'theory_en': r"""Cross-Attention allows a decoder sequence to attend to an encoder sequence. Q comes from the decoder, while K and V come from the encoder.

**Formula:**
$$\text{Attention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

**Applications:**
- Machine Translation: decoder attends to source sentence
- T2I models (Stable Diffusion): image tokens attend to text CLIP embeddings
- Multimodal LLMs: vision tokens attend to language queries

Unlike self-attention, cross-attention has no causal mask since encoder tokens are fully visible.""",
    'theory_zh': r"""交叉注意力允许解码器序列关注编码器序列。Q 来自解码器，K 和 V 来自编码器。

**公式：**
$$\text{Attention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

**应用：**
- 机器翻译：解码器关注源句子
- 文生图模型（Stable Diffusion）：图像 token 关注文本 CLIP 嵌入
- 多模态 LLM：视觉 token 关注语言查询

与自注意力不同，交叉注意力不使用因果掩码，因为编码器 token 是完全可见的。""",
    'diagram_en': """```mermaid
flowchart LR
    subgraph Encoder
        E[Encoder Output]
    end
    subgraph Decoder
        D[Decoder Input] -->|W_q| Q[Q heads]
    end
    E -->|W_k| K[K heads]
    E -->|W_v| V[V heads]
    Q -->|scaled dot-product| A[Attention]
    K --> A
    A -->|@ V| O[Output]
    V --> O
```""",
    'diagram_zh': """```mermaid
flowchart LR
    subgraph 编码器
        E[编码器输出]
    end
    subgraph 解码器
        D[解码器输入] -->|W_q| Q[Q 多头]
    end
    E -->|W_k| K[K 多头]
    E -->|W_v| V[V 多头]
    Q -->|缩放点积| A[注意力]
    K --> A
    A -->|@ V| O[输出]
    V --> O
```""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch, torch.nn as nn
attn = {fn}(d_model=64, num_heads=4)
assert isinstance(attn, nn.Module), 'Must inherit from nn.Module'
out = attn(torch.randn(2, 6, 64), torch.randn(2, 10, 64))
assert out.shape == (2, 6, 64), f'Output shape: {out.shape}'""",
        },
        {
            "name": 'Q and KV different lengths',
            "code": """import torch
attn = {fn}(d_model=32, num_heads=2)
out = attn(torch.randn(1, 3, 32), torch.randn(1, 20, 32))
assert out.shape == (1, 3, 32), f'Shape: {out.shape}'""",
        },
        {
            "name": 'No causal mask — all KV affects all Q',
            "code": """import torch
torch.manual_seed(0)
attn = {fn}(d_model=32, num_heads=2)
x_q = torch.randn(1, 4, 32)
x_kv = torch.randn(1, 6, 32)
out1 = attn(x_q, x_kv)
x_kv2 = x_kv.clone()
x_kv2[:, -1] = torch.randn(1, 32)
out2 = attn(x_q, x_kv2)
assert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Changing last KV should affect all Q positions'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
attn = {fn}(d_model=32, num_heads=2)
x_q = torch.randn(1, 4, 32, requires_grad=True)
x_kv = torch.randn(1, 6, 32, requires_grad=True)
attn(x_q, x_kv).sum().backward()
assert x_q.grad is not None and x_kv.grad is not None, 'Missing gradients'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch, torch.nn as nn
torch.manual_seed(5)
B, S_q, S_kv, D, num_heads = 2, 4, 6, 32, 4
attn = {fn}(d_model=D, num_heads=num_heads)
attn.eval()
x_q  = torch.randn(B, S_q,  D)
x_kv = torch.randn(B, S_kv, D)
with torch.no_grad():
    out = attn(x_q, x_kv)
    d_k = D // num_heads
    # Project using module weights
    Q = x_q  @ (((((((attn.W_q if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q).weight.T + (((((((attn.W_q if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q) if hasattr(attn, 'W_q') else attn.w_q).bias   # (B, S_q,  D)
    K = x_kv @ (((((((attn.W_k if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k).weight.T + (((((((attn.W_k if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k) if hasattr(attn, 'W_k') else attn.w_k).bias   # (B, S_kv, D)
    V = x_kv @ (((((((attn.W_v if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v).weight.T + (((((((attn.W_v if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v) if hasattr(attn, 'W_v') else attn.w_v).bias   # (B, S_kv, D)
    # Split heads: (B, H, S, d_k)
    Q = Q.view(B, S_q,  num_heads, d_k).transpose(1, 2)
    K = K.view(B, S_kv, num_heads, d_k).transpose(1, 2)
    V = V.view(B, S_kv, num_heads, d_k).transpose(1, 2)
    scores  = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    ctx     = (weights @ V).transpose(1, 2).contiguous().view(B, S_q, D)
    expected = ctx @ (((((((attn.W_o if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o).weight.T + (((((((attn.W_o if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o) if hasattr(attn, 'W_o') else attn.w_o).bias
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff {(out - expected).abs().max()}'""",
        },
    ],
    'solution': '''class MultiHeadCrossAttention(nn.Module):
    """实现多头交叉注意力 (Multi-Head Cross-Attention)。"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        B, S_q, D = x_q.shape
        S_k = x_kv.shape[1]

        # Step 1: 分别投影查询与键值，并拆分多头
        query = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(x_kv).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(x_kv).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

        # Step 2: 计算无因果掩码的缩放点积注意力
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ value

        # Step 3: 合并各头并执行输出投影
        context = context.transpose(1, 2).contiguous().view(B, S_q, D)
        return self.W_o(context)''',
    'demo': """torch.manual_seed(42)
B, S_q, S_k, D = 2, 6, 10, 64
layer = MultiHeadCrossAttention(D, 4)
out = layer(torch.randn(B, S_q, D), torch.randn(B, S_k, D))
print("Output shape:", out.shape)
print("Sample values:", out[0, 0, :4])""",
    'explanation': r"""### 核心思路
查询来自 `x_q`，键和值来自 `x_kv`；每个头计算 $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$，再拼接并投影回模型维度。

### 代码步骤拆解
- **Step 1（多头投影）**：将 `(B, S_q, D)` 与 `(B, S_k, D)` 投影并重排为 `(B, H, S, d_k)`。
- **Step 2（注意力聚合）**：`scores` 形状为 `(B, H, S_q, S_k)`，沿键序列维执行 softmax 后乘 `value`。
- **Step 3（合并输出）**：将 `context` 转回 `(B, S_q, D)`，经 `W_o` 得到最终结果。

### 重点与易错点分析
- **注意点 1（维度约束）**：`d_model` 必须能被 `num_heads` 整除，否则无法均匀拆头。
- **注意点 2（缩放位置）**：softmax 前除以 $\sqrt{d_k}$ 可避免维度增大导致分数过尖和梯度不稳定。
- **注意点 3（序列长度）**：`S_q` 与 `S_k` 可以不同，交叉注意力不应误加自回归因果掩码。

### 复杂度分析
- **时间复杂度**：$O(BS_qS_kD)$，主要瓶颈是查询与键的两次批量矩阵乘法。
- **空间复杂度**：$O(BHS_qS_k)$，注意力权重矩阵占据主要显存。""",
}
