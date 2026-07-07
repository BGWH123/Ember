"""Multi-Head Cross-Attention task."""

TASK = {
    "title": "Multi-Head Cross-Attention",
    "title_zh": "多头交叉注意力",
    "difficulty": "Medium",
    "category": "注意力机制",
    "description_en": (
        "Implement multi-head cross-attention as an nn.Module.\n\nCross-attention lets a decoder attend to encoder outputs: Q comes from one sequence, K/V from another. No causal mask is applied.\n\n**Signature:** `MultiHeadCrossAttention(d_model, num_heads)` (nn.Module)\n\n**Forward:** `forward(x_q, x_kv) -> Tensor`\n- `x_q` — query input (B, S_q, d_model)\n- `x_kv` — key/value input (B, S_kv, d_model)\n\n**Returns:** attention output (B, S_q, d_model)\n\n**Constraints:**\n- Use separate W_q, W_k, W_v, W_o linear projections\n- Q and KV can have different sequence lengths"
    ),
    "description_zh": (
        "实现多头交叉注意力（nn.Module）。\n\n交叉注意力让解码器关注编码器输出：Q 来自一个序列，K/V 来自另一个序列，不使用因果掩码。\n\n**签名:** `MultiHeadCrossAttention(d_model, num_heads)`（nn.Module）\n\n**前向传播:** `forward(x_q, x_kv) -> Tensor`\n- `x_q` — 查询输入 (B, S_q, d_model)\n- `x_kv` — 键/值输入 (B, S_kv, d_model)\n\n**返回:** 注意力输出 (B, S_q, d_model)\n\n**约束:**\n- 使用独立的 W_q、W_k、W_v、W_o 线性投影\n- Q 和 KV 可以有不同的序列长度"
    ),
    "function_name": "MultiHeadCrossAttention",
    "hint": "Q from `x_q`, K/V from `x_kv`. Project → reshape to `(B, H, S, d_k)` → scaled dot-product (no causal mask) → concat heads → `W_o`.",
    "hint_zh": "Q 来自 `x_q`，K/V 来自 `x_kv`。投影 → reshape 为 `(B, H, S, d_k)` → 缩放点积（无因果遮蔽）→ 拼接各头 → `W_o`。",
    "theory_en": (
        "Cross-Attention allows a decoder sequence to attend to an encoder sequence. Q comes from the decoder, while K and V come from the encoder.\n\n**Formula:**\n$$\\text{Attention}(Q_{dec}, K_{enc}, V_{enc}) = \\text{softmax}\\left(\\frac{Q_{dec} K_{enc}^T}{\\sqrt{d_k}}\\right) V_{enc}$$\n\n**Applications:**\n- Machine Translation: decoder attends to source sentence\n- T2I models (Stable Diffusion): image tokens attend to text CLIP embeddings\n- Multimodal LLMs: vision tokens attend to language queries\n\nUnlike self-attention, cross-attention has no causal mask since encoder tokens are fully visible."
    ),
    "theory_zh": (
        "交叉注意力允许解码器序列关注编码器序列。Q 来自解码器，K 和 V 来自编码器。\n\n**公式：**\n$$\\text{Attention}(Q_{dec}, K_{enc}, V_{enc}) = \\text{softmax}\\left(\\frac{Q_{dec} K_{enc}^T}{\\sqrt{d_k}}\\right) V_{enc}$$\n\n**应用：**\n- 机器翻译：解码器关注源句子\n- 文生图模型（Stable Diffusion）：图像 token 关注文本 CLIP 嵌入\n- 多模态 LLM：视觉 token 关注语言查询\n\n与自注意力不同，交叉注意力不使用因果掩码，因为编码器 token 是完全可见的。"
    ),
    "diagram_en": (
        "```mermaid\nflowchart LR\n    subgraph Encoder\n        E[Encoder Output]\n    end\n    subgraph Decoder\n        D[Decoder Input] -->|W_q| Q[Q heads]\n    end\n    E -->|W_k| K[K heads]\n    E -->|W_v| V[V heads]\n    Q -->|scaled dot-product| A[Attention]\n    K --> A\n    A -->|@ V| O[Output]\n    V --> O\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart LR\n    subgraph 编码器\n        E[编码器输出]\n    end\n    subgraph 解码器\n        D[解码器输入] -->|W_q| Q[Q 多头]\n    end\n    E -->|W_k| K[K 多头]\n    E -->|W_v| V[V 多头]\n    Q -->|缩放点积| A[注意力]\n    K --> A\n    A -->|@ V| O[输出]\n    V --> O\n```"
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """









import torch, torch.nn as nn
attn = {fn}(d_model=64, num_heads=4)
assert isinstance(attn, nn.Module), 'Must inherit from nn.Module'
out = attn(torch.randn(2, 6, 64), torch.randn(2, 10, 64))
assert out.shape == (2, 6, 64), f'Output shape: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Q and KV different lengths",
            "code": """









import torch
attn = {fn}(d_model=32, num_heads=2)
out = attn(torch.randn(1, 3, 32), torch.randn(1, 20, 32))
assert out.shape == (1, 3, 32), f'Shape: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "No causal mask — all KV affects all Q",
            "code": """









import torch
torch.manual_seed(0)
attn = {fn}(d_model=32, num_heads=2)
x_q = torch.randn(1, 4, 32)
x_kv = torch.randn(1, 6, 32)
out1 = attn(x_q, x_kv)
x_kv2 = x_kv.clone()
x_kv2[:, -1] = torch.randn(1, 32)
out2 = attn(x_q, x_kv2)
assert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Changing last KV should affect all Q positions'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
attn = {fn}(d_model=32, num_heads=2)
x_q = torch.randn(1, 4, 32, requires_grad=True)
x_kv = torch.randn(1, 6, 32, requires_grad=True)
attn(x_q, x_kv).sum().backward()
assert x_q.grad is not None and x_kv.grad is not None, 'Missing gradients'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Numerical correctness",
            "code": """









import torch, torch.nn as nn
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
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff {(out - expected).abs().max()}'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''





class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention implementation.
    解码器通过交叉注意力关注编码器输出。Q 来自解码器，K/V 来自编码器。
    与自注意力不同，交叉注意力不使用因果掩码（编码器输出完全可见）。
    """
    def __init__(self, d_model, num_heads):
        super().__init__()                               # 调用父类 nn.Module 初始化
        self.num_heads = num_heads                       # 注意力头数
        self.d_k = d_model // num_heads                  # 每个头的维度

        # 独立的线性投影层
        self.W_q = nn.Linear(d_model, d_model)           # Q 投影 (来自解码器)
        self.W_k = nn.Linear(d_model, d_model)           # K 投影 (来自编码器)
        self.W_v = nn.Linear(d_model, d_model)           # V 投影 (来自编码器)
        self.W_o = nn.Linear(d_model, d_model)           # 输出投影

    def forward(self, x_q, x_kv):
        B, S_q, _ = x_q.shape                            # B: batch, S_q: decoder 序列长度
        S_kv = x_kv.shape[1]                             # S_kv: encoder 序列长度 (可与 S_q 不同)

        # Q 来自解码器输入 x_q, K/V 来自编码器输入 x_kv
        q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)    # (B, H, S_q, d_k)
        k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S_kv, d_k)
        v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S_kv, d_k)

        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)         # (B, H, S_q, S_kv)

        # Softmax 归一化 (无因果掩码，编码器输出完全可见)
        weights = torch.softmax(scores, dim=-1)          # (B, H, S_q, S_kv)

        # 加权求和
        attn = torch.matmul(weights, v)                  # (B, H, S_q, d_k)

        # 合并多头并输出投影
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S_q, -1))         # (B, S_q, d_model)

    
    
    
    
    
    ''',
    "demo": '''








attn = MultiHeadCrossAttention(64, 4)
x_q = torch.randn(2, 6, 64)
x_kv = torch.randn(2, 10, 64)
print('Output:', attn(x_q, x_kv).shape)
    
    
    
    
    
    
    
    
    ''',
}
