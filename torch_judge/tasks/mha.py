"""Multi-Head Attention task."""

TASK = {
    "title": "Multi-Head Attention",
    "title_zh": "多头注意力",
    "difficulty": "Hard",
    "category": "注意力机制",
    "description_en": (
        "Implement Multi-Head Attention from scratch.\n\nMHA projects inputs into multiple heads, computes scaled dot-product attention per head, then concatenates and projects the results.\n\n**Signature:** `MultiHeadAttention(d_model, num_heads)`\n\n**Method:** `forward(Q, K, V) -> Tensor`\n- `Q` — query tensor (B, S_q, d_model)\n- `K` — key tensor (B, S_k, d_model)\n- `V` — value tensor (B, S_k, d_model)\n\n**Returns:** attention output (B, S_q, d_model)\n\n**Constraints:**\n- Use W_q, W_k, W_v, W_o as `nn.Linear(d_model, d_model)`\n- `d_k = d_model // num_heads`\n- Support cross-attention (S_q != S_k)"
    ),
    "description_zh": (
        "从零实现多头注意力。\n\nMHA 将输入投影到多个头，每个头计算缩放点积注意力，然后拼接并投影结果。\n\n**签名:** `MultiHeadAttention(d_model, num_heads)`\n\n**方法:** `forward(Q, K, V) -> Tensor`\n- `Q` — 查询张量 (B, S_q, d_model)\n- `K` — 键张量 (B, S_k, d_model)\n- `V` — 值张量 (B, S_k, d_model)\n\n**返回:** 注意力输出 (B, S_q, d_model)\n\n**约束:**\n- 使用 W_q、W_k、W_v、W_o 作为 `nn.Linear(d_model, d_model)`\n- `d_k = d_model // num_heads`\n- 支持交叉注意力（S_q != S_k）"
    ),
    "function_name": "MultiHeadAttention",
    "hint": (
        "1. Project Q/K/V with `nn.Linear(d_model, d_model)`\n2. Reshape to `(B, heads, S, d_k)` via `.view(...).transpose(1,2)`\n3. `scores = Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`\n4. Transpose + reshape → `W_o` projection"
    ),
    "hint_zh": (
        "1. 用 `nn.Linear(d_model, d_model)` 投影 Q/K/V\n2. `.view(...).transpose(1,2)` → `(B, heads, S, d_k)`\n3. `scores = Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`\n4. transpose + reshape → `W_o` 投影"
    ),
    "theory_en": (
        "Multi-Head Attention splits the input into multiple heads, allowing the model to jointly attend to information from different representation subspaces.\n\n**Mathematical Formulation:**\nFor each head $h$:\n$$\\text{head}_h = \\text{Attention}(QW_q^h, KW_k^h, VW_v^h) = \\text{softmax}\\left(\\frac{QW_q^h (KW_k^h)^T}{\\sqrt{d_k}}\\right) VW_v^h$$\n$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_H) W_o$$\n\n**Key Insight:**\n- $d_k = d_{model} / H$ ensures total computation stays constant\n- Each head can learn different attention patterns (syntax, semantics, long-range dependencies)\n- The scaling factor $1/\\sqrt{d_k}$ prevents softmax saturation for large $d_k$"
    ),
    "theory_zh": (
        "多头注意力将输入拆分为多个头，使模型能够联合关注来自不同表示子空间的信息。\n\n**数学公式：**\n对于每个头 $h$：\n$$\\text{head}_h = \\text{Attention}(QW_q^h, KW_k^h, VW_v^h) = \\text{softmax}\\left(\\frac{QW_q^h (KW_k^h)^T}{\\sqrt{d_k}}\\right) VW_v^h$$\n$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_H) W_o$$\n\n**核心要点：**\n- $d_k = d_{model} / H$ 保证总计算量不变\n- 每个头可以学习不同的注意力模式（语法、语义、长距离依赖）\n- 缩放因子 $1/\\sqrt{d_k}$ 防止大 $d_k$ 时 softmax 饱和"
    ),
    "diagram_en": (
        "```mermaid\nflowchart LR\n    Q[Q] -->|W_q| QH[Q heads]\n    K[K] -->|W_k| KH[K heads]\n    V[V] -->|W_v| VH[V heads]\n    QH -->|scaled dot-product| ATTN[Attention weights]\n    KH --> ATTN\n    ATTN -->|@ V| OUT[Head outputs]\n    VH --> OUT\n    OUT -->|concat| CAT[Concatenated]\n    CAT -->|W_o| FINAL[Output]\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart LR\n    Q[Q] -->|W_q| QH[Q 多头]\n    K[K] -->|W_k| KH[K 多头]\n    V[V] -->|W_v| VH[V 多头]\n    QH -->|缩放点积| ATTN[注意力权重]\n    KH --> ATTN\n    ATTN -->|@ V| OUT[头输出]\n    VH --> OUT\n    OUT -->|拼接| CAT[拼接结果]\n    CAT -->|W_o| FINAL[输出]\n```"
    ),
    "tests": [
        {
            "name": "Is nn.Module",
            "code": """









import torch, torch.nn as nn
mha = {fn}(d_model=16, num_heads=2)
assert isinstance(mha, nn.Module), 'MultiHeadAttention should inherit from nn.Module'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Output shape",
            "code": """









import torch
torch.manual_seed(0)
B, S, D, H = 2, 6, 32, 4
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(B, S, D)
out = mha.forward(x, x, x)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Uses nn.Linear with correct shapes",
            "code": """









import torch, torch.nn as nn
mha = {fn}(d_model=32, num_heads=4)
for name in ['W_q', 'W_k', 'W_v', 'W_o']:
    layer = (((((((getattr(mha, name) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower()))
    assert isinstance(layer, nn.Linear), f'{name} should be nn.Linear, got {type(layer)}'
    assert layer.weight.shape == (32, 32), f'{name}.weight shape: {layer.weight.shape}'
    assert layer.weight.requires_grad, f'{name}.weight must require grad'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Numerical correctness vs reference",
            "code": """









import torch, torch.nn as nn, math
torch.manual_seed(0)
D, H = 16, 2
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
Q = torch.randn(1, 4, D)
K = torch.randn(1, 4, D)
V = torch.randn(1, 4, D)
out = mha.forward(Q, K, V)
q = (((((((mha.W_q if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q)(Q).view(1, 4, H, d_k).transpose(1, 2)
k = (((((((mha.W_k if hasattr(mha, 'W_k') else mha.w_k) if hasattr(mha, 'W_k') else mha.w_k) if hasattr(mha, 'W_k') else mha.w_k) if hasattr(mha, 'W_k') else mha.w_k) if hasattr(mha, 'W_k') else mha.w_k) if hasattr(mha, 'W_k') else mha.w_k) if hasattr(mha, 'W_k') else mha.w_k)(K).view(1, 4, H, d_k).transpose(1, 2)
v = (((((((mha.W_v if hasattr(mha, 'W_v') else mha.w_v) if hasattr(mha, 'W_v') else mha.w_v) if hasattr(mha, 'W_v') else mha.w_v) if hasattr(mha, 'W_v') else mha.w_v) if hasattr(mha, 'W_v') else mha.w_v) if hasattr(mha, 'W_v') else mha.w_v) if hasattr(mha, 'W_v') else mha.w_v)(V).view(1, 4, H, d_k).transpose(1, 2)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
ref = (((((((mha.W_o if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o)(attn.transpose(1, 2).contiguous().view(1, 4, D))
assert torch.allclose(out, ref, atol=1e-5), 'Output does not match reference'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
torch.manual_seed(0)
mha = {fn}(d_model=16, num_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
out = mha.forward(x, x, x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert (((((((mha.W_q if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q).weight.grad is not None, 'W_q.weight.grad is None'
assert (((((((mha.W_o if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o).weight.grad is not None, 'W_o.weight.grad is None'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Cross-attention (seq_q != seq_k)",
            "code": """









import torch
mha = {fn}(d_model=32, num_heads=4)
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out = mha.forward(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Different heads give different outputs",
            "code": """









import torch
torch.manual_seed(42)
D, H = 16, 4
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(1, 4, D)
q = (((((((mha.W_q if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q)(x).view(1, 4, H, d_k).transpose(1, 2)
assert not torch.allclose(q[:, 0], q[:, 1], atol=1e-3), 'Heads produce identical queries'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''





class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) implementation.
    将输入投影到多个注意力头，分别计算缩放点积注意力，最后拼接并输出投影。
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()                               # 调用父类 nn.Module 的初始化，注册所有子模块
        self.num_heads = num_heads                       # 注意力头数 H
        self.d_k = d_model // num_heads                  # 每个头的维度 d_k = d_model / H
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 四个线性投影层: W_q, W_k, W_v 分别投影 Q/K/V, W_o 合并多头输出
        self.W_q = nn.Linear(d_model, d_model)           # Q 投影: (B, S, d_model) -> (B, S, d_model)
        self.W_k = nn.Linear(d_model, d_model)           # K 投影: (B, S, d_model) -> (B, S, d_model)
        self.W_v = nn.Linear(d_model, d_model)           # V 投影: (B, S, d_model) -> (B, S, d_model)
        self.W_o = nn.Linear(d_model, d_model)           # 输出投影: 拼接后的 (B, S, d_model) -> (B, S, d_model)

    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape                              # B: batch size, S_q: query 序列长度
        S_k = K.shape[1]                                 # S_k: key/value 序列长度 (支持 cross-attention)

        # Step 1: 线性投影 + 多头拆分
        # view(B, S, H, d_k) 将最后一维拆分为 H 个头，每个头 d_k 维
        # transpose(1, 2) 交换维度得到 (B, H, S, d_k)，便于并行做矩阵乘法
        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)   # (B, H, S_q, d_k)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)   # (B, H, S_k, d_k)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)   # (B, H, S_k, d_k)

        # Step 2: 计算缩放点积注意力分数
        # scores[b,h,i,j] = q[b,h,i] · k[b,h,j] / sqrt(d_k)
        # 除以 sqrt(d_k) 防止大 d_k 时 softmax 饱和（梯度消失）
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)      # (B, H, S_q, S_k)

        # Step 3: Softmax 归一化得到注意力权重
        # dim=-1 表示沿 key 维度归一化，每行和为 1
        weights = torch.softmax(scores, dim=-1)          # (B, H, S_q, S_k)

        # Step 4: 加权求和: 注意力权重 @ V 得到上下文向量
        attn = torch.matmul(weights, v)                  # (B, H, S_q, d_k)

        # Step 5: 合并多头 + 输出投影
        # transpose(1,2): (B, H, S_q, d_k) -> (B, S_q, H, d_k)
        # contiguous(): 保证内存连续，因为 transpose 改变了 strides
        # view(B, S_q, -1): 将 H * d_k = d_model 合并回最后一维
        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)                  # (B, S_q, d_model)
        return self.W_o(out)                             # 最终线性投影: (B, S_q, d_model) -> (B, S_q, d_model)

    
    
    
    
    
    ''',
    "demo": '''








torch.manual_seed(0)
mha = MultiHeadAttention(d_model=32, num_heads=4)
x = torch.randn(2, 6, 32)
out = mha.forward(x, x, x)
print("Self-attn shape:", out.shape)

Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out2 = mha.forward(Q, K, V)
print("Cross-attn shape:", out2.shape)
    
    
    
    
    
    
    
    
    ''',
}
