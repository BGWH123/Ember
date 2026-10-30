"""Multi-Head Attention task."""

TASK = {
    'title': 'Multi-Head Attention',
    'title_zh': '多头注意力',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'MultiHeadAttention',
    'starter_code': """class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        pass

    def forward(self, Q, K, V):
        pass""",
    'hint': """1. Project Q/K/V with `nn.Linear(d_model, d_model)`
2. Reshape to `(B, heads, S, d_k)` via `.view(...).transpose(1,2)`
3. `scores = Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`
4. Transpose + reshape → `W_o` projection""",
    'hint_zh': """1. 用 `nn.Linear(d_model, d_model)` 投影 Q/K/V
2. `.view(...).transpose(1,2)` → `(B, heads, S, d_k)`
3. `scores = Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`
4. transpose + reshape → `W_o` 投影""",
    'description_en': """Implement Multi-Head Attention from scratch.

MHA projects inputs into multiple heads, computes scaled dot-product attention per head, then concatenates and projects the results.

**Signature:** `MultiHeadAttention(d_model, num_heads)`

**Method:** `forward(Q, K, V) -> Tensor`
- `Q` — query tensor (B, S_q, d_model)
- `K` — key tensor (B, S_k, d_model)
- `V` — value tensor (B, S_k, d_model)

**Returns:** attention output (B, S_q, d_model)

**Constraints:**
- Use W_q, W_k, W_v, W_o as `nn.Linear(d_model, d_model)`
- `d_k = d_model // num_heads`
- Support cross-attention (S_q != S_k)""",
    'description_zh': """从零实现多头注意力。

MHA 将输入投影到多个头，每个头计算缩放点积注意力，然后拼接并投影结果。

**签名:** `MultiHeadAttention(d_model, num_heads)`

**方法:** `forward(Q, K, V) -> Tensor`
- `Q` — 查询张量 (B, S_q, d_model)
- `K` — 键张量 (B, S_k, d_model)
- `V` — 值张量 (B, S_k, d_model)

**返回:** 注意力输出 (B, S_q, d_model)

**约束:**
- 使用 W_q、W_k、W_v、W_o 作为 `nn.Linear(d_model, d_model)`
- `d_k = d_model // num_heads`
- 支持交叉注意力（S_q != S_k）""",
    'theory_en': r"""Multi-Head Attention splits the input into multiple heads, allowing the model to jointly attend to information from different representation subspaces.

**Mathematical Formulation:**
For each head $h$:
$$\text{head}_h = \text{Attention}(QW_q^h, KW_k^h, VW_v^h) = \text{softmax}\left(\frac{QW_q^h (KW_k^h)^T}{\sqrt{d_k}}\right) VW_v^h$$
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) W_o$$

**Key Insight:**
- $d_k = d_{model} / H$ ensures total computation stays constant
- Each head can learn different attention patterns (syntax, semantics, long-range dependencies)
- The scaling factor $1/\sqrt{d_k}$ prevents softmax saturation for large $d_k$""",
    'theory_zh': r"""多头注意力将输入拆分为多个头，使模型能够联合关注来自不同表示子空间的信息。

**数学公式：**
对于每个头 $h$：
$$\text{head}_h = \text{Attention}(QW_q^h, KW_k^h, VW_v^h) = \text{softmax}\left(\frac{QW_q^h (KW_k^h)^T}{\sqrt{d_k}}\right) VW_v^h$$
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) W_o$$

**核心要点：**
- $d_k = d_{model} / H$ 保证总计算量不变
- 每个头可以学习不同的注意力模式（语法、语义、长距离依赖）
- 缩放因子 $1/\sqrt{d_k}$ 防止大 $d_k$ 时 softmax 饱和""",
    'diagram_en': """```mermaid
flowchart LR
    Q[Q] -->|W_q| QH[Q heads]
    K[K] -->|W_k| KH[K heads]
    V[V] -->|W_v| VH[V heads]
    QH -->|scaled dot-product| ATTN[Attention weights]
    KH --> ATTN
    ATTN -->|@ V| OUT[Head outputs]
    VH --> OUT
    OUT -->|concat| CAT[Concatenated]
    CAT -->|W_o| FINAL[Output]
```""",
    'diagram_zh': """```mermaid
flowchart LR
    Q[Q] -->|W_q| QH[Q 多头]
    K[K] -->|W_k| KH[K 多头]
    V[V] -->|W_v| VH[V 多头]
    QH -->|缩放点积| ATTN[注意力权重]
    KH --> ATTN
    ATTN -->|@ V| OUT[头输出]
    VH --> OUT
    OUT -->|拼接| CAT[拼接结果]
    CAT -->|W_o| FINAL[输出]
```""",
    "tests": [
        {
            "name": 'Is nn.Module',
            "code": """import torch, torch.nn as nn
mha = {fn}(d_model=16, num_heads=2)
assert isinstance(mha, nn.Module), 'MultiHeadAttention should inherit from nn.Module'""",
        },
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, S, D, H = 2, 6, 32, 4
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(B, S, D)
out = mha.forward(x, x, x)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'""",
        },
        {
            "name": 'Uses nn.Linear with correct shapes',
            "code": """import torch, torch.nn as nn
mha = {fn}(d_model=32, num_heads=4)
for name in ['W_q', 'W_k', 'W_v', 'W_o']:
    layer = (((((((getattr(mha, name) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower())) if hasattr(mha, name) else getattr(mha, name.lower()))
    assert isinstance(layer, nn.Linear), f'{name} should be nn.Linear, got {type(layer)}'
    assert layer.weight.shape == (32, 32), f'{name}.weight shape: {layer.weight.shape}'
    assert layer.weight.requires_grad, f'{name}.weight must require grad'""",
        },
        {
            "name": 'Numerical correctness vs reference',
            "code": """import torch, torch.nn as nn, math
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
assert torch.allclose(out, ref, atol=1e-5), 'Output does not match reference'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(0)
mha = {fn}(d_model=16, num_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
out = mha.forward(x, x, x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert (((((((mha.W_q if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q).weight.grad is not None, 'W_q.weight.grad is None'
assert (((((((mha.W_o if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o) if hasattr(mha, 'W_o') else mha.w_o).weight.grad is not None, 'W_o.weight.grad is None'""",
        },
        {
            "name": 'Cross-attention (seq_q != seq_k)',
            "code": """import torch
mha = {fn}(d_model=32, num_heads=4)
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out = mha.forward(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'""",
        },
        {
            "name": 'Different heads give different outputs',
            "code": """import torch
torch.manual_seed(42)
D, H = 16, 4
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(1, 4, D)
q = (((((((mha.W_q if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q) if hasattr(mha, 'W_q') else mha.w_q)(x).view(1, 4, H, d_k).transpose(1, 2)
assert not torch.allclose(q[:, 0], q[:, 1], atol=1e-3), 'Heads produce identical queries'""",
        },
    ],
    'solution': '''class MultiHeadAttention(nn.Module):
    """实现多头缩放点积注意力 (Multi-Head Attention)。"""

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

    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]

        # Step 1: 投影 Q、K、V 并拆分为多个头
        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

        # Step 2: 计算缩放点积分数并沿键序列归一化
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)

        # Step 3: 聚合值、合并多头并完成输出投影
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(B, S_q, -1)
        out = self.W_o(context)
        return out''',
    'demo': """torch.manual_seed(42)
attention = MultiHeadAttention(d_model=32, num_heads=4)
Q = torch.randn(2, 3, 32)
K = torch.randn(2, 7, 32)
V = torch.randn(2, 7, 32)
out = attention(Q, K, V)
print("Output shape:", out.shape)
print("Head dimension:", attention.d_k)
print("Sample values:", out[0, 0, :4])""",
    'explanation': r"""### 核心思路
多头注意力将模型维拆为 `num_heads` 个 $d_k$ 子空间，每个头计算 $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$，再拼接并输出投影。支持 `S_q != S_k` 的交叉注意力。

### 代码步骤拆解
- **Step 1（投影拆头）**：把 Q/K/V 变换为 `(B, H, S, d_k)`。
- **Step 2（权重计算）**：产生 `(B, H, S_q, S_k)` 分数并沿 `S_k` softmax。
- **Step 3（合并输出）**：上下文转回 `(B, S_q, d_model)` 后经 `W_o`。

### 重点与易错点分析
- **注意点 1（可整除性）**：`d_model` 必须能被 `num_heads` 整除，否则 view 会破坏维度。
- **注意点 2（缩放作用）**：除以 $\sqrt{d_k}$ 可防止点积方差过大导致 softmax 饱和。
- **注意点 3（连续内存）**：transpose 后需 `.contiguous()` 再 view，避免 stride 不兼容。

### 复杂度分析
- **时间复杂度**：$O(BHS_qS_kd_k)$，注意力点积与值聚合占主导。
- **空间复杂度**：$O(BHS_qS_k)$，注意力权重是主要中间激活。""",
}
