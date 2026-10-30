"""SwiGLU Activation task."""

TASK = {
    'title': 'SwiGLU Activation',
    'title_zh': 'SwiGLU 激活函数',
    'difficulty': 'Medium',
    'category': '激活函数',
    'function_name': 'swiglu',
    'starter_code': """def swiglu(x, W1, W2, Wgate):
    pass""",
    'hint': """1. `gate = (x @ Wgate) * sigmoid(x @ Wgate)`  ← swish
2. `hidden = (x @ W1) * gate`
3. `return hidden @ W2`""",
    'hint_zh': """1. `gate = (x @ Wgate) * sigmoid(x @ Wgate)`  ← swish
2. `hidden = (x @ W1) * gate`
3. `return hidden @ W2`""",
    'description_en': """Implement the SwiGLU gated activation function.

SwiGLU is used in LLaMA, PaLM, and other modern LLMs as the MLP activation. It gates one linear projection with a Swish-activated version of another.

**Signature:** `swiglu(x, W1, W2, Wgate) -> Tensor`

**Parameters:**
- `x` — input tensor `(B, d_model)`
- `W1` — weight matrix `(d_model, d_ff)`
- `W2` — output projection `(d_ff, d_model)`
- `Wgate` — gate weight `(d_model, d_ff)`

**Returns:** output tensor `(B, d_model)`

**Formula:** `hidden = (x @ W1) * swish(x @ Wgate)`, output = `hidden @ W2`

**Swish:** `swish(z) = z * sigmoid(z)`""",
    'description_zh': """实现 SwiGLU 门控激活函数。

SwiGLU 被 LLaMA、PaLM 等现代大语言模型用作 MLP 激活函数，用一个线性投影的 Swish 激活版本对另一个进行门控。

**签名:** `swiglu(x, W1, W2, Wgate) -> Tensor`

**参数:**
- `x` — 输入张量 `(B, d_model)`
- `W1` — 权重矩阵 `(d_model, d_ff)`
- `W2` — 输出投影 `(d_ff, d_model)`
- `Wgate` — 门控权重 `(d_model, d_ff)`

**返回:** 输出张量 `(B, d_model)`

**公式:** 隐藏状态 = `(x @ W1) * swish(x @ Wgate)`，输出 = `隐藏状态 @ W2`

**Swish:** `swish(z) = z * sigmoid(z)`""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, d, d_ff = 2, 16, 32
x = torch.randn(B, d)
W1 = torch.randn(d, d_ff)
W2 = torch.randn(d_ff, d)
Wg = torch.randn(d, d_ff)
out = {fn}(x, W1, W2, Wg)
assert out.shape == (B, d), f'Expected ({B}, {d}), got {out.shape}'""",
        },
        {
            "name": 'Gate is element-wise product',
            "code": """import torch
torch.manual_seed(1)
B, d, d_ff = 1, 8, 16
x = torch.randn(B, d)
W1 = torch.randn(d, d_ff)
W2 = torch.randn(d_ff, d)
Wg = torch.randn(d, d_ff)
out = {fn}(x, W1, W2, Wg)
# Manually compute expected
h1 = x @ W1
hg = x @ Wg
swish_g = hg * torch.sigmoid(hg)
expected = (h1 * swish_g) @ W2
assert torch.allclose(out, expected, atol=1e-5), f'SwiGLU output mismatch'""",
        },
        {
            "name": 'Gradient flows through gate',
            "code": """import torch
B, d, d_ff = 2, 8, 16
x = torch.randn(B, d, requires_grad=True)
W1 = torch.randn(d, d_ff, requires_grad=True)
W2 = torch.randn(d_ff, d, requires_grad=True)
Wg = torch.randn(d, d_ff, requires_grad=True)
out = {fn}(x, W1, W2, Wg)
out.sum().backward()
assert x.grad is not None, 'No gradient for x'
assert Wg.grad is not None, 'No gradient for Wgate'""",
        },
    ],
    'solution': '''def swiglu(x, W1, W2, Wgate):
    """实现SwiGLU 激活函数（SwiGLU Activation）。"""
    # Step 1: 将输入并行投影到值分支和门控分支的前馈宽度
    value = x @ W1
    gate_logits = x @ Wgate
    # Step 2: 对门控分支施加 SiLU 并逐通道调制值分支
    swish_gate = gate_logits * torch.sigmoid(gate_logits)
    hidden = value * swish_gate
    # Step 3: 将门控后的隐藏表示投影回模型维度
    return hidden @ W2''',
    'demo': """torch.manual_seed(42)
B, D, H = (2, 8, 16)
x = torch.randn(B, D)
w1 = torch.randn(D, H)
w2 = torch.randn(H, D)
w_gate = torch.randn(D, H)
out = swiglu(x, w1, w2, w_gate)
print('Output shape:', out.shape)
gate = x @ w_gate
swish_gate = gate * torch.sigmoid(gate)
print('SiLU gate sample:', swish_gate[0, :4])
print('Output norm:', out.norm(dim=-1))""",
    'explanation': r"""### 核心思路
SwiGLU 使用两条输入投影：值分支 $v=xW_1$ 与门控分支 $g=xW_g$。门控采用 $\operatorname{SiLU}(g)=g\sigma(g)$，随后逐元素组合 $h=v\odot\operatorname{SiLU}(g)$，最后 $y=hW_2$ 投影回模型宽度。相比单一路径激活，门控可按样本、按隐藏通道动态调节信息流。

### 代码步骤拆解
- **Step 1（并行上投影）**：`x` 为 $(B,D)$，`W1` 与 `Wgate` 都是 $(D,H)$，得到同形状 $(B,H)$ 的 `value` 和 `gate_logits`。
- **Step 2（SiLU 门控融合）**：计算 `gate_logits * sigmoid(gate_logits)`，再与 `value` 逐元素相乘；这里是 Hadamard 乘积而不是矩阵乘法。
- **Step 3（下投影）**：`hidden @ W2` 使用 $(H,D)$ 权重把中间表示从 $H$ 维映射回 $(B,D)$。

### 重点与易错点分析
- **注意点 1（门控非线性位置）**：SiLU 只作用于 `gate_logits`，值分支保持线性；若对两个分支都激活或先相乘再激活，函数族和梯度门控行为都会改变。
- **注意点 2（矩阵乘与逐元素乘）**：`value` 和 `swish_gate` 的形状必须完全对齐并使用 `*`；只有输入/输出投影使用 `@`，否则无法实现逐通道选择。
- **注意点 3（参数与计算权衡）**：SwiGLU 有两次上投影而普通 FFN 通常只有一次，因此常把 $H$ 选得约为传统 FFN 宽度的 $2/3$，在近似参数预算下换取门控表达能力。

### 复杂度分析
- **时间复杂度**：$O(BDH)$，两次 $(B,D)@(D,H)$ 上投影和一次 $(B,H)@(H,D)$ 下投影主导，逐元素 SiLU 为 $O(BH)$。
- **空间复杂度**：$O(BH+BD)$，训练前向需保留两个分支、门控隐藏表示及输出，其中中间宽度 $H$ 通常占主导。""",
}
