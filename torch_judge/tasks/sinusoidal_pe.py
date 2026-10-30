"""Sinusoidal Position Encoding task."""

TASK = {
    'title': 'Sinusoidal Position Encoding',
    'title_zh': '正弦位置编码',
    'difficulty': 'Easy',
    'category': '位置编码与嵌入',
    'function_name': 'sinusoidal_pe',
    'starter_code': """def sinusoidal_pe(seq_len, d_model):
    pass""",
    'hint': """1. `freqs = 1 / 10000^(2i/d_model)` for i in `range(d_model//2)`
2. `angles = pos[:, None] * freqs[None, :]` → shape `(seq_len, d_model//2)`
3. `pe[:, 0::2] = sin(angles)`, `pe[:, 1::2] = cos(angles)`""",
    'hint_zh': """1. `freqs = 1 / 10000^(2i/d_model)`，i 取 `range(d_model//2)`
2. `angles = pos[:, None] * freqs[None, :]` → 形状 `(seq_len, d_model//2)`
3. `pe[:, 0::2] = sin(angles)`，`pe[:, 1::2] = cos(angles)`""",
    'description_en': """Implement the sinusoidal position encoding from 'Attention Is All You Need'.

Position encodings are added to token embeddings to give the model information about token positions. The original Transformer uses fixed sinusoidal functions.

**Signature:** `sinusoidal_pe(seq_len, d_model) -> Tensor`

**Parameters:**
- `seq_len` — number of positions
- `d_model` — embedding dimension (must be even)

**Returns:** position encoding tensor of shape `(seq_len, d_model)`

**Formula:**
- `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`""",
    'description_zh': """实现《Attention Is All You Need》中的正弦位置编码。

位置编码加到词嵌入上，让模型感知 token 的位置信息。原始 Transformer 使用固定的正弦函数。

**签名:** `sinusoidal_pe(seq_len, d_model) -> Tensor`

**参数:**
- `seq_len` — 位置数量
- `d_model` — 嵌入维度（必须为偶数）

**返回:** 形状为 `(seq_len, d_model)` 的位置编码张量

**公式:**
- `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
pe = {fn}(10, 16)
assert pe.shape == (10, 16), f'Expected (10, 16), got {pe.shape}'""",
        },
        {
            "name": 'Even columns are sin, odd are cos',
            "code": """import torch
pe = {fn}(5, 8)
pos = torch.arange(5).float().unsqueeze(1)
freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
angles = pos * freq
assert torch.allclose(pe[:, 0::2], torch.sin(angles), atol=1e-5), 'Even columns should be sin'
assert torch.allclose(pe[:, 1::2], torch.cos(angles), atol=1e-5), 'Odd columns should be cos'""",
        },
        {
            "name": 'Position 0 has sin=0',
            "code": """import torch
pe = {fn}(8, 32)
assert torch.allclose(pe[0, 0::2], torch.zeros(16), atol=1e-6), 'sin(0) should be 0'
assert torch.allclose(pe[0, 1::2], torch.ones(16), atol=1e-6), 'cos(0) should be 1'""",
        },
        {
            "name": 'Values in [-1, 1]',
            "code": """import torch
pe = {fn}(100, 64)
assert pe.min() >= -1.0 - 1e-5 and pe.max() <= 1.0 + 1e-5, 'PE values must be in [-1, 1]'""",
        },
    ],
    'solution': '''def sinusoidal_pe(seq_len, d_model):
    """实现正弦位置编码（Sinusoidal Position Encoding）。"""
    # Step 1: 为位置轴和偶数通道轴构造指数间隔的角频率
    pos = torch.arange(seq_len).float().unsqueeze(1)
    dim = torch.arange(0, d_model, 2).float()
    freqs = 1.0 / 10000.0 ** (dim / d_model)
    angles = pos * freqs
    # Step 2: 将每个频率的正弦写入偶数通道、余弦写入配对奇数通道
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    # Step 3: 返回可直接与长度为 seq_len 的 token embedding 相加的位置表
    return pe''',
    'demo': """torch.manual_seed(42)
pe = sinusoidal_pe(10, 16)
print("Encoding shape:", pe.shape)
print("Position 0 (first 6 dims):", pe[0, :6])
print("Value range:", pe.min().item(), pe.max().item())""",
    'explanation': r"""### 核心思路
固定位置编码为每个通道对分配不同波长：$PE_{p,2i}=\sin(p\,10000^{-2i/D})$，$PE_{p,2i+1}=\cos(p\,10000^{-2i/D})$。低索引通道变化快、高索引通道变化慢；利用三角恒等式，$PE_{p+\Delta}$ 可由 $PE_p$ 经只依赖 $\Delta$ 的线性旋转表示。

### 代码步骤拆解
- **Step 1（生成位置-频率相位表）**：`pos` 形状为 $(S,1)$，`dim` 取 $0,2,\ldots,D-2$，`freqs` 形状为 $(D/2,)$；广播相乘得到 $(S,D/2)$ 的 `angles`。
- **Step 2（交错写入正余弦）**：`pe[:, 0::2]` 接收 `sin(angles)`，与之配对的 `pe[:, 1::2]` 接收相同相位的 `cos(angles)`。
- **Step 3（形成可加位置表）**：返回 $(S,D)$ 的固定表；与形状 $(B,S,D)$ 的 token embedding 相加时，批次维自然广播。

### 重点与易错点分析
- **注意点 1（共享相位）**：奇数列不能使用新的频率索引，它必须与前一偶数列共享同一个 `angles`，才能构成完整的正余弦基。
- **注意点 2（零位置与偶数维）**：位置 0 的偶数列应全为 0、奇数列全为 1，是检查索引是否颠倒的直接方法；本题约定 `d_model` 为偶数。
- **注意点 3（固定编码特性）**：编码无可训练参数，可提前缓存并按序列长度切片；相较 learned embedding，它能计算训练长度之外的位置，但超长位置的浮点相位精度仍可能下降。

### 复杂度分析
- **时间复杂度**：$O(SD)$，需要计算 $SD/2$ 个相位的正弦和余弦。
- **空间复杂度**：$O(SD)$，输出位置表占主导，`angles` 另需 $O(SD)$ 的同阶临时空间。""",
}
