"""Attention Mask (Padding) task."""

TASK = {
    'title': 'Attention Mask (Padding)',
    'title_zh': '注意力掩码（Padding Mask）',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'attention_mask',
    'starter_code': """def attention_mask(lengths, max_seq_len):
    pass""",
    'hint': 'Create a tensor of zeros, then for each batch index i, set mask[i, lengths[i]:] = -inf.',
    'hint_zh': '创建全零张量，然后对每个 batch 索引 i，将 mask[i, lengths[i]:] 设为 -inf。',
    'description_en': """Implement a padding mask for attention. Given a batch of sequence lengths `lengths` (list of ints) and max_seq_len, return a mask tensor of shape (batch_size, max_seq_len) where valid positions are 0 and padding positions are -inf (or a large negative number). This mask is added to attention scores before softmax.""",
    'description_zh': """实现注意力机制的 padding mask。给定一批序列长度 `lengths`（整数列表）和最大序列长度 max_seq_len，返回形状为 (batch_size, max_seq_len) 的掩码张量，有效位置为 0，padding 位置为 -inf（或大负数）。该掩码在 softmax 前加到注意力分数上。""",
    'theory_en': """Padding masks prevent attention from attending to padded positions in variable-length sequences. By adding -inf to attention scores, softmax produces zero probability for padded positions.""",
    'theory_zh': 'Padding mask 防止注意力关注变长序列中的填充位置。通过在注意力分数上加 -inf，softmax 会为填充位置输出零概率。',
    "tests": [
        {
            "name": 'Variable lengths and empty sequences',
            "code": """mask = {fn}([3, 1, 0, 4], 4)
assert mask.shape == (4, 4)
expected = torch.arange(4).unsqueeze(0) < torch.tensor([3, 1, 0, 4]).unsqueeze(1)
assert torch.equal(mask[expected], torch.zeros_like(mask[expected]))
assert torch.all(mask[~expected] < -1e8)""",
        },
        {
            "name": 'Padding receives no attention',
            "code": """mask = {fn}([2, 4], 4)
scores = torch.tensor([[2., -1., 8., 9.], [1., 2., 3., 4.]])
p = torch.softmax(scores + mask, dim=-1)
assert torch.allclose(p[0, :2], torch.softmax(scores[0, :2], dim=0))
assert torch.equal(p[0, 2:], torch.zeros(2))
assert torch.allclose(p[1], torch.softmax(scores[1], dim=0))""",
        },
        {
            "name": 'Zero-width batch',
            "code": """mask = {fn}([0, 0], 0)
assert mask.shape == (2, 0)""",
        },
    ],
    'solution': '''def attention_mask(lengths, max_seq_len):
    """注意力填充掩码 (Attention Padding Mask)。"""
    # Step 1: 构造位置索引与每个样本的有效长度
    positions = torch.arange(max_seq_len).unsqueeze(0)
    valid_lengths = torch.tensor(lengths).unsqueeze(1)

    # Step 2: 标记超出有效长度的填充位置
    padding = positions >= valid_lengths

    # Step 3: 将填充位置转换为可加到 logits 的负无穷
    mask = torch.zeros(len(lengths), max_seq_len)
    return mask.masked_fill(padding, float("-inf"))''',
    'demo': """torch.manual_seed(42)
lengths = [3, 1, 4]
mask = attention_mask(lengths, 4)
print("Output shape:", mask.shape)
print("First mask:", mask[0])
print("Padding count:", torch.isinf(mask).sum().item())""",
    'explanation': r"""### 核心思路
比较位置索引与样本有效长度，生成形状 `(B, S)` 的加性掩码：有效位置为 0，填充位置为 $-\infty$。

### 代码步骤拆解
- **Step 1（索引构造）**：`positions` 为 `(1, S)`，`valid_lengths` 为 `(B, 1)`。
- **Step 2（广播比较）**：两者比较得到 `(B, S)` 的布尔 `padding`。
- **Step 3（加性掩码）**：把 `padding=True` 的位置填成负无穷。

### 重点与易错点分析
- **注意点 1（广播对齐）**：长度必须保留第二维，才能与位置轴正确广播。
- **注意点 2（掩码顺序）**：掩码应在 softmax 前加到 logits，使填充位置概率严格为零。
- **注意点 3（零长度边界）**：`max_seq_len=0` 时仍应返回 `(B, 0)` 空张量。

### 复杂度分析
- **时间复杂度**：$O(BS)$，主要是广播比较和填充。
- **空间复杂度**：$O(BS)$，保存布尔条件与输出掩码。""",
}
