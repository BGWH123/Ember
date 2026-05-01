"""Attention Mask (Padding) task."""

TASK = {
    "title": "Attention Mask (Padding)",
    "title_zh": "注意力掩码（Padding Mask）",
    "difficulty": "Medium",
    "category": "注意力机制",
    "description_en": "Implement a padding mask for attention. Given a batch of sequence lengths `lengths` (list of ints) and max_seq_len, return a mask tensor of shape (batch_size, max_seq_len) where valid positions are 0 and padding positions are -inf (or a large negative number). This mask is added to attention scores before softmax.",
    "description_zh": "实现注意力机制的 padding mask。给定一批序列长度 `lengths`（整数列表）和最大序列长度 max_seq_len，返回形状为 (batch_size, max_seq_len) 的掩码张量，有效位置为 0，padding 位置为 -inf（或大负数）。该掩码在 softmax 前加到注意力分数上。",
    "function_name": "attention_mask",
    "hint": "Create a tensor of zeros, then for each batch index i, set mask[i, lengths[i]:] = -inf.",
    "hint_zh": "创建全零张量，然后对每个 batch 索引 i，将 mask[i, lengths[i]:] 设为 -inf。",
    "theory_en": "Padding masks prevent attention from attending to padded positions in variable-length sequences. By adding -inf to attention scores, softmax produces zero probability for padded positions.",
    "theory_zh": "Padding mask 防止注意力关注变长序列中的填充位置。通过在注意力分数上加 -inf，softmax 会为填充位置输出零概率。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
        {
            "name": "basic",
            "code": "",
        },
    ],
    "solution": '''

import torch

def attention_mask(lengths: list[int], max_seq_len: int) -> torch.Tensor:
    """
    Create padding mask for attention.
    Valid positions = 0, padding positions = -inf.
    Shape: (batch_size, max_seq_len)
    """
    batch_size = len(lengths)
    # 创建全零掩码
    # Create zero mask
    mask = torch.zeros(batch_size, max_seq_len)

    # 对每个序列，将超出实际长度的位置设为 -inf
    # For each sequence, set positions beyond actual length to -inf
    for i, length in enumerate(lengths):
        if length < max_seq_len:
            mask[i, length:] = float('-inf')

    return mask

    
    ''',
    "demo": "",
}
