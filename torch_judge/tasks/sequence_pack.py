"""Sequence Packing task."""

TASK = {
    "title": "Sequence Packing",
    "title_zh": "序列打包",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "description_en": "Implement sequence packing for efficient training. Given a list of variable-length sequences (each is a 1D tensor), pack them into a single padded tensor of shape (batch_size, max_len). Also return a binary mask indicating valid positions (1 for real data, 0 for padding).",
    "description_zh": "实现序列打包以提高训练效率。给定一批变长序列（每个是一维张量），将它们打包成单个填充张量，形状为 (batch_size, max_len)。同时返回一个二元掩码，标记有效位置（1 表示真实数据，0 表示填充）。",
    "function_name": "sequence_pack",
    "hint": "Find max length, create a zero tensor of shape (batch_size, max_len), then copy each sequence into its corresponding row.",
    "hint_zh": "找到最大长度，创建形状为 (batch_size, max_len) 的零张量，然后将每个序列复制到对应行。",
    "theory_en": "Sequence packing is essential for batched training of variable-length sequences (NLP, time series). Padding allows tensor operations while masks prevent attention to padded positions.",
    "theory_zh": "序列打包是变长序列（NLP、时间序列）批量训练的基础。填充使张量运算成为可能，掩码防止注意力关注填充位置。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
    ],
    "solution": '''

import torch
from torch.nn.utils.rnn import pad_sequence

def sequence_pack(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack variable-length sequences into a padded tensor.
    Returns: (padded_tensor, mask)
    padded_tensor: (batch_size, max_len)
    mask: (batch_size, max_len), 1 for valid, 0 for padding.
    """
    # 找到最大序列长度
    # Find maximum sequence length
    max_len = max(seq.shape[0] for seq in sequences)
    batch_size = len(sequences)

    # 创建填充张量和掩码
    # Create padded tensor and mask
    padded = torch.zeros(batch_size, max_len, dtype=sequences[0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    # 将每个序列复制到对应行，并标记有效位置
    # Copy each sequence to its row and mark valid positions
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length] = seq
        mask[i, :length] = 1.0

    return padded, mask

    
    ''',
    "demo": "",
}
