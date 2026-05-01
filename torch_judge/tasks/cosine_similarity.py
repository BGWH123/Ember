"""Cosine Similarity task."""

TASK = {
    "title": "Cosine Similarity",
    "title_zh": "余弦相似度",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "description_en": "Implement cosine similarity between two vectors or matrices. cos_sim(a, b) = (a·b) / (||a|| * ||b||). For matrices, compute pairwise cosine similarity between rows. Add epsilon (1e-8) to denominators for numerical stability.",
    "description_zh": "实现两个向量或矩阵之间的余弦相似度。cos_sim(a, b) = (a·b) / (||a|| * ||b||)。对矩阵计算行之间的成对余弦相似度。分母加 epsilon（1e-8）保证数值稳定。",
    "function_name": "cosine_similarity",
    "hint": "Compute L2 norm with `torch.norm` or `(x**2).sum(-1, keepdim=True).sqrt()`. Divide dot product by product of norms.",
    "hint_zh": "使用 `torch.norm` 或 `(x**2).sum(-1, keepdim=True).sqrt()` 计算 L2 范数。将点积除以范数乘积。",
    "theory_en": "Cosine similarity measures the angle between two vectors, ignoring magnitude. It is the core of contrastive learning (CLIP, SimCLR) and semantic search.",
    "theory_zh": "余弦相似度衡量两个向量之间的夹角，忽略模长。是对比学习（CLIP、SimCLR）和语义搜索的核心。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
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

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity: cos(a,b) = (a·b) / (||a|| * ||b||)
    For matrices, compute pairwise similarity between rows.
    """
    # 计算 L2 范数（沿最后一维），keepdim 用于广播
    # Compute L2 norm along last dim, keepdim for broadcasting
    norm_a = torch.norm(a, p=2, dim=-1, keepdim=True)
    norm_b = torch.norm(b, p=2, dim=-1, keepdim=True)

    # 归一化向量（加 eps 避免除零）
    # Normalize vectors (add eps to avoid division by zero)
    a_norm = a / (norm_a + eps)
    b_norm = b / (norm_b + eps)

    # 计算余弦相似度：归一化向量的点积
    # Cosine similarity = dot product of normalized vectors
    if a.ndim == 1 and b.ndim == 1:
        return torch.dot(a_norm, b_norm)
    else:
        return torch.matmul(a_norm, b_norm.T)

    
    ''',
    "demo": "",
}
