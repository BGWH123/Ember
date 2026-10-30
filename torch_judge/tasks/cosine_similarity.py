"""Cosine Similarity task."""

TASK = {
    'title': 'Cosine Similarity',
    'title_zh': '余弦相似度',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'cosine_similarity',
    'starter_code': """def cosine_similarity(a, b, eps=1e-8):
    pass""",
    'hint': """Compute L2 norm with `torch.norm` or `(x**2).sum(-1, keepdim=True).sqrt()`. Divide dot product by product of norms.""",
    'hint_zh': '使用 `torch.norm` 或 `(x**2).sum(-1, keepdim=True).sqrt()` 计算 L2 范数。将点积除以范数乘积。',
    'description_en': """Implement cosine similarity between two vectors or matrices. cos_sim(a, b) = (a·b) / (||a|| * ||b||). For matrices, compute pairwise cosine similarity between rows. Add epsilon (1e-8) to denominators for numerical stability.""",
    'description_zh': '实现两个向量或矩阵之间的余弦相似度。cos_sim(a, b) = (a·b) / (||a|| * ||b||)。对矩阵计算行之间的成对余弦相似度。分母加 epsilon（1e-8）保证数值稳定。',
    'theory_en': """Cosine similarity measures the angle between two vectors, ignoring magnitude. It is the core of contrastive learning (CLIP, SimCLR) and semantic search.""",
    'theory_zh': '余弦相似度衡量两个向量之间的夹角，忽略模长。是对比学习（CLIP、SimCLR）和语义搜索的核心。',
    "tests": [
        {
            "name": 'Vector directions',
            "code": """a = torch.tensor([3., 4., 0.])
assert torch.allclose({fn}(a, a), torch.tensor(1.), atol=1e-6)
assert torch.allclose({fn}(a, -a), torch.tensor(-1.), atol=1e-6)
assert torch.allclose({fn}(torch.tensor([1., 0.]), torch.tensor([0., 2.])), torch.tensor(0.), atol=1e-6)""",
        },
        {
            "name": 'All row pairs and zero vector',
            "code": """a = torch.tensor([[1., 0., 0.], [1., 2., 3.]])
b = torch.tensor([[0., 1., 0.], [3., 2., 1.], [0., 0., 0.], [-1., 0., 0.]])
expected = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
out = {fn}(a, b)
assert out.shape == (2, 4)
assert torch.isfinite(out).all()
assert torch.allclose(out, expected, atol=1e-6)""",
        },
        {
            "name": 'Custom epsilon',
            "code": """a = torch.tensor([1e-10, 0.], dtype=torch.float64)
b = torch.tensor([1., 1.], dtype=torch.float64)
eps = 1e-3
expected = torch.dot(a / (a.norm() + eps), b / (b.norm() + eps))
assert torch.allclose({fn}(a, b, eps), expected, atol=1e-12)""",
        },
    ],
    'solution': '''def cosine_similarity(a, b, eps=1e-8):
    """向量与矩阵余弦相似度 (Cosine Similarity)。"""
    # Step 1: 沿特征维计算并稳定化 L2 范数
    norm_a = torch.norm(a, p=2, dim=-1, keepdim=True)
    norm_b = torch.norm(b, p=2, dim=-1, keepdim=True)

    # Step 2: 将输入归一化到单位方向
    a_normalized = a / (norm_a + eps)
    b_normalized = b / (norm_b + eps)

    # Step 3: 向量计算点积，矩阵计算全部行对相似度
    if a.ndim == 1 and b.ndim == 1:
        return torch.dot(a_normalized, b_normalized)
    return a_normalized @ b_normalized.T''',
    'demo': """torch.manual_seed(42)
a = torch.randn(3, 8)
b = torch.randn(4, 8)
similarities = cosine_similarity(a, b)
print("Output shape:", similarities.shape)
print("First row:", similarities[0])
print("Value range:", similarities.min().item(), similarities.max().item())""",
    'explanation': r"""### 核心思路
余弦相似度为 $\frac{a\cdot b}{(\|a\|_2+\epsilon)(\|b\|_2+\epsilon)}$；矩阵输入时计算所有行之间的成对相似度。

### 代码步骤拆解
- **Step 1（范数计算）**：沿最后一维求 L2 范数，`keepdim=True` 保留广播轴。
- **Step 2（方向归一化）**：分别除以加 `eps` 的范数，零向量得到有限值。
- **Step 3（点积计算）**：一维输入用 `torch.dot`，二维输入用矩阵乘法得到 `(N, M)`。

### 重点与易错点分析
- **注意点 1（eps 语义）**：本题将 `eps` 加在范数上，而非对范数做下界截断，两者对微小向量结果不同。
- **注意点 2（成对矩阵）**：矩阵模式需要转置 `b_normalized`，否则只会尝试逐位置相乘。
- **注意点 3（零向量）**：加入 `eps` 后零向量相似度为 0 且不会产生 NaN。

### 复杂度分析
- **时间复杂度**：向量为 $O(D)$，矩阵成对计算为 $O(NMD)$。
- **空间复杂度**：矩阵模式为 $O(NM+ND+MD)$，输出相似度矩阵占主导。""",
}
