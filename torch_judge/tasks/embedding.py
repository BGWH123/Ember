"""Embedding Layer task."""

TASK = {
    'title': 'Embedding Layer',
    'title_zh': 'Embedding 层',
    'difficulty': 'Easy',
    'category': '位置编码与嵌入',
    'function_name': 'MyEmbedding',
    'starter_code': """class MyEmbedding(nn.Module):
    pass""",
    'hint': """`self.weight = nn.Parameter(...)` shape `(num_embeddings, embedding_dim)`
Forward: `return self.weight[indices]`""",
    'hint_zh': """`self.weight = nn.Parameter(...)` 形状 `(num_embeddings, embedding_dim)`
前向：`return self.weight[indices]`""",
    'description_en': """Implement an embedding lookup layer as an nn.Module.

An embedding layer maps integer indices to dense vectors via a learnable weight matrix, used as the first layer in NLP models.

**Signature:** `MyEmbedding(num_embeddings, embedding_dim)` (nn.Module)

**Forward:** `forward(indices) -> Tensor`
- `indices` — integer tensor of any shape

**Returns:** embedded vectors with an extra trailing dimension of size embedding_dim

**Constraints:**
- Store weights as `nn.Parameter` of shape (num_embeddings, embedding_dim)
- Forward is simply `weight[indices]`""",
    'description_zh': """实现嵌入查找层（nn.Module）。

嵌入层通过可学习的权重矩阵将整数索引映射为稠密向量，是 NLP 模型的第一层。

**签名:** `MyEmbedding(num_embeddings, embedding_dim)`（nn.Module）

**前向传播:** `forward(indices) -> Tensor`
- `indices` — 任意形状的整数张量

**返回:** 嵌入向量，末尾增加 embedding_dim 维度

**约束:**
- 权重存储为 `nn.Parameter`，形状 (num_embeddings, embedding_dim)
- 前向传播即 `weight[indices]`""",
    "tests": [
        {
            "name": 'Weight shape',
            "code": """import torch, torch.nn as nn
emb = {fn}(100, 32)
assert isinstance(emb, nn.Module), 'Must inherit from nn.Module'
assert hasattr(emb, 'weight'), 'Need self.weight'
assert emb.weight.shape == (100, 32), f'Weight shape: {emb.weight.shape}'
assert emb.weight.requires_grad, 'weight must require grad'""",
        },
        {
            "name": 'Lookup correctness',
            "code": """import torch
emb = {fn}(10, 4)
idx = torch.tensor([0, 3, 7])
out = emb(idx)
assert out.shape == (3, 4), f'Output shape: {out.shape}'
assert torch.equal(out[0], emb.weight[0]), 'Mismatch at index 0'
assert torch.equal(out[1], emb.weight[3]), 'Mismatch at index 3'""",
        },
        {
            "name": 'Batch of indices',
            "code": """import torch
emb = {fn}(20, 8)
idx = torch.tensor([[1, 2], [3, 4]])
out = emb(idx)
assert out.shape == (2, 2, 8), f'Batch output shape: {out.shape}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
emb = {fn}(10, 4)
out = emb(torch.tensor([2, 5]))
out.sum().backward()
assert emb.weight.grad is not None, 'weight.grad is None'
assert emb.weight.grad[2].abs().sum() > 0, 'Grad at used index should be non-zero'
assert emb.weight.grad[0].abs().sum() == 0, 'Grad at unused index should be zero'""",
        },
    ],
    'solution': '''class MyEmbedding(nn.Module):
    """实现可学习的嵌入查找层 (Embedding Layer)。"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        # Step 1: 创建并注册词表嵌入矩阵
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        # Step 2: 按整数索引收集对应嵌入向量
        out = self.weight[indices]
        return out''',
    'demo': """torch.manual_seed(42)
layer = MyEmbedding(10, 4)
indices = torch.tensor([[0, 3, 7], [2, 5, 1]])
out = layer(indices)
print("Output shape:", out.shape)
print("First embedding:", out[0, 0])""",
    'explanation': """### 核心思路
嵌入层维护形状为 `(num_embeddings, embedding_dim)` 的可学习矩阵，并用整数 `indices` 在第 0 维直接查表，将每个索引替换为稠密向量。

### 代码步骤拆解
- **Step 1（参数注册）**：用 `nn.Parameter` 创建二维 `weight`，使其自动出现在模块参数和优化器中。
- **Step 2（索引查表）**：`self.weight[indices]` 保留索引张量的全部形状，并追加 `embedding_dim` 维。
- **Step 3（稀疏梯度位置）**：反向传播只为本批次实际访问的权重行累积非零梯度。

### 重点与易错点分析
- **注意点 1（索引类型）**：`indices` 必须是整数类型且落在词表范围内，浮点索引不能直接查表。
- **注意点 2（参数而非普通张量）**：若不用 `nn.Parameter`，权重不会被优化器发现，也不会正常进入模块状态。
- **注意点 3（输出形状）**：任意形状 `(...,)` 的索引都会映射为 `(..., embedding_dim)`，不能假设输入仅是一维。

### 复杂度分析
- **时间复杂度**：$O(ND)$，其中 $N$ 为索引元素数、$D$ 为嵌入维度。
- **空间复杂度**：$O(VD+ND)$，分别对应参数矩阵与查表输出。""",
}
