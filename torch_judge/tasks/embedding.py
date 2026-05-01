"""Embedding Layer task."""

TASK = {
    "title": "Embedding Layer",
    "title_zh": "Embedding 层",
    "difficulty": "Easy",
    "category": "位置编码与嵌入",
    "description_en": (
        "Implement an embedding lookup layer as an nn.Module.\n\nAn embedding layer maps integer indices to dense vectors via a learnable weight matrix, used as the first layer in NLP models.\n\n**Signature:** `MyEmbedding(num_embeddings, embedding_dim)` (nn.Module)\n\n**Forward:** `forward(indices) -> Tensor`\n- `indices` — integer tensor of any shape\n\n**Returns:** embedded vectors with an extra trailing dimension of size embedding_dim\n\n**Constraints:**\n- Store weights as `nn.Parameter` of shape (num_embeddings, embedding_dim)\n- Forward is simply `weight[indices]`"
    ),
    "description_zh": (
        "实现嵌入查找层（nn.Module）。\n\n嵌入层通过可学习的权重矩阵将整数索引映射为稠密向量，是 NLP 模型的第一层。\n\n**签名:** `MyEmbedding(num_embeddings, embedding_dim)`（nn.Module）\n\n**前向传播:** `forward(indices) -> Tensor`\n- `indices` — 任意形状的整数张量\n\n**返回:** 嵌入向量，末尾增加 embedding_dim 维度\n\n**约束:**\n- 权重存储为 `nn.Parameter`，形状 (num_embeddings, embedding_dim)\n- 前向传播即 `weight[indices]`"
    ),
    "function_name": "MyEmbedding",
    "hint": (
        "`self.weight = nn.Parameter(...)` shape `(num_embeddings, embedding_dim)`\nForward: `return self.weight[indices]`"
    ),
    "hint_zh": (
        "`self.weight = nn.Parameter(...)` 形状 `(num_embeddings, embedding_dim)`\n前向：`return self.weight[indices]`"
    ),
    "tests": [
        {
            "name": "Weight shape",
            "code": """









import torch, torch.nn as nn
emb = {fn}(100, 32)
assert isinstance(emb, nn.Module), 'Must inherit from nn.Module'
assert hasattr(emb, 'weight'), 'Need self.weight'
assert emb.weight.shape == (100, 32), f'Weight shape: {emb.weight.shape}'
assert emb.weight.requires_grad, 'weight must require grad'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Lookup correctness",
            "code": """









import torch
emb = {fn}(10, 4)
idx = torch.tensor([0, 3, 7])
out = emb(idx)
assert out.shape == (3, 4), f'Output shape: {out.shape}'
assert torch.equal(out[0], emb.weight[0]), 'Mismatch at index 0'
assert torch.equal(out[1], emb.weight[3]), 'Mismatch at index 3'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Batch of indices",
            "code": """









import torch
emb = {fn}(20, 8)
idx = torch.tensor([[1, 2], [3, 4]])
out = emb(idx)
assert out.shape == (2, 2, 8), f'Batch output shape: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
emb = {fn}(10, 4)
out = emb(torch.tensor([2, 5]))
out.sum().backward()
assert emb.weight.grad is not None, 'weight.grad is None'
assert emb.weight.grad[2].abs().sum() > 0, 'Grad at used index should be non-zero'
assert emb.weight.grad[0].abs().sum() == 0, 'Grad at unused index should be zero'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








class MyEmbedding(nn.Module):  # 继承 nn.Module，注册为可训练模块
    """
    Embedding Layer module.
    """
    def __init__(self, num_embeddings, embedding_dim):  # 初始化: 定义模型结构和参数
        # Initialize layers and parameters
        super().__init__()  # 调用父类 nn.Module 初始化，注册所有子模块
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):  # 前向传播: 定义数据流
        # Compute forward pass
        return self.weight[indices]
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








emb = MyEmbedding(10, 4)
idx = torch.tensor([0, 3, 7])
print('Output shape:', emb(idx).shape)
print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))
    
    
    
    
    
    
    
    
    ''',
}
