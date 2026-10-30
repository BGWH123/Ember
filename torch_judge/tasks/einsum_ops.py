"""PyTorch Einsum Operations task."""

TASK = {
    'title': 'PyTorch Einsum Operations',
    'title_zh': 'PyTorch Einsum 操作',
    'difficulty': 'Medium',
    'category': '基础网络组件',
    'function_name': 'einsum_operations',
    'starter_code': """def einsum_operations(A, B, C, operation):
    pass""",
    'hint': """- batch_matmul: einsum('bmk,bkn->bmn', A, B)
- bilinear: einsum('bm,mn,bn->b', A, B, C)
- trace: einsum('bii->b', A)
- outer: einsum('m,n->mn', A, B)
- diagonal: einsum('ii->i', A)""",
    'hint_zh': """- batch_matmul: einsum('bmk,bkn->bmn', A, B)
- bilinear: einsum('bm,mn,bn->b', A, B, C)
- trace: einsum('bii->b', A)
- outer: einsum('m,n->mn', A, B)
- diagonal: einsum('ii->i', A)""",
    'description_en': """Implement common deep learning operations using torch.einsum.

Einsum is a powerful unified notation for tensor operations. Mastering it is essential for reading research code and implementing custom operations efficiently.

Signature: einsum_operations(A, B, C, operation) -> Tensor

Parameters:
- A, B, C -- input tensors
- operation -- string, one of: 'batch_matmul', 'bilinear', 'trace', 'outer', 'diagonal'

Operations:
- 'batch_matmul': batch matrix multiplication (B, M, K) @ (B, K, N) -> (B, M, N)
- 'bilinear': bilinear form (B, M) @ (M, N) @ (B, N) -> (B,)
- 'trace': trace of each matrix in batch (B, N, N) -> (B,)
- 'outer': outer product of two vectors (M,) @ (N,) -> (M, N)
- 'diagonal': extract diagonal (N, N) -> (N,)

Constraints:
- Must use torch.einsum for all operations
- No torch.matmul, torch.bmm, or other built-in ops""",
    'description_zh': """使用 torch.einsum 实现常见的深度学习操作。

Einsum 是一种强大的张量操作统一表示法。掌握它对阅读研究代码和高效实现自定义操作至关重要。

签名: einsum_operations(A, B, C, operation) -> Tensor

参数:
- A, B, C -- 任意形状的输入张量
- operation -- 字符串，取以下之一: 'batch_matmul', 'bilinear', 'trace', 'outer', 'diagonal'

操作:
- 'batch_matmul': 批次矩阵乘法 (B, M, K) @ (B, K, N) -> (B, M, N)
- 'bilinear': 双线性形式 (B, M) @ (M, N) @ (B, N) -> (B,)
- 'trace': 批次中每个矩阵的迹 (B, N, N) -> (B,)
- 'outer': 两个向量的外积 (M,) @ (N,) -> (M, N)
- 'diagonal': 提取对角线 (N, N) -> (N,)

约束:
- 所有操作必须使用 torch.einsum
- 禁止使用 torch.matmul、torch.bmm 或其他内置运算""",
    'theory_en': """Einsum (Einstein summation) provides a unified notation for tensor contractions.

Basic Rules:
- Repeated indices in input are summed over (contraction)
- Output indices specify which dimensions remain
- Batch dimensions appear in all inputs and the output

Examples:
- Matrix multiplication: ik,kj->ij
- Batch matrix multiplication: bik,bkj->bij
- Trace: ii->
- Diagonal extraction: ii->i
- Outer product: i,j->ij
- Batch dot product: bi,bi->b

Advantages over explicit ops:
- Single function covers matmul, bmm, trace, diagonal, outer, dot, transpose
- Often fused by PyTorch for better performance
- More readable for complex tensor contractions""",
    'theory_zh': """Einsum（爱因斯坦求和约定）为张量缩并提供统一表示法。

基本规则:
- 输入中重复的索引被求和（收缩）
- 输出索引指定保留哪些维度
- 批次维度出现在所有输入和输出中

示例:
- 矩阵乘法: ik,kj->ij
- 批次矩阵乘法: bik,bkj->bij
- 迹: ii->
- 提取对角线: ii->i
- 外积: i,j->ij
- 批次点积: bi,bi->b

相比显式操作的优势:
- 单个函数覆盖 matmul、bmm、trace、diagonal、outer、dot、transpose
- PyTorch 经常将其融合以获得更好性能
- 复杂张量收缩更易读""",
    'diagram_en': """flowchart LR
    subgraph batch_matmul
        A1[A: b,m,k] --> E1[einsum bmk,bkn-bmn]
        B1[B: b,k,n] --> E1
        E1 --> O1[Out: b,m,n]
    end
    subgraph trace
        A2[A: b,i,i] --> E2[einsum bii-b]
        E2 --> O2[Out: b]
    end""",
    'diagram_zh': """flowchart LR
    subgraph batch_matmul
        A1[A: b,m,k] --> E1[einsum bmk,bkn-bmn]
        B1[B: b,k,n] --> E1
        E1 --> O1[Out: b,m,n]
    end
    subgraph trace
        A2[A: b,i,i] --> E2[einsum bii-b]
        E2 --> O2[Out: b]
    end""",
    "tests": [
        {
            "name": 'Batch matrix multiplication',
            "code": """import torch
torch.manual_seed(0)
A = torch.randn(2, 3, 4)
B = torch.randn(2, 4, 5)
out = {fn}(A, B, None, 'batch_matmul')
ref = torch.bmm(A, B)
assert out.shape == (2, 3, 5), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Does not match torch.bmm'""",
        },
        {
            "name": 'Bilinear form',
            "code": """import torch
torch.manual_seed(0)
A = torch.randn(3, 4)
M = torch.randn(4, 5)
C = torch.randn(3, 5)
out = {fn}(A, M, C, 'bilinear')
ref = torch.einsum('bm,mn,bn->b', A, M, C)
assert out.shape == (3,), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Bilinear mismatch'""",
        },
        {
            "name": 'Trace',
            "code": """import torch
torch.manual_seed(0)
A = torch.randn(3, 4, 4)
out = {fn}(A, None, None, 'trace')
ref = torch.einsum('bii->b', A)
assert out.shape == (3,), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Trace mismatch'""",
        },
        {
            "name": 'Outer product',
            "code": """import torch
torch.manual_seed(0)
a = torch.randn(4)
b = torch.randn(5)
out = {fn}(a, b, None, 'outer')
ref = torch.outer(a, b)
assert out.shape == (4, 5), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Outer product mismatch'""",
        },
        {
            "name": 'Diagonal extraction',
            "code": """import torch
torch.manual_seed(0)
A = torch.randn(4, 4)
out = {fn}(A, None, None, 'diagonal')
ref = torch.diag(A)
assert out.shape == (4,), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Diagonal mismatch'""",
        },
        {
            "name": 'Uses torch.einsum only',
            "code": """import torch, inspect
source = inspect.getsource({fn})
assert 'torch.einsum' in source, 'Must use torch.einsum'
assert 'torch.matmul' not in source, 'Cannot use torch.matmul'
assert 'torch.bmm' not in source, 'Cannot use torch.bmm'""",
        },
    ],
    'solution': '''def einsum_operations(A, B, C, operation):
    """用爱因斯坦求和实现常见张量运算 (Einsum Operations)。"""
    # Step 1: 根据操作名选择对应的索引缩并规则
    if operation == "batch_matmul":
        return torch.einsum("bmk,bkn->bmn", A, B)
    if operation == "bilinear":
        return torch.einsum("bm,mn,bn->b", A, B, C)

    # Step 2: 用重复索引处理迹与对角线提取
    if operation == "trace":
        return torch.einsum("bii->b", A)
    if operation == "diagonal":
        return torch.einsum("ii->i", A)

    # Step 3: 保留两个独立索引以构造向量外积
    if operation == "outer":
        return torch.einsum("m,n->mn", A, B)
    raise ValueError(f"Unsupported operation: {operation}")

einsum_operations.__wrapped__ = torch.einsum''',
    'demo': """torch.manual_seed(42)
A = torch.randn(2, 3, 4)
B = torch.randn(2, 4, 5)
batch_out = einsum_operations(A, B, None, "batch_matmul")
outer_out = einsum_operations(torch.randn(4), torch.randn(5), None, "outer")
print("Batch matmul shape:", batch_out.shape)
print("Outer product shape:", outer_out.shape)""",
    'explanation': """### 核心思路
`torch.einsum` 用索引是否出现在输出中决定保留或缩并维度，例如 `bmk,bkn->bmn` 对共享的 $k$ 求和并保留批次与矩阵轴。

### 代码步骤拆解
- **Step 1（矩阵缩并）**：批量矩阵乘法缩并 `k`，双线性形式同时缩并 `m`、`n` 并保留批次 `b`。
- **Step 2（重复索引）**：`bii->b` 对对角线求和得到迹，`ii->i` 保留重复索引得到对角元素。
- **Step 3（外积构造）**：`m,n->mn` 不缩并输入轴，生成所有向量元素的两两乘积。

### 重点与易错点分析
- **注意点 1（索引含义）**：相同字母代表维度必须一致；未出现在输出中的索引会被求和。
- **注意点 2（迹与对角线）**：两者输入写法相同，区别在输出是否保留 `i`，面试中经常混淆。
- **注意点 3（接口分支）**：不同操作使用的输入数量不同，未知 `operation` 应显式报错而非静默返回错误结果。

### 复杂度分析
- **时间复杂度**：依操作而定；批量矩阵乘法为 $O(BMKN)$，是给定操作中的主要瓶颈。
- **空间复杂度**：等于输出规模，批量矩阵乘法为 $O(BMN)$，外积为 $O(MN)$。""",
}
