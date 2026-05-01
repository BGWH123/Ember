"""PyTorch Einsum Operations task."""

TASK = {
    "title": "PyTorch Einsum Operations",
    "title_zh": "PyTorch Einsum 操作",
    "difficulty": "Medium",
    "category": "基础网络组件",
    "description_en": (
        "Implement common deep learning operations using torch.einsum.\n\nEinsum is a powerful unified notation for tensor operations. Mastering it is essential for reading research code and implementing custom operations efficiently.\n\nSignature: einsum_operations(A, B, C, operation) -> Tensor\n\nParameters:\n- A, B, C -- input tensors\n- operation -- string, one of: 'batch_matmul', 'bilinear', 'trace', 'outer', 'diagonal'\n\nOperations:\n- 'batch_matmul': batch matrix multiplication (B, M, K) @ (B, K, N) -> (B, M, N)\n- 'bilinear': bilinear form (B, M) @ (M, N) @ (B, N) -> (B,)\n- 'trace': trace of each matrix in batch (B, N, N) -> (B,)\n- 'outer': outer product of two vectors (M,) @ (N,) -> (M, N)\n- 'diagonal': extract diagonal (N, N) -> (N,)\n\nConstraints:\n- Must use torch.einsum for all operations\n- No torch.matmul, torch.bmm, or other built-in ops"
    ),
    "description_zh": (
        "使用 torch.einsum 实现常见的深度学习操作。\n\nEinsum 是一种强大的张量操作统一表示法。掌握它对阅读研究代码和高效实现自定义操作至关重要。\n\n签名: einsum_operations(A, B, C, operation) -> Tensor\n\n参数:\n- A, B, C -- 任意形状的输入张量\n- operation -- 字符串，取以下之一: 'batch_matmul', 'bilinear', 'trace', 'outer', 'diagonal'\n\n操作:\n- 'batch_matmul': 批次矩阵乘法 (B, M, K) @ (B, K, N) -> (B, M, N)\n- 'bilinear': 双线性形式 (B, M) @ (M, N) @ (B, N) -> (B,)\n- 'trace': 批次中每个矩阵的迹 (B, N, N) -> (B,)\n- 'outer': 两个向量的外积 (M,) @ (N,) -> (M, N)\n- 'diagonal': 提取对角线 (N, N) -> (N,)\n\n约束:\n- 所有操作必须使用 torch.einsum\n- 禁止使用 torch.matmul、torch.bmm 或其他内置运算"
    ),
    "function_name": "einsum_operations",
    "hint": (
        "- batch_matmul: einsum('bmk,bkn->bmn', A, B)\n- bilinear: einsum('bm,mn,bn->b', A, B, C)\n- trace: einsum('bii->b', A)\n- outer: einsum('m,n->mn', A, B)\n- diagonal: einsum('ii->i', A)"
    ),
    "hint_zh": (
        "- batch_matmul: einsum('bmk,bkn->bmn', A, B)\n- bilinear: einsum('bm,mn,bn->b', A, B, C)\n- trace: einsum('bii->b', A)\n- outer: einsum('m,n->mn', A, B)\n- diagonal: einsum('ii->i', A)"
    ),
    "theory_en": (
        "Einsum (Einstein summation) provides a unified notation for tensor contractions.\n\nBasic Rules:\n- Repeated indices in input are summed over (contraction)\n- Output indices specify which dimensions remain\n- Batch dimensions appear in all inputs and the output\n\nExamples:\n- Matrix multiplication: ik,kj->ij\n- Batch matrix multiplication: bik,bkj->bij\n- Trace: ii->\n- Diagonal extraction: ii->i\n- Outer product: i,j->ij\n- Batch dot product: bi,bi->b\n\nAdvantages over explicit ops:\n- Single function covers matmul, bmm, trace, diagonal, outer, dot, transpose\n- Often fused by PyTorch for better performance\n- More readable for complex tensor contractions"
    ),
    "theory_zh": (
        "Einsum（爱因斯坦求和约定）为张量缩并提供统一表示法。\n\n基本规则:\n- 输入中重复的索引被求和（收缩）\n- 输出索引指定保留哪些维度\n- 批次维度出现在所有输入和输出中\n\n示例:\n- 矩阵乘法: ik,kj->ij\n- 批次矩阵乘法: bik,bkj->bij\n- 迹: ii->\n- 提取对角线: ii->i\n- 外积: i,j->ij\n- 批次点积: bi,bi->b\n\n相比显式操作的优势:\n- 单个函数覆盖 matmul、bmm、trace、diagonal、outer、dot、transpose\n- PyTorch 经常将其融合以获得更好性能\n- 复杂张量收缩更易读"
    ),
    "diagram_en": (
        "flowchart LR\n    subgraph batch_matmul\n        A1[A: b,m,k] --> E1[einsum bmk,bkn-bmn]\n        B1[B: b,k,n] --> E1\n        E1 --> O1[Out: b,m,n]\n    end\n    subgraph trace\n        A2[A: b,i,i] --> E2[einsum bii-b]\n        E2 --> O2[Out: b]\n    end"
    ),
    "diagram_zh": (
        "flowchart LR\n    subgraph batch_matmul\n        A1[A: b,m,k] --> E1[einsum bmk,bkn-bmn]\n        B1[B: b,k,n] --> E1\n        E1 --> O1[Out: b,m,n]\n    end\n    subgraph trace\n        A2[A: b,i,i] --> E2[einsum bii-b]\n        E2 --> O2[Out: b]\n    end"
    ),
    "tests": [
        {
            "name": "Batch matrix multiplication",
            "code": """







import torch
torch.manual_seed(0)
A = torch.randn(2, 3, 4)
B = torch.randn(2, 4, 5)
out = {fn}(A, B, None, 'batch_matmul')
ref = torch.bmm(A, B)
assert out.shape == (2, 3, 5), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Does not match torch.bmm'

            
            
            
            
            
            
            """,
        },
        {
            "name": "Bilinear form",
            "code": """







import torch
torch.manual_seed(0)
A = torch.randn(3, 4)
M = torch.randn(4, 5)
C = torch.randn(3, 5)
out = {fn}(A, M, C, 'bilinear')
ref = torch.einsum('bm,mn,bn->b', A, M, C)
assert out.shape == (3,), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Bilinear mismatch'

            
            
            
            
            
            
            """,
        },
        {
            "name": "Trace",
            "code": """







import torch
torch.manual_seed(0)
A = torch.randn(3, 4, 4)
out = {fn}(A, None, None, 'trace')
ref = torch.einsum('bii->b', A)
assert out.shape == (3,), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Trace mismatch'

            
            
            
            
            
            
            """,
        },
        {
            "name": "Outer product",
            "code": """







import torch
torch.manual_seed(0)
a = torch.randn(4)
b = torch.randn(5)
out = {fn}(a, b, None, 'outer')
ref = torch.outer(a, b)
assert out.shape == (4, 5), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Outer product mismatch'

            
            
            
            
            
            
            """,
        },
        {
            "name": "Diagonal extraction",
            "code": """







import torch
torch.manual_seed(0)
A = torch.randn(4, 4)
out = {fn}(A, None, None, 'diagonal')
ref = torch.diag(A)
assert out.shape == (4,), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Diagonal mismatch'

            
            
            
            
            
            
            """,
        },
        {
            "name": "Uses torch.einsum only",
            "code": """







import torch, inspect
source = inspect.getsource({fn})
assert 'torch.einsum' in source, 'Must use torch.einsum'
assert 'torch.matmul' not in source, 'Cannot use torch.matmul'
assert 'torch.bmm' not in source, 'Cannot use torch.bmm'

            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


def einsum_operations(A, B):
    """
    Einsum operations demonstration.
    实现 batch matrix multiplication、trace、outer product 和 diagonal extraction。
    """
    results = {}

    # 1. Batch Matrix Multiplication: 'bmk,bkn->bmn'
    # A: (B, M, K), B: (B, K, N) -> out: (B, M, N)
    # 相当于对每个 batch 做独立的矩阵乘法
    results['batch_matmul'] = torch.einsum('bmk,bkn->bmn', A, B)

    # 2. Matrix Trace (对角线元素之和): 'bii->b'
    # A: (B, I, I) -> out: (B,)
    # trace = Σ_i A[:, i, i]，即每个 batch 矩阵对角线元素之和
    results['trace'] = torch.einsum('bii->b', A)

    # 3. Outer Product: 'i,j->ij'
    # a: (I,), b: (J,) -> out: (I, J)
    # outer[i,j] = a[i] * b[j]，所有元素两两相乘
    a = A[0, :, 0]                                     # 取第一个 batch 的第一列: (M,)
    b = B[0, 0, :]                                     # 取第一个 batch 的第一行: (N,)
    results['outer'] = torch.einsum('i,j->ij', a, b)   # (M, N)

    # 4. Diagonal Extraction: 'ii->i'
    # A: (I, I) -> out: (I,)
    # 提取方阵的对角线元素
    mat = A[0] if A.ndim >= 2 else A                   # 取第一个 batch 的 (M, M) 子矩阵
    min_dim = min(mat.shape[-2], mat.shape[-1])
    results['diagonal'] = torch.einsum('ii->i', mat[:min_dim, :min_dim])   # (min_dim,)

    return results

    
    
    ''',
    "demo": '''






torch.manual_seed(0)
A = torch.randn(2, 3, 4)
B = torch.randn(2, 4, 5)
print("batch_matmul:", einsum_operations(A, B, None, 'batch_matmul').shape)

a = torch.randn(4)
b = torch.randn(5)
print("outer:", einsum_operations(a, b, None, 'outer').shape)

M = torch.randn(3, 3)
print("diagonal:", einsum_operations(M, None, None, 'diagonal'))
    
    
    
    
    
    
    ''',
}
