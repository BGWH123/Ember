"""PyTorch Broadcasting Mechanics task."""

TASK = {
    "title": "PyTorch Broadcasting Mechanics",
    "title_zh": "PyTorch 广播机制",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "description_en": (
        "Implement a function that manually performs PyTorch-style broadcasting between two tensors.\n\nBroadcasting allows tensors with different shapes to be combined element-wise by implicitly expanding dimensions of size 1 to match the other tensor.\n\n**Signature:** `broadcast_and_add(A, B) -> Tensor`\n\n**Parameters:**\n- `A`, `B` — input tensors of arbitrary shape\n\n**Returns:** element-wise sum after broadcasting, same as `A + B` in PyTorch\n\n**Broadcasting Rules:**\n1. Align dimensions from the right\n2. Two dimensions are compatible if they are equal or one of them is 1\n3. A dimension of size 1 is stretched to match the other\n4. If dimensions are incompatible, raise ValueError\n\n**Examples:**\n- `(3, 1) + (1, 4) -> (3, 4)`\n- `(2, 3, 1) + (1, 3, 5) -> (2, 3, 5)`\n- `(5,) + (3, 1) -> incompatible (raise ValueError)`"
    ),
    "description_zh": (
        "实现一个手动执行 PyTorch 风格广播的函数。\n\n广播允许不同形状的张量通过隐式扩展大小为 1 的维度来进行逐元素组合。\n\n**签名:** `broadcast_and_add(A, B) -> Tensor`\n\n**参数:**\n- `A`, `B` — 任意形状的输入张量\n\n**返回:** 广播后的逐元素和，与 PyTorch 中 `A + B` 相同\n\n**广播规则：**\n1. 从右侧对齐维度\n2. 两个维度兼容的条件是相等或其中一个为 1\n3. 大小为 1 的维度被拉伸以匹配另一个\n4. 如果维度不兼容，抛出 ValueError\n\n**示例：**\n- `(3, 1) + (1, 4) -> (3, 4)`\n- `(2, 3, 1) + (1, 3, 5) -> (2, 3, 5)`\n- `(5,) + (3, 1) -> 不兼容（抛出 ValueError）`"
    ),
    "function_name": "broadcast_and_add",
    "hint": (
        "1. Compute broadcasted shape by comparing dims from right to left\n2. Use `unsqueeze` and `expand` to manually broadcast each tensor\n3. Return the sum of expanded tensors"
    ),
    "hint_zh": (
        "1. 从右到左比较维度，计算广播后的形状\n2. 使用 `unsqueeze` 和 `expand` 手动广播每个张量\n3. 返回扩展后张量的和"
    ),
    "theory_en": (
        "Broadcasting is a fundamental mechanism in NumPy/PyTorch that eliminates the need for explicit dimension expansion in many operations.\n\n**Rules (aligned from the right):**\n| A dim | B dim | Result |\n|-------|-------|--------|\n| 3     | 3     | 3      |\n| 3     | 1     | 3      |\n| 1     | 3     | 3      |\n| 3     | 4     | Error  |\n\n**Memory Efficiency:**\nBroadcasting does not actually allocate memory for the expanded tensor. It uses stride tricks to repeat elements without copying. However, for this exercise, we use explicit `expand` for clarity.\n\n**Common Use Cases:**\n- Adding bias to batched features: `(B, D) + (D,)`\n- Scaling per-channel: `(B, C, H, W) * (C, 1, 1)`\n- Attention mask broadcasting: `(B, 1, S) + (B, S, S)`"
    ),
    "theory_zh": (
        "广播是 NumPy/PyTorch 中的基础机制，消除了许多操作中显式维度扩展的需要。\n\n**规则（从右侧对齐）：**\n| A 维度 | B 维度 | 结果 |\n|--------|--------|------|\n| 3      | 3      | 3    |\n| 3      | 1      | 3    |\n| 1      | 3      | 3    |\n| 3      | 4      | 错误 |\n\n**内存效率：**\n广播不会实际为扩展后的张量分配内存。它使用步长技巧在不复制的情况下重复元素。但在此练习中，我们使用显式 `expand` 以清晰展示原理。\n\n**常见用例：**\n- 为批次特征添加偏置：`(B, D) + (D,)`\n- 逐通道缩放：`(B, C, H, W) * (C, 1, 1)`\n- 注意力掩码广播：`(B, 1, S) + (B, S, S)`"
    ),
    "diagram_en": (
        "```mermaid\nflowchart LR\n    A[(3,1)] --> BROADCAST[Broadcast]\n    B[(1,4)] --> BROADCAST\n    BROADCAST --> A2[(3,4)]\n    BROADCAST --> B2[(3,4)]\n    A2 --> ADD[Element-wise Add]\n    B2 --> ADD\n    ADD --> OUT[(3,4)]\n\n    style BROADCAST fill:#e1f5fe\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart LR\n    A[(3,1)] --> BROADCAST[广播]\n    B[(1,4)] --> BROADCAST\n    BROADCAST --> A2[(3,4)]\n    BROADCAST --> B2[(3,4)]\n    A2 --> ADD[逐元素相加]\n    B2 --> ADD\n    ADD --> OUT[(3,4)]\n\n    style BROADCAST fill:#e1f5fe\n```"
    ),
    "tests": [
        {
            "name": "Simple broadcasting",
            "code": """








import torch
A = torch.tensor([[1], [2], [3]])  # (3, 1)
B = torch.tensor([[10, 20, 30, 40]])  # (1, 4)
out = {fn}(A, B)
ref = A + B
assert out.shape == (3, 4), f'Shape mismatch: {out.shape}'
assert torch.equal(out, ref), f'Mismatch: {out} vs {ref}'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "3D broadcasting",
            "code": """








import torch
torch.manual_seed(0)
A = torch.randn(2, 3, 1)
B = torch.randn(1, 3, 5)
out = {fn}(A, B)
ref = A + B
assert out.shape == (2, 3, 5), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Mismatch with reference'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "No broadcasting needed",
            "code": """








import torch
torch.manual_seed(0)
A = torch.randn(4, 5)
B = torch.randn(4, 5)
out = {fn}(A, B)
ref = A + B
assert torch.allclose(out, ref, atol=1e-5), 'Should work without broadcasting'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Raises ValueError for incompatible shapes",
            "code": """








import torch
A = torch.randn(3, 4)
B = torch.randn(2, 4)
try:
    {fn}(A, B)
    assert False, 'Should raise ValueError for incompatible shapes'
except ValueError:
    pass

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Scalar broadcasting",
            "code": """








import torch
A = torch.randn(3, 4)
B = torch.tensor(5.0)
out = {fn}(A, B)
ref = A + B
assert torch.allclose(out, ref, atol=1e-5), 'Scalar broadcasting failed'

            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


def broadcast_and_add(A, B):
    """
    手动实现 PyTorch 风格的广播机制（Broadcasting）。
    广播规则: 从后往前比较维度，要么相等，要么其中一个为 1，要么缺失（视为 1）。
    """
    # 获取形状并在左侧补 1，使两个张量维度相同
    shape_a = list(A.shape)                            # 例如 [3, 1]
    shape_b = list(B.shape)                            # 例如 [1, 4]
    max_ndim = max(len(shape_a), len(shape_b))         # 最大维度数
    shape_a = [1] * (max_ndim - len(shape_a)) + shape_a   # 左侧补 1: [3, 1] -> [3, 1] (假设 2D)
    shape_b = [1] * (max_ndim - len(shape_b)) + shape_b   # 左侧补 1: [1, 4] -> [1, 4]

    # 计算广播后的输出形状
    out_shape = []
    for da, db in zip(shape_a, shape_b):
        if da == db:
            out_shape.append(da)                       # 维度相同，直接保留
        elif da == 1:
            out_shape.append(db)                       # A 维度为 1，广播为 B 的维度
        elif db == 1:
            out_shape.append(da)                       # B 维度为 1，广播为 A 的维度
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {A.shape} and {B.shape}")

    # 辅助函数: 将张量扩展到目标形状
    def expand_tensor(t, from_shape, to_shape):
        # 在左侧补充维度 (unsqueeze)
        for _ in range(len(to_shape) - t.ndim):
            t = t.unsqueeze(0)                         # 在最左侧添加新维度
        # 对大小为 1 的维度进行扩展
        for i, (fs, ts) in enumerate(zip(from_shape, to_shape)):
            if fs == 1 and ts != 1:
                t = t.expand(*to_shape[:i+1], *t.shape[i+1:])   # expand 到目标形状
        return t

    # 扩展两个张量到广播后的形状
    A_exp = expand_tensor(A, shape_a, out_shape)       # (out_shape)
    B_exp = expand_tensor(B, shape_b, out_shape)       # (out_shape)

    # 逐元素相加
    return A_exp + B_exp                               # (out_shape)

    
    
    ''',
    "demo": '''







A = torch.tensor([[1], [2], [3]])  # (3, 1)
B = torch.tensor([[10, 20, 30, 40]])  # (1, 4)
out = broadcast_and_add(A, B)
print("A shape:", A.shape)
print("B shape:", B.shape)
print("Output shape:", out.shape)
print("Output:
", out)
    
    
    
    
    
    
    
    ''',
}
