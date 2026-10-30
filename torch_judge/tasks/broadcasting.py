"""PyTorch Broadcasting Mechanics task."""

TASK = {
    'title': 'PyTorch Broadcasting Mechanics',
    'title_zh': 'PyTorch 广播机制',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'broadcast_and_add',
    'starter_code': """def broadcast_and_add(A, B):
    pass""",
    'hint': """1. Compute broadcasted shape by comparing dims from right to left
2. Use `unsqueeze` and `expand` to manually broadcast each tensor
3. Return the sum of expanded tensors""",
    'hint_zh': """1. 从右到左比较维度，计算广播后的形状
2. 使用 `unsqueeze` 和 `expand` 手动广播每个张量
3. 返回扩展后张量的和""",
    'description_en': """Implement a function that manually performs PyTorch-style broadcasting between two tensors.

Broadcasting allows tensors with different shapes to be combined element-wise by implicitly expanding dimensions of size 1 to match the other tensor.

**Signature:** `broadcast_and_add(A, B) -> Tensor`

**Parameters:**
- `A`, `B` — input tensors of arbitrary shape

**Returns:** element-wise sum after broadcasting, same as `A + B` in PyTorch

**Broadcasting Rules:**
1. Align dimensions from the right
2. Two dimensions are compatible if they are equal or one of them is 1
3. A dimension of size 1 is stretched to match the other
4. If dimensions are incompatible, raise ValueError

**Examples:**
- `(3, 1) + (1, 4) -> (3, 4)`
- `(2, 3, 1) + (1, 3, 5) -> (2, 3, 5)`
- `(5,) + (3, 1) -> incompatible (raise ValueError)`""",
    'description_zh': """实现一个手动执行 PyTorch 风格广播的函数。

广播允许不同形状的张量通过隐式扩展大小为 1 的维度来进行逐元素组合。

**签名:** `broadcast_and_add(A, B) -> Tensor`

**参数:**
- `A`, `B` — 任意形状的输入张量

**返回:** 广播后的逐元素和，与 PyTorch 中 `A + B` 相同

**广播规则：**
1. 从右侧对齐维度
2. 两个维度兼容的条件是相等或其中一个为 1
3. 大小为 1 的维度被拉伸以匹配另一个
4. 如果维度不兼容，抛出 ValueError

**示例：**
- `(3, 1) + (1, 4) -> (3, 4)`
- `(2, 3, 1) + (1, 3, 5) -> (2, 3, 5)`
- `(5,) + (3, 1) -> 不兼容（抛出 ValueError）`""",
    'theory_en': """Broadcasting is a fundamental mechanism in NumPy/PyTorch that eliminates the need for explicit dimension expansion in many operations.

**Rules (aligned from the right):**
| A dim | B dim | Result |
|-------|-------|--------|
| 3     | 3     | 3      |
| 3     | 1     | 3      |
| 1     | 3     | 3      |
| 3     | 4     | Error  |

**Memory Efficiency:**
Broadcasting does not actually allocate memory for the expanded tensor. It uses stride tricks to repeat elements without copying. However, for this exercise, we use explicit `expand` for clarity.

**Common Use Cases:**
- Adding bias to batched features: `(B, D) + (D,)`
- Scaling per-channel: `(B, C, H, W) * (C, 1, 1)`
- Attention mask broadcasting: `(B, 1, S) + (B, S, S)`""",
    'theory_zh': """广播是 NumPy/PyTorch 中的基础机制，消除了许多操作中显式维度扩展的需要。

**规则（从右侧对齐）：**
| A 维度 | B 维度 | 结果 |
|--------|--------|------|
| 3      | 3      | 3    |
| 3      | 1      | 3    |
| 1      | 3      | 3    |
| 3      | 4      | 错误 |

**内存效率：**
广播不会实际为扩展后的张量分配内存。它使用步长技巧在不复制的情况下重复元素。但在此练习中，我们使用显式 `expand` 以清晰展示原理。

**常见用例：**
- 为批次特征添加偏置：`(B, D) + (D,)`
- 逐通道缩放：`(B, C, H, W) * (C, 1, 1)`
- 注意力掩码广播：`(B, 1, S) + (B, S, S)`""",
    'diagram_en': """```mermaid
flowchart LR
    A[(3,1)] --> BROADCAST[Broadcast]
    B[(1,4)] --> BROADCAST
    BROADCAST --> A2[(3,4)]
    BROADCAST --> B2[(3,4)]
    A2 --> ADD[Element-wise Add]
    B2 --> ADD
    ADD --> OUT[(3,4)]

    style BROADCAST fill:#e1f5fe
```""",
    'diagram_zh': """```mermaid
flowchart LR
    A[(3,1)] --> BROADCAST[广播]
    B[(1,4)] --> BROADCAST
    BROADCAST --> A2[(3,4)]
    BROADCAST --> B2[(3,4)]
    A2 --> ADD[逐元素相加]
    B2 --> ADD
    ADD --> OUT[(3,4)]

    style BROADCAST fill:#e1f5fe
```""",
    "tests": [
        {
            "name": 'Simple broadcasting',
            "code": """import torch
A = torch.tensor([[1], [2], [3]])  # (3, 1)
B = torch.tensor([[10, 20, 30, 40]])  # (1, 4)
out = {fn}(A, B)
ref = A + B
assert out.shape == (3, 4), f'Shape mismatch: {out.shape}'
assert torch.equal(out, ref), f'Mismatch: {out} vs {ref}'""",
        },
        {
            "name": '3D broadcasting',
            "code": """import torch
torch.manual_seed(0)
A = torch.randn(2, 3, 1)
B = torch.randn(1, 3, 5)
out = {fn}(A, B)
ref = A + B
assert out.shape == (2, 3, 5), f'Shape mismatch: {out.shape}'
assert torch.allclose(out, ref, atol=1e-5), 'Mismatch with reference'""",
        },
        {
            "name": 'No broadcasting needed',
            "code": """import torch
torch.manual_seed(0)
A = torch.randn(4, 5)
B = torch.randn(4, 5)
out = {fn}(A, B)
ref = A + B
assert torch.allclose(out, ref, atol=1e-5), 'Should work without broadcasting'""",
        },
        {
            "name": 'Raises ValueError for incompatible shapes',
            "code": """import torch
A = torch.randn(3, 4)
B = torch.randn(2, 4)
try:
    {fn}(A, B)
    assert False, 'Should raise ValueError for incompatible shapes'
except ValueError:
    pass""",
        },
        {
            "name": 'Scalar broadcasting',
            "code": """import torch
A = torch.randn(3, 4)
B = torch.tensor(5.0)
out = {fn}(A, B)
ref = A + B
assert torch.allclose(out, ref, atol=1e-5), 'Scalar broadcasting failed'""",
        },
    ],
    'solution': '''def broadcast_and_add(A, B):
    """手动广播加法 (Tensor Broadcasting)。"""
    # Step 1: 从左侧补 1，使两个形状具有相同秩
    shape_a = list(A.shape)
    shape_b = list(B.shape)
    max_ndim = max(len(shape_a), len(shape_b))
    shape_a = [1] * (max_ndim - len(shape_a)) + shape_a
    shape_b = [1] * (max_ndim - len(shape_b)) + shape_b

    # Step 2: 逐维验证兼容性并确定输出形状
    output_shape = []
    for dim_a, dim_b in zip(shape_a, shape_b):
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            output_shape.append(max(dim_a, dim_b))
        else:
            raise ValueError(
                f"Incompatible shapes for broadcasting: {A.shape} and {B.shape}"
            )

    # Step 3: 补齐维度、扩展视图并逐元素相加
    A_view = A.reshape(*shape_a).expand(*output_shape)
    B_view = B.reshape(*shape_b).expand(*output_shape)
    return A_view + B_view''',
    'demo': """torch.manual_seed(42)
A = torch.tensor([[1], [2], [3]])
B = torch.tensor([[10, 20, 30, 40]])
out = broadcast_and_add(A, B)
print("Output shape:", out.shape)
print("First row:", out[0])
print("Sample values:", out[:, 0])""",
    'explanation': """### 核心思路
广播从末维向前对齐；每一维必须相等或至少一方为 1，输出维度取两者较大值，再通过零拷贝 `expand` 对齐。

### 代码步骤拆解
- **Step 1（秩对齐）**：在较短形状左侧补 1，得到等长的 `shape_a/shape_b`。
- **Step 2（兼容检查）**：逐维生成 `output_shape`，不兼容时立即抛出异常。
- **Step 3（视图扩展）**：`reshape` 补维后用 `expand` 创建广播视图并相加。

### 重点与易错点分析
- **注意点 1（对齐方向）**：广播从末尾维度对齐，不能在形状右侧补 1。
- **注意点 2（零拷贝视图）**：`expand` 通过步长为 0 复用数据，不应对展开视图做冲突的原地写入。
- **注意点 3（标量边界）**：空形状标量补成全 1 后，同样可广播到任意兼容输出。

### 复杂度分析
- **时间复杂度**：$O(R+N)$，检查 $R$ 个维度并生成 $N$ 个输出元素。
- **空间复杂度**：除输出外为 $O(R)$，`expand` 本身不复制底层数据。""",
}
