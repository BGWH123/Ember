"""One-Hot Encoding task."""

TASK = {
    'title': 'One-Hot Encoding',
    'title_zh': 'One-Hot 编码',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'one_hot_encoding',
    'starter_code': """def one_hot_encoding(indices, num_classes):
    pass""",
    'hint': 'Use `torch.zeros` to create the output tensor, then use advanced indexing or `scatter_` to set 1s.',
    'hint_zh': '使用 `torch.zeros` 创建输出张量，然后用高级索引或 `scatter_` 设置1。',
    'description_en': """Implement one-hot encoding. Given a tensor of class indices (0 to num_classes-1), return a tensor where each row is a one-hot vector. For example, indices [0, 2, 1] with num_classes=4 → [[1,0,0,0], [0,0,1,0], [0,1,0,0]].""",
    'description_zh': """实现 One-Hot 编码。给定类别索引张量（0 到 num_classes-1），返回 one-hot 向量张量。例如 indices [0, 2, 1] 且 num_classes=4 → [[1,0,0,0], [0,0,1,0], [0,1,0,0]]。""",
    'theory_en': """One-hot encoding converts categorical class indices into binary vectors. It is the standard input format for classification tasks and the target format for cross-entropy loss.""",
    'theory_zh': 'One-Hot 编码将类别索引转换为二元向量。是分类任务的标准输入格式，也是交叉熵损失的目标格式。',
    "tests": [
        {
            "name": 'Class positions including repeated indices',
            "code": """idx = torch.tensor([0, 3, 1, 3, 0])
out = {fn}(idx, 4)
assert out.shape == (5, 4)
assert torch.equal(out, F.one_hot(idx, 4).float())""",
        },
        {
            "name": 'Single class and empty input',
            "code": """assert torch.equal({fn}(torch.zeros(3, dtype=torch.long), 1), torch.ones(3, 1))
assert {fn}(torch.tensor([], dtype=torch.long), 5).shape == (0, 5)""",
        },
        {
            "name": 'Does not change indices',
            "code": """idx = torch.tensor([2, 1, 0])
before = idx.clone()
out = {fn}(idx, 5)
assert torch.equal(idx, before)
assert torch.equal(out.sum(-1), torch.ones(3))
assert torch.all((out == 0) | (out == 1))""",
        },
    ],
    'solution': '''def one_hot_encoding(indices, num_classes):
    """One-Hot 编码 (One-Hot Encoding)。"""
    # Step 1: 为 N 个类别索引分配 N×C 的全零浮点指示矩阵
    N = indices.shape[0]
    one_hot = torch.zeros(N, num_classes, dtype=torch.float32)
    # Step 2: 将索引扩为列坐标，并在每行对应类别位置写入 1
    one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
    # Step 3: 返回每行恰有一个激活类别的 one-hot 表示
    return one_hot''',
    'demo': """torch.manual_seed(42)
indices = torch.tensor([0, 2, 1, 3])
out = one_hot_encoding(indices, 4)
print('类别索引:', indices.tolist())
print('One-hot 形状:', tuple(out.shape))
print('编码矩阵:', out.tolist())""",
    'explanation': r"""### 核心思路
One-hot 编码把第 i 个离散类别索引 $x_i\in\{0,\ldots,C-1\}$ 映射为标准基向量：$Y_{i,c}=\mathbf1[c=x_i]$。因此输出是 `(N,C)` 浮点矩阵，每行只有索引列为 1，其余为 0；类别之间不引入虚假的数值距离关系。

### 代码步骤拆解
- **Step 1（初始化类别平面）**：令 `N=indices.shape[0]`，创建 `(N,num_classes)` 的 float32 零矩阵；空输入自然得到 `(0,C)`。
- **Step 2（按行散射）**：`indices.unsqueeze(1)` 从 `(N,)` 变成 `(N,1)`，作为 `dim=1` 的列坐标，`scatter_` 在每行目标类别处写入 1。
- **Step 3（保持样本顺序）**：散射只修改新建的 `one_hot`，不会改动原 `indices`；返回行顺序与输入索引顺序一一对应。

### 重点与易错点分析
- **注意点 1（索引 dtype 与范围）**：`scatter_` 的索引必须是整数类型，且每个值满足 $0\le x_i<C$；越界类别会直接报错而不是被截断。
- **注意点 2（设备一致性）**：通用 GPU 实现应让零矩阵位于 `indices.device`；本题测试使用 CPU，当前代码按题目参考创建 CPU 输出。
- **注意点 3（与嵌入查表的取舍）**：one-hot 乘嵌入矩阵在数学上等于按索引查表，但显式 one-hot 需要 $NC$ 存储；大词表模型通常直接 embedding lookup。

### 复杂度分析
- **时间复杂度**：零初始化写入 $NC$ 个元素为 $O(NC)$，散射 N 个 1 为 $O(N)$；初始化占主导。
- **空间复杂度**：输出矩阵占 $O(NC)$，列索引视图仅占 $O(N)$ 的逻辑大小。""",
}
