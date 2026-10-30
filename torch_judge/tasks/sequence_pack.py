"""Sequence Packing task."""

TASK = {
    'title': 'Sequence Packing',
    'title_zh': '序列打包',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'sequence_pack',
    'starter_code': """def sequence_pack(sequences):
    pass""",
    'hint': """Find max length, create a zero tensor of shape (batch_size, max_len), then copy each sequence into its corresponding row.""",
    'hint_zh': '找到最大长度，创建形状为 (batch_size, max_len) 的零张量，然后将每个序列复制到对应行。',
    'description_en': """Implement sequence packing for efficient training. Given a list of variable-length sequences (each is a 1D tensor), pack them into a single padded tensor of shape (batch_size, max_len). Also return a binary mask indicating valid positions (1 for real data, 0 for padding).""",
    'description_zh': """实现序列打包以提高训练效率。给定一批变长序列（每个是一维张量），将它们打包成单个填充张量，形状为 (batch_size, max_len)。同时返回一个二元掩码，标记有效位置（1 表示真实数据，0 表示填充）。""",
    'theory_en': """Sequence packing is essential for batched training of variable-length sequences (NLP, time series). Padding allows tensor operations while masks prevent attention to padded positions.""",
    'theory_zh': '序列打包是变长序列（NLP、时间序列）批量训练的基础。填充使张量运算成为可能，掩码防止注意力关注填充位置。',
    "tests": [
        {
            "name": 'Ragged sequences and valid zeros',
            "code": """seqs = [torch.tensor([0, 4, 2]), torch.tensor([7]), torch.tensor([0, 8])]
padded, mask = {fn}(seqs)
assert torch.equal(padded, torch.tensor([[0, 4, 2], [7, 0, 0], [0, 8, 0]]))
assert torch.equal(mask, torch.tensor([[1., 1., 1.], [1., 0., 0.], [1., 1., 0.]]))
assert padded.dtype == seqs[0].dtype""",
        },
        {
            "name": 'Empty member and all empty members',
            "code": """padded, mask = {fn}([torch.tensor([]), torch.tensor([1., 2.])])
assert torch.equal(padded, torch.tensor([[0., 0.], [1., 2.]]))
assert torch.equal(mask, torch.tensor([[0., 0.], [1., 1.]]))
padded, mask = {fn}([torch.tensor([]), torch.tensor([])])
assert padded.shape == mask.shape == (2, 0)""",
        },
        {
            "name": 'Preserves input and gradients',
            "code": """a = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
b = torch.tensor([-1.], dtype=torch.float64, requires_grad=True)
padded, mask = {fn}([a, b])
assert padded.dtype == torch.float64
padded.sum().backward()
assert torch.equal(a.grad, torch.ones_like(a))
assert torch.equal(b.grad, torch.ones_like(b))
assert torch.equal(a.detach(), torch.tensor([2., 3.], dtype=torch.float64))""",
        },
    ],
    'solution': '''def sequence_pack(sequences):
    """实现序列打包（Sequence Packing）。"""
    # Step 1: 扫描变长序列并确定批张量的统一时间宽度
    max_len = max((seq.shape[0] for seq in sequences))
    batch_size = len(sequences)
    padded = torch.zeros(batch_size, max_len, dtype=sequences[0].dtype, device=sequences[0].device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32, device=sequences[0].device)
    # Step 2: 将每条序列复制到对应行的左对齐有效区间
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length] = seq
        # Step 3: 仅依据原始长度标记有效位置，避免混淆真实零值与填充零
        mask[i, :length] = 1.0
    return (padded, mask)''',
    'demo': r"""torch.manual_seed(42)
sequences = [torch.tensor([0, 4, 2]), torch.tensor([7]), torch.tensor([0, 8])]
padded, mask = sequence_pack(sequences)
print("Packed tokens:\n", padded)
print("Validity mask:\n", mask)
print("Recovered lengths:", mask.sum(dim=1).tolist())""",
    'explanation': r"""### 核心思路
设批中第 $i$ 条序列长度为 $L_i$，先取 $L_{\max}=\max_iL_i$，再构造两个 $(B,L_{\max})$ 张量：`padded` 在每行前 $L_i$ 个位置保存原数据，其余补零；`mask` 在同一区间为 1。掩码记录的是结构长度而非数据值，因此即使真实 token 等于 0 也不会被误判为 padding。

### 代码步骤拆解
- **Step 1（确定批次画布）**：扫描 `sequences` 得到 `batch_size` 和 `max_len`，在首个序列的 dtype、device 上分配 `padded` 与浮点 `mask`。
- **Step 2（左对齐复制有效片段）**：对第 `i` 条长度为 `length` 的 `seq`，写入 `padded[i, :length]`；空序列对应切片长度为 0，不会写入元素。
- **Step 3（按长度生成有效位）**：把 `mask[i, :length]` 置为 1，最终两个返回张量具有相同形状，且 `mask.sum(1)` 可恢复每条原始长度。

### 重点与易错点分析
- **注意点 1（真实零值）**：不能用 `padded != 0` 推导 mask，词表中的 token 0 或时间序列中的零观测都可能是有效数据。
- **注意点 2（全空批次）**：若批中成员都为空，`max_len=0`，合法输出形状是 $(B,0)$；但输入列表本身必须非空，因为还需从首元素确定 dtype 与 device。
- **注意点 3（梯度路径）**：切片复制生成 `CopySlices` 计算图，`padded` 上的损失仍可回传到各浮点输入序列；不能先把序列转成 Python 列表再重新建张量，否则梯度会断开。

### 复杂度分析
- **时间复杂度**：$O(BL_{\max}+\sum_iL_i)$，输出清零分配和逐序列有效片段复制共同决定成本。
- **空间复杂度**：$O(BL_{\max})$，`padded` 与 `mask` 是两块同尺寸的主要输出存储。""",
}
