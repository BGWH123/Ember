"""Tree Speculative Decoding Attention Mask task."""

TASK = {
    'title': 'Tree Speculative Decoding Attention Mask',
    'title_zh': '树状推测解码注意力掩码（Medusa / EAGLE）',
    'difficulty': 'Medium',
    'category': '推理优化',
    'function_name': 'tree_speculative_mask',
    'starter_code': """def tree_speculative_mask(parent_indices):
    pass""",
    'hint': """Process nodes in index order. Copy the parent's already-computed ancestor row, then mark the node's own diagonal entry true.""",
    'hint_zh': '按索引顺序处理节点：复制父节点已计算好的祖先行，再把当前节点自身的对角位置设为 True。',
    'description_en': 'Construct the ancestor visibility mask for a parent-before-child speculative decoding tree.',
    'description_zh': '为父节点先于子节点排列的推测解码树构造祖先可见性掩码。',
    'theory_en': """A draft-tree node may attend only to tokens on its own root-to-node path; sibling branches must remain isolated.""",
    'theory_zh': '草稿树节点只能关注自身从根到该节点的路径，兄弟分支必须相互隔离。',
    'diagram_en': r"""```text
0
+- 1 +- 3
|    \- 4
\- 2
row(4) sees {0,1,4}
```""",
    'diagram_zh': r"""```text
0
+- 1 +- 3
|    \- 4
\- 2
第 4 行可见 {0,1,4}
```""",
    "tests": [
        {
            "name": 'shape dtype and diagonal',
            "code": """import torch
mask = {fn}(torch.tensor([-1, 0, 0, 1, 1, 2]))
assert mask.shape == (6, 6) and mask.dtype == torch.bool
assert torch.equal(mask.diagonal(), torch.ones(6, dtype=torch.bool))""",
        },
        {
            "name": 'exact branching tree',
            "code": """import torch
mask = {fn}([-1, 0, 0, 1, 1, 2])
ref = torch.tensor([
    [1,0,0,0,0,0],
    [1,1,0,0,0,0],
    [1,0,1,0,0,0],
    [1,1,0,1,0,0],
    [1,1,0,0,1,0],
    [1,0,1,0,0,1],
], dtype=torch.bool)
assert torch.equal(mask, ref)""",
        },
        {
            "name": 'single root edge case',
            "code": """import torch
mask = {fn}([-1])
assert torch.equal(mask, torch.ones(1, 1, dtype=torch.bool))
assert mask.sum().item() == 1""",
        },
        {
            "name": 'mask preserves downstream gradient',
            "code": """import torch
mask = {fn}(torch.tensor([-1, 0, 1, 1]))
scores = torch.randn(4, 4, requires_grad=True)
visible = scores.masked_fill(~mask, 0.0).sum()
visible.backward()
assert scores.grad is not None and scores.grad.shape == scores.shape
assert torch.equal(scores.grad != 0, mask)""",
        },
    ],
    'solution': '''def tree_speculative_mask(parent_indices):
    """实现树状推测解码注意力掩码（Medusa / EAGLE）（Tree Speculative Decoding Attention Mask）。"""
    # Step 1: 将父指针序列转为设备一致的张量并分配 N×N 可见性矩阵
    parents = torch.as_tensor(parent_indices)
    N = parents.numel()
    # Step 2: 按父先子后的拓扑顺序复制父行，使节点继承完整祖先路径
    mask = torch.zeros((N, N), dtype=torch.bool, device=parents.device)
    for i in range(N):
        if i > 0:
            mask[i] = mask[parents[i]]
        mask[i, i] = True
    # Step 3: 置位每个节点自身，形成隔离兄弟分支的祖先闭包掩码
    return mask''',
    'demo': """torch.manual_seed(42)
parents = [-1, 0, 0, 1, 1, 2]
mask = tree_speculative_mask(parents)
print("Mask shape:", mask.shape)
print("Last visibility row:", mask[-1].int())""",
    'explanation': r"""### 核心思路
树草稿中的节点只能读取其根到自身路径。所求布尔矩阵满足 $M_{ij}=1\iff j\in\operatorname{Ancestors}(i)\cup\{i\}$。在父节点编号小于子节点的前提下，父行已经是其祖先闭包，因此子行只需复制父行并置位自身。

### 代码步骤拆解
- **Step 1（分配路径矩阵）**：`parents` 含 `N` 个父指针，在同一 device 创建 `(N,N)` 的全假 `mask`。
- **Step 2（递推祖先闭包）**：节点 `i>0` 执行 `mask[i]=mask[parents[i]]`，一次复制继承父节点及其全部祖先。
- **Step 3（加入自身可见性）**：设置 `mask[i,i]=True`；例如父链 `0→1→4` 使第 4 行仅在列 `{0,1,4}` 为真。

### 重点与易错点分析
- **注意点 1（拓扑编号假设）**：算法依赖父节点先于子节点；若 `parents[i]>=i`，复制到的是尚未完成的行，必须先拓扑排序或改用显式追溯。
- **注意点 2（兄弟隔离）**：不能使用普通下三角因果 mask，因为它会让后编号节点看到所有早先兄弟；树 mask 只传播父链上的列。
- **注意点 3（mask 语义）**：这里 `True` 表示可见，传给某些接受“True 表示屏蔽”的注意力 API 时需要取反；布尔 mask 本身不求梯度，但会精确门控 score 梯度。

### 复杂度分析
- **时间复杂度**：复制 `N` 行、每行长度 `N`，为 $O(N^2)$。
- **空间复杂度**：显式祖先可见性矩阵占 $O(N^2)$；父指针本身仅占 $O(N)$。""",
}
