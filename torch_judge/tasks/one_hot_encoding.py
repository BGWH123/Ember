"""One-Hot Encoding task."""

TASK = {
    "title": "One-Hot Encoding",
    "title_zh": "One-Hot 编码",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "description_en": "Implement one-hot encoding. Given a tensor of class indices (0 to num_classes-1), return a tensor where each row is a one-hot vector. For example, indices [0, 2, 1] with num_classes=4 → [[1,0,0,0], [0,0,1,0], [0,1,0,0]].",
    "description_zh": "实现 One-Hot 编码。给定类别索引张量（0 到 num_classes-1），返回 one-hot 向量张量。例如 indices [0, 2, 1] 且 num_classes=4 → [[1,0,0,0], [0,0,1,0], [0,1,0,0]]。",
    "function_name": "one_hot_encoding",
    "hint": "Use `torch.zeros` to create the output tensor, then use advanced indexing or `scatter_` to set 1s.",
    "hint_zh": "使用 `torch.zeros` 创建输出张量，然后用高级索引或 `scatter_` 设置1。",
    "theory_en": "One-hot encoding converts categorical class indices into binary vectors. It is the standard input format for classification tasks and the target format for cross-entropy loss.",
    "theory_zh": "One-Hot 编码将类别索引转换为二元向量。是分类任务的标准输入格式，也是交叉熵损失的目标格式。",
    "tests": [
    {
        "name": "Class positions including repeated indices",
        "code": "idx = torch.tensor([0, 3, 1, 3, 0])\nout = {fn}(idx, 4)\nassert out.shape == (5, 4)\nassert torch.equal(out, F.one_hot(idx, 4).float())"
    },
    {
        "name": "Single class and empty input",
        "code": "assert torch.equal({fn}(torch.zeros(3, dtype=torch.long), 1), torch.ones(3, 1))\nassert {fn}(torch.tensor([], dtype=torch.long), 5).shape == (0, 5)"
    },
    {
        "name": "Does not change indices",
        "code": "idx = torch.tensor([2, 1, 0])\nbefore = idx.clone()\nout = {fn}(idx, 5)\nassert torch.equal(idx, before)\nassert torch.equal(out.sum(-1), torch.ones(3))\nassert torch.all((out == 0) | (out == 1))"
    }
],
    "solution": '''

import torch

def one_hot_encoding(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    One-hot encode class indices.
    indices: (N,) with values in [0, num_classes-1]
    Returns: (N, num_classes) one-hot tensor (float32).
    """
    # 创建全零输出张量
    # Create zero output tensor
    one_hot = torch.zeros(indices.shape[0], num_classes, dtype=torch.float32)

    # 使用 scatter 在对应位置填1
    # Use scatter to fill 1s at corresponding positions
    one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

    return one_hot

    
    ''',
    "demo": "",
}
