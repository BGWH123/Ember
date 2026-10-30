"""Tree speculative-decoding attention-mask task."""

TASK = {
    "title": "Tree Speculative Decoding Attention Mask",
    "title_zh": "树状推测解码注意力掩码（Medusa / EAGLE）",
    "difficulty": "Medium",
    "category": "推理优化",
    "function_name": "tree_speculative_mask",
    "starter_code": '''def tree_speculative_mask(parent_indices):
    pass
''',
    "hint": "Process nodes in index order. Copy the parent's already-computed ancestor row, then mark the node's own diagonal entry true.",
    "hint_zh": "按索引顺序处理节点：复制父节点已计算好的祖先行，再把当前节点自身的对角位置设为 True。",
    "description_en": "Construct the ancestor visibility mask for a parent-before-child speculative decoding tree.",
    "description_zh": "为父节点先于子节点排列的推测解码树构造祖先可见性掩码。",
    "theory_en": "A draft-tree node may attend only to tokens on its own root-to-node path; sibling branches must remain isolated.",
    "theory_zh": "草稿树节点只能关注自身从根到该节点的路径，兄弟分支必须相互隔离。",
    "diagram_en": "```text\n0\n+- 1 +- 3\n|    \\- 4\n\\- 2\nrow(4) sees {0,1,4}\n```",
    "diagram_zh": "```text\n0\n+- 1 +- 3\n|    \\- 4\n\\- 2\n第 4 行可见 {0,1,4}\n```",
    "tests": [
        {
            "name": "shape dtype and diagonal",
            "code": '''import torch
mask = {fn}(torch.tensor([-1, 0, 0, 1, 1, 2]))
assert mask.shape == (6, 6) and mask.dtype == torch.bool
assert torch.equal(mask.diagonal(), torch.ones(6, dtype=torch.bool))
''',
        },
        {
            "name": "exact branching tree",
            "code": '''import torch
mask = {fn}([-1, 0, 0, 1, 1, 2])
ref = torch.tensor([
    [1,0,0,0,0,0],
    [1,1,0,0,0,0],
    [1,0,1,0,0,0],
    [1,1,0,1,0,0],
    [1,1,0,0,1,0],
    [1,0,1,0,0,1],
], dtype=torch.bool)
assert torch.equal(mask, ref)
''',
        },
        {
            "name": "single root edge case",
            "code": '''import torch
mask = {fn}([-1])
assert torch.equal(mask, torch.ones(1, 1, dtype=torch.bool))
assert mask.sum().item() == 1
''',
        },
        {
            "name": "mask preserves downstream gradient",
            "code": '''import torch
mask = {fn}(torch.tensor([-1, 0, 1, 1]))
scores = torch.randn(4, 4, requires_grad=True)
visible = scores.masked_fill(~mask, 0.0).sum()
visible.backward()
assert scores.grad is not None and scores.grad.shape == scores.shape
assert torch.equal(scores.grad != 0, mask)
''',
        },
    ],
    "solution": '''import torch

def tree_speculative_mask(parent_indices):
    parents = torch.as_tensor(parent_indices)
    N = parents.numel()
    mask = torch.zeros((N, N), dtype=torch.bool, device=parents.device)
    for i in range(N):
        if i > 0:
            mask[i] = mask[parents[i]]
        mask[i, i] = True
    return mask
''',
    "demo": '''parents = [-1, 0, 0, 1, 1, 2]
print(tree_speculative_mask(parents).int())
''',
    "explanation": "Because every parent index is smaller than its child index, the parent's complete ancestor row is available when the child is processed.",
}
