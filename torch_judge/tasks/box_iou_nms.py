"""2D Bounding Box IoU & Non-Maximum Suppression (NMS) task."""

TASK = {
    'title': '2D Bounding Box IoU & Non-Maximum Suppression (NMS)',
    'title_zh': '二维边界框 IoU 与非极大值抑制（NMS）',
    'difficulty': 'Medium',
    'category': '三维与具身感知',
    'function_name': 'box_iou_and_nms',
    'starter_code': """def box_iou_and_nms(boxes, scores, iou_threshold=0.5):
    pass""",
    'hint': """Broadcast box coordinates to construct all pairwise intersections, then repeatedly keep the first score-sorted index and filter the remainder by its IoU row.""",
    'hint_zh': '广播框坐标以构造全部成对交集，然后反复保留按分数排序后的首个索引，并用其 IoU 行过滤剩余索引。',
    'description_en': """Compute a full pairwise IoU matrix for axis-aligned boxes and perform greedy score-ordered non-maximum suppression.""",
    'description_zh': '计算轴对齐边界框的完整成对 IoU 矩阵，并执行按置信度排序的贪心非极大值抑制。',
    'theory_en': """IoU divides intersection area by union area. NMS keeps each highest-scoring remaining box and suppresses lower-scoring boxes whose overlap is strictly above the threshold.""",
    'theory_zh': 'IoU 为交集面积除以并集面积。NMS 保留当前剩余框中分数最高者，并抑制与其重叠严格超过阈值的低分框。',
    'diagram_en': """```text
boxes -> pairwise intersections/unions -> IoU matrix -> score sort -> greedy suppression
```""",
    'diagram_zh': """```text
边界框 -> 成对交集/并集 -> IoU 矩阵 -> 分数排序 -> 贪心抑制
```""",
    "tests": [
        {
            "name": 'matrix and indices shapes',
            "code": """import torch
boxes = torch.tensor([[0., 0., 1., 1.], [2., 2., 3., 3.]])
iou, keep = {fn}(boxes, torch.tensor([.2, .8]))
assert iou.shape == (2, 2) and keep.shape == (2,) and keep.dtype == torch.long""",
        },
        {
            "name": 'exact IoU and suppression',
            "code": """import torch
boxes = torch.tensor([[0., 0., 2., 2.], [1., 1., 3., 3.], [0., 0., 2., 2.]])
iou, keep = {fn}(boxes, torch.tensor([.9, .8, .7]), iou_threshold=.5)
ref = torch.tensor([[1., 1/7, 1.], [1/7, 1., 1/7], [1., 1/7, 1.]])
assert torch.allclose(iou, ref, atol=1e-7) and torch.equal(keep, torch.tensor([0, 1]))""",
        },
        {
            "name": 'degenerate box edge case',
            "code": """import torch
boxes = torch.tensor([[1., 1., 1., 2.]])
iou, keep = {fn}(boxes, torch.tensor([1.]))
assert torch.equal(iou, torch.zeros(1, 1)) and torch.equal(keep, torch.tensor([0]))""",
        },
        {
            "name": 'IoU gradient flow',
            "code": """import torch
boxes = torch.tensor([[0., 0., 2., 2.], [.5, .5, 3., 3.]], requires_grad=True)
iou, keep = {fn}(boxes, torch.tensor([1., .5]))
iou.sum().backward()
assert boxes.grad is not None and torch.isfinite(boxes.grad).all() and keep.ndim == 1""",
        },
    ],
    'solution': '''def box_iou_and_nms(boxes, scores, iou_threshold=0.5):
    """边界框交并比与非极大值抑制 (IoU and NMS)。"""
    # Step 1: 广播计算全部边界框的成对 IoU
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    areas = widths * heights
    inter_x1 = torch.maximum(boxes[:, None, 0], boxes[None, :, 0])
    inter_y1 = torch.maximum(boxes[:, None, 1], boxes[None, :, 1])
    inter_x2 = torch.minimum(boxes[:, None, 2], boxes[None, :, 2])
    inter_y2 = torch.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (
        inter_y2 - inter_y1
    ).clamp(min=0)
    union = areas[:, None] + areas[None, :] - inter_area
    iou_matrix = inter_area / (union + 1e-8)

    # Step 2: 按分数降序执行贪心抑制
    remaining = torch.argsort(scores, descending=True)
    kept = []
    while remaining.numel() > 0:
        current = remaining[0]
        kept.append(current)
        if remaining.numel() == 1:
            break
        candidates = remaining[1:]
        remaining = candidates[iou_matrix[current, candidates] <= iou_threshold]

    # Step 3: 汇总保留索引
    kept_indices = torch.stack(kept) if kept else torch.empty(
        0, device=boxes.device, dtype=torch.long
    )
    return iou_matrix, kept_indices''',
    'demo': """torch.manual_seed(42)
boxes = torch.tensor([[0., 0., 2., 2.], [.2, .2, 2.2, 2.2], [4., 4., 5., 5.]])
scores = torch.tensor([0.9, 0.8, 0.7])
iou, kept = box_iou_and_nms(boxes, scores)
print("Output shape:", iou.shape)
print("Kept indices:", kept)
print("Sample values:", iou[0])""",
    'explanation': """### 核心思路
IoU 用交集面积除以并集面积；NMS 每次保留最高分框，并删除与它的 IoU 严格超过阈值的剩余框。

### 代码步骤拆解
- **Step 1（成对 IoU）**：利用新增维度广播，构造 `(N, N)` 的交集、并集与 `iou_matrix`。
- **Step 2（贪心抑制）**：按 `scores` 降序迭代，保留当前框并筛选后续候选。
- **Step 3（索引输出）**：将标量索引堆叠为一维 long 张量。

### 重点与易错点分析
- **注意点 1（退化框）**：宽高先截断到非负，并集分母加入 `1e-8`，避免零面积产生 NaN。
- **注意点 2（阈值语义）**：只抑制 `IoU > threshold`，因此等于阈值的框保留。
- **注意点 3（排序稳定性）**：NMS 的选择顺序由置信度排序决定，不能按输入顺序遍历。

### 复杂度分析
- **时间复杂度**：$O(N^2)$，成对 IoU 与最坏情况下的贪心筛选均为平方级。
- **空间复杂度**：$O(N^2)$，主要保存完整 IoU 矩阵。""",
}
