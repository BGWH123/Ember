"""Depth Map 3D Unprojection & Farthest Point Sampling (FPS) task."""

TASK = {
    'title': 'Depth Map 3D Unprojection & Farthest Point Sampling (FPS)',
    'title_zh': '相机深度图三维反投影与最远点采样（FPS）',
    'difficulty': 'Medium',
    'category': '三维与具身感知',
    'function_name': 'depth_unproject_and_fps',
    'starter_code': """def depth_unproject_and_fps(depth, fx, fy, cx, cy, num_samples):
    pass""",
    'hint': """Use an ij-indexed row/column meshgrid, flatten the stacked XYZ coordinates in row-major order, then repeatedly update each point's minimum squared distance to the selected set.""",
    'hint_zh': '使用 ij 索引的行列网格，将堆叠的 XYZ 坐标按行优先展平，再反复更新每个点到已选集合的最小平方距离。',
    'description_en': """Unproject an (H,W) pinhole-camera depth map into an (H*W,3) point cloud, then select num_samples point indices by deterministic iterative farthest point sampling starting at index zero.""",
    'description_zh': '将 (H,W) 针孔相机深度图反投影为 (H*W,3) 点云，再从索引零开始用确定性的迭代最远点采样选择 num_samples 个点索引。',
    'theory_en': """Pinhole unprojection scales normalized image-plane rays by depth. FPS greedily maximizes coverage by choosing the point whose distance to its nearest selected point is largest.""",
    'theory_zh': '针孔反投影用深度缩放归一化像平面射线。FPS 每次选择到最近已选点距离最大的点，从而贪心地扩大覆盖范围。',
    'diagram_en': """```text
depth + intrinsics -> pixel rays -> XYZ points
index 0 -> update nearest distances -> argmax -> repeat
```""",
    'diagram_zh': """```text
深度 + 内参 -> 像素射线 -> XYZ 点
索引 0 -> 更新最近距离 -> argmax -> 重复
```""",
    "tests": [
        {
            "name": 'output shapes and start index',
            "code": """import torch
d=torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.]])
p,idx={fn}(d,2.,2.,1.5,1.,5)
assert p.shape==(12,3) and idx.shape==(5,)
assert idx[0].item()==0""",
        },
        {
            "name": 'exact pinhole unprojection',
            "code": """import torch
d=torch.tensor([[1.,2.],[3.,4.]])
p,idx={fn}(d,2.,4.,.5,1.,2)
u=torch.tensor([[0.,1.],[0.,1.]]); v=torch.tensor([[0.,0.],[1.,1.]])
ref=torch.stack(((u-.5)*d/2.,(v-1.)*d/4.,d),-1).reshape(-1,3)
assert torch.equal(p,ref)
assert idx[0].item()==0""",
        },
        {
            "name": 'deterministic FPS edge',
            "code": """import torch
d=torch.ones(1,4)
p,idx={fn}(d,1.,1.,0.,0.,3)
assert torch.equal(p,torch.tensor([[0.,0.,1.],[1.,0.,1.],[2.,0.,1.],[3.,0.,1.]]))
assert torch.equal(idx,torch.tensor([0,3,1]))
p1,i1={fn}(d,1.,1.,0.,0.,1)
assert torch.equal(i1,torch.tensor([0]))""",
        },
        {
            "name": 'gradient flow through unprojection',
            "code": """import torch
d=torch.rand(3,3,requires_grad=True)+.5; d.retain_grad()
p,idx={fn}(d,2.,3.,1.,1.,4); p.square().sum().backward()
assert d.grad is not None and torch.isfinite(d.grad).all()
assert idx.requires_grad is False""",
        },
    ],
    'solution': '''def depth_unproject_and_fps(depth, fx, fy, cx, cy, num_samples):
    """相机深度图三维反投影与最远点采样（FPS） (Depth Map 3D Unprojection & Farthest Point Sampling (FPS))。"""
    # Step 1: 用针孔相机内参将每个像素 (u,v,Z) 反投影为三维点 (X,Y,Z)
    height, width = depth.shape
    row_coords = torch.arange(height, device=depth.device, dtype=depth.dtype)
    col_coords = torch.arange(width, device=depth.device, dtype=depth.dtype)
    grid_rows, grid_cols = torch.meshgrid(row_coords, col_coords, indexing='ij')
    x_coords = (grid_cols - cx) * depth / fx
    y_coords = (grid_rows - cy) * depth / fy
    points = torch.stack((x_coords, y_coords, depth), dim=-1).reshape(-1, 3)
    point_count = points.size(0)
    sampled_indices = torch.empty(num_samples, dtype=torch.long, device=depth.device)
    min_dists = torch.full((point_count,), float('inf'), dtype=depth.dtype, device=depth.device)
    selected_index = torch.tensor(0, dtype=torch.long, device=depth.device)
    # Step 2: 每轮更新所有点到已选集合的最小平方距离
    for sample_index in range(num_samples):
        sampled_indices[sample_index] = selected_index
        dist_sq = ((points - points[selected_index]) ** 2).sum(dim=-1)
        min_dists = torch.minimum(min_dists, dist_sq)
        # Step 3: 贪心选择最近距离最大的点作为下一个覆盖中心
        selected_index = torch.argmax(min_dists)
    return (points, sampled_indices)''',
    'demo': """torch.manual_seed(42)
depth = torch.linspace(1.0, 2.0, 12).reshape(3, 4)
points, indices = depth_unproject_and_fps(depth, 2.0, 2.0, 1.5, 1.0, 5)
print('Point cloud / samples:', points.shape, indices.shape)
print('First unprojected point:', points[0])
print('FPS indices:', indices.tolist())""",
    'explanation': r"""### 核心思路
针孔模型把像素 $(u,v)$ 与深度 $Z$ 还原为 $X=(u-c_x)Z/f_x,\ Y=(v-c_y)Z/f_y$，从而得到按图像行优先排列的 $N=HW$ 个三维点。FPS 从索引 0 开始，维护每个点到当前已选集合的最近平方距离 $d_i=\min_{j\in S}\lVert p_i-p_j\rVert_2^2$，每轮选择 $\arg\max_i d_i$，贪心扩大点集覆盖范围。

### 代码步骤拆解
- **Step 1（深度反投影）**：`meshgrid(..., indexing='ij')` 生成行坐标 `grid_rows=v` 与列坐标 `grid_cols=u`，结合 `depth` 计算 `x_coords/y_coords`，再展平为 `(H*W,3)`。
- **Step 2（增量最近距离）**：第 `sample_index` 轮计算所有 `points` 到 `points[selected_index]` 的平方距离，并以 `torch.minimum` 更新 `min_dists`，无需保存完整两两距离矩阵。
- **Step 3（最远点选择）**：`torch.argmax(min_dists)` 选出离已选集合最近距离仍最大的点；首点固定为 0，因此相同输入的采样序列确定。

### 重点与易错点分析
- **注意点 1（u/v 不可对调）**：列坐标 $u$ 使用 `fx,cx`，行坐标 $v$ 使用 `fy,cy`；`indexing='ij'` 保证展平次序与 `depth.reshape(-1)` 一致。
- **注意点 2（平方距离足够）**：FPS 只比较大小，平方根单调且可省略；维护“到集合的最小距离”而非对已选点距离的最大值才符合覆盖定义。
- **注意点 3（梯度边界）**：反投影坐标对 `depth` 可微；`argmax` 得到的离散索引不可微，且采样选择在距离并列处不连续。固定首点也会影响最终覆盖但保证可复现。

### 复杂度分析
- **时间复杂度**：反投影为 $O(HW)$；设 $K=$ `num_samples`，FPS 每轮扫描 $N=HW$ 个点并计算三维距离，总计 $O(HWK)$。
- **空间复杂度**：点云和最近距离向量占 $O(HW)$，采样索引占 $O(K)$；没有构造 $O((HW)^2)$ 的距离矩阵。""",
}
