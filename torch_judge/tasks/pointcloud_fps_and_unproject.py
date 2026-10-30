"""Depth unprojection and farthest point sampling task."""

TASK = {
    "title": "Depth Map 3D Unprojection & Farthest Point Sampling (FPS)",
    "title_zh": "相机深度图三维反投影与最远点采样（FPS）",
    "difficulty": "Medium",
    "category": "三维与具身感知",
    "function_name": "depth_unproject_and_fps",
    "starter_code": '''def depth_unproject_and_fps(depth, fx, fy, cx, cy, num_samples):
    pass
''',
    "hint": "Use an ij-indexed row/column meshgrid, flatten the stacked XYZ coordinates in row-major order, then repeatedly update each point's minimum squared distance to the selected set.",
    "hint_zh": "使用 ij 索引的行列网格，将堆叠的 XYZ 坐标按行优先展平，再反复更新每个点到已选集合的最小平方距离。",
    "description_en": "Unproject an (H,W) pinhole-camera depth map into an (H*W,3) point cloud, then select num_samples point indices by deterministic iterative farthest point sampling starting at index zero.",
    "description_zh": "将 (H,W) 针孔相机深度图反投影为 (H*W,3) 点云，再从索引零开始用确定性的迭代最远点采样选择 num_samples 个点索引。",
    "theory_en": "Pinhole unprojection scales normalized image-plane rays by depth. FPS greedily maximizes coverage by choosing the point whose distance to its nearest selected point is largest.",
    "theory_zh": "针孔反投影用深度缩放归一化像平面射线。FPS 每次选择到最近已选点距离最大的点，从而贪心地扩大覆盖范围。",
    "diagram_en": "```text\ndepth + intrinsics -> pixel rays -> XYZ points\nindex 0 -> update nearest distances -> argmax -> repeat\n```",
    "diagram_zh": "```text\n深度 + 内参 -> 像素射线 -> XYZ 点\n索引 0 -> 更新最近距离 -> argmax -> 重复\n```",
    "tests": [
        {
            "name": "output shapes and start index",
            "code": '''import torch
d=torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.]])
p,idx={fn}(d,2.,2.,1.5,1.,5)
assert p.shape==(12,3) and idx.shape==(5,)
assert idx[0].item()==0
''',
        },
        {
            "name": "exact pinhole unprojection",
            "code": '''import torch
d=torch.tensor([[1.,2.],[3.,4.]])
p,idx={fn}(d,2.,4.,.5,1.,2)
u=torch.tensor([[0.,1.],[0.,1.]]); v=torch.tensor([[0.,0.],[1.,1.]])
ref=torch.stack(((u-.5)*d/2.,(v-1.)*d/4.,d),-1).reshape(-1,3)
assert torch.equal(p,ref)
assert idx[0].item()==0
''',
        },
        {
            "name": "deterministic FPS edge",
            "code": '''import torch
d=torch.ones(1,4)
p,idx={fn}(d,1.,1.,0.,0.,3)
assert torch.equal(p,torch.tensor([[0.,0.,1.],[1.,0.,1.],[2.,0.,1.],[3.,0.,1.]]))
assert torch.equal(idx,torch.tensor([0,3,1]))
p1,i1={fn}(d,1.,1.,0.,0.,1)
assert torch.equal(i1,torch.tensor([0]))
''',
        },
        {
            "name": "gradient flow through unprojection",
            "code": '''import torch
d=torch.rand(3,3,requires_grad=True)+.5; d.retain_grad()
p,idx={fn}(d,2.,3.,1.,1.,4); p.square().sum().backward()
assert d.grad is not None and torch.isfinite(d.grad).all()
assert idx.requires_grad is False
''',
        },
    ],
    "solution": '''import torch

def depth_unproject_and_fps(depth, fx, fy, cx, cy, num_samples):
    height, width = depth.shape
    v = torch.arange(height, device=depth.device, dtype=depth.dtype)
    u = torch.arange(width, device=depth.device, dtype=depth.dtype)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    x = (grid_u - cx) * depth / fx
    y = (grid_v - cy) * depth / fy
    points = torch.stack((x, y, depth), dim=-1).reshape(-1, 3)
    point_count = points.size(0)
    sampled_indices = torch.empty(
        num_samples, dtype=torch.long, device=depth.device
    )
    min_dists = torch.full(
        (point_count,), float('inf'), dtype=depth.dtype, device=depth.device
    )
    idx = torch.tensor(0, dtype=torch.long, device=depth.device)
    for sample_index in range(num_samples):
        sampled_indices[sample_index] = idx
        dist_sq = ((points - points[idx]) ** 2).sum(dim=-1)
        min_dists = torch.minimum(min_dists, dist_sq)
        idx = torch.argmax(min_dists)
    return points, sampled_indices
''',
    "demo": '''depth = torch.ones(24, 32)
points, indices = depth_unproject_and_fps(depth, 30.0, 30.0, 15.5, 11.5, 64)
print(points.shape, indices.shape)
''',
    "explanation": "The meshgrid and reshape preserve row-major pixel order. Squared distances avoid unnecessary square roots because argmax ordering is unchanged.",
}
