"""Kabsch rigid point-cloud registration task."""

TASK = {
    "title": "3D Point Cloud Rigid Registration (Kabsch SVD)",
    "title_zh": "三维点云刚体配准（Kabsch SVD 算法）",
    "difficulty": "Hard",
    "category": "三维与具身感知",
    "function_name": "kabsch_align",
    "starter_code": '''def kabsch_align(src, dst):
    pass
''',
    "hint": "Center both clouds, take the SVD of X.T @ Y, and insert the determinant sign into the final diagonal factor before recovering translation.",
    "hint_zh": "将两组点云中心化，对 X.T @ Y 做 SVD，在恢复平移前把行列式符号放入最后一个对角修正因子。",
    "description_en": "Recover the proper rigid transform that least-squares aligns paired 3D source points to destination points using the Kabsch algorithm.",
    "description_zh": "使用 Kabsch 算法恢复最小二乘意义下将配对三维源点对齐到目标点的正规刚体变换。",
    "theory_en": "SVD factorizes cross-covariance into orthogonal bases. A determinant correction excludes reflections, and centroid alignment then gives the translation.",
    "theory_zh": "SVD 将互协方差分解为正交基；行列式修正排除反射，随后由质心对齐得到平移。",
    "diagram_en": "```text\nsrc,dst -> center -> H=X^T Y -> SVD -> reflection correction -> R,t -> aligned\n```",
    "diagram_zh": "```text\n源点、目标点 -> 中心化 -> H=X^T Y -> SVD -> 反射修正 -> R,t -> 对齐点\n```",
    "tests": [
        {"name": "return shapes", "code": '''import torch
src = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
dst = src + torch.tensor([1., -2., 3.])
R, t, aligned = {fn}(src, dst)
assert R.shape == (3, 3) and t.shape == (3,) and aligned.shape == src.shape
'''},
        {"name": "exact rigid transform", "code": '''import torch
src = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
R0 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
t0 = torch.tensor([2., -1., .5])
dst = src @ R0.T + t0
R, t, aligned = {fn}(src, dst)
assert torch.allclose(R, R0, atol=1e-5) and torch.allclose(t, t0, atol=1e-5) and torch.allclose(aligned, dst, atol=1e-5)
'''},
        {"name": "translation-only edge case", "code": '''import torch
src = torch.tensor([[0., 0., 0.], [1., 2., 0.], [-1., 0., 2.], [2., -1., 1.]])
shift = torch.tensor([3., -2., 4.])
R, t, aligned = {fn}(src, src + shift)
assert torch.allclose(R, torch.eye(3), atol=1e-5) and torch.allclose(t, shift, atol=1e-5) and torch.allclose(aligned, src + shift, atol=1e-5)
'''},
        {"name": "gradient flow", "code": '''import torch
torch.manual_seed(4)
src = torch.randn(7, 3, requires_grad=True)
dst = torch.randn(7, 3, requires_grad=True)
R, t, aligned = {fn}(src, dst)
(aligned - dst).square().sum().backward()
assert src.grad is not None and dst.grad is not None and torch.isfinite(src.grad).all() and torch.isfinite(dst.grad).all()
'''},
    ],
    "solution": '''import torch

def kabsch_align(src, dst):
    c_src = src.mean(dim=0)
    c_dst = dst.mean(dim=0)
    X = src - c_src
    Y = dst - c_dst
    H = X.T @ Y
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T
    d = torch.linalg.det(V @ U.T)
    D = torch.diag(torch.stack((src.new_tensor(1.0), src.new_tensor(1.0), torch.sign(d))))
    R = V @ D @ U.T
    t = c_dst - R @ c_src
    aligned = (src @ R.T) + t.unsqueeze(0)
    return R, t, aligned
''',
    "demo": '''source = torch.randn(100, 3)
destination = source + torch.tensor([1., 2., 3.])
R, t, aligned = kabsch_align(source, destination)
print(R, t, (aligned - destination).norm())
''',
    "explanation": "The final singular-vector direction is flipped only when needed, forcing det(R)=+1. Translation is computed after rotation from the two centroids.",
}
