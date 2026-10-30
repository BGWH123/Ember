"""3D Point Cloud Rigid Registration (Kabsch SVD) task."""

TASK = {
    'title': '3D Point Cloud Rigid Registration (Kabsch SVD)',
    'title_zh': '三维点云刚体配准（Kabsch SVD 算法）',
    'difficulty': 'Hard',
    'category': '三维与具身感知',
    'function_name': 'kabsch_align',
    'starter_code': """def kabsch_align(src, dst):
    pass""",
    'hint': """Center both clouds, take the SVD of X.T @ Y, and insert the determinant sign into the final diagonal factor before recovering translation.""",
    'hint_zh': '将两组点云中心化，对 X.T @ Y 做 SVD，在恢复平移前把行列式符号放入最后一个对角修正因子。',
    'description_en': """Recover the proper rigid transform that least-squares aligns paired 3D source points to destination points using the Kabsch algorithm.""",
    'description_zh': '使用 Kabsch 算法恢复最小二乘意义下将配对三维源点对齐到目标点的正规刚体变换。',
    'theory_en': """SVD factorizes cross-covariance into orthogonal bases. A determinant correction excludes reflections, and centroid alignment then gives the translation.""",
    'theory_zh': 'SVD 将互协方差分解为正交基；行列式修正排除反射，随后由质心对齐得到平移。',
    'diagram_en': """```text
src,dst -> center -> H=X^T Y -> SVD -> reflection correction -> R,t -> aligned
```""",
    'diagram_zh': """```text
源点、目标点 -> 中心化 -> H=X^T Y -> SVD -> 反射修正 -> R,t -> 对齐点
```""",
    "tests": [
        {
            "name": 'return shapes',
            "code": """import torch
src = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
dst = src + torch.tensor([1., -2., 3.])
R, t, aligned = {fn}(src, dst)
assert R.shape == (3, 3) and t.shape == (3,) and aligned.shape == src.shape""",
        },
        {
            "name": 'exact rigid transform',
            "code": """import torch
src = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
R0 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
t0 = torch.tensor([2., -1., .5])
dst = src @ R0.T + t0
R, t, aligned = {fn}(src, dst)
assert torch.allclose(R, R0, atol=1e-5) and torch.allclose(t, t0, atol=1e-5) and torch.allclose(aligned, dst, atol=1e-5)""",
        },
        {
            "name": 'translation-only edge case',
            "code": """import torch
src = torch.tensor([[0., 0., 0.], [1., 2., 0.], [-1., 0., 2.], [2., -1., 1.]])
shift = torch.tensor([3., -2., 4.])
R, t, aligned = {fn}(src, src + shift)
assert torch.allclose(R, torch.eye(3), atol=1e-5) and torch.allclose(t, shift, atol=1e-5) and torch.allclose(aligned, src + shift, atol=1e-5)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
torch.manual_seed(4)
src = torch.randn(7, 3, requires_grad=True)
dst = torch.randn(7, 3, requires_grad=True)
R, t, aligned = {fn}(src, dst)
(aligned - dst).square().sum().backward()
assert src.grad is not None and dst.grad is not None and torch.isfinite(src.grad).all() and torch.isfinite(dst.grad).all()""",
        },
    ],
    'solution': '''def kabsch_align(src, dst):
    """使用 Kabsch 算法对齐三维点云 (Kabsch Alignment)。"""
    # Step 1: 计算质心并中心化配对点云
    src_center = src.mean(dim=0)
    dst_center = dst.mean(dim=0)
    src_centered = src - src_center
    dst_centered = dst - dst_center

    # Step 2: 分解互协方差并修正反射
    covariance = src_centered.T @ dst_centered
    U, _, Vh = torch.linalg.svd(covariance)
    V = Vh.T
    determinant = torch.linalg.det(V @ U.T)
    correction = torch.diag(torch.stack((src.new_tensor(1.0), src.new_tensor(1.0), torch.sign(determinant))))
    rotation = V @ correction @ U.T

    # Step 3: 恢复平移并变换源点云
    translation = dst_center - rotation @ src_center
    aligned = src @ rotation.T + translation
    return rotation, translation, aligned''',
    'demo': """torch.manual_seed(42)
source = torch.randn(10, 3)
shift = torch.tensor([1.0, 2.0, 3.0])
destination = source + shift
rotation, translation, aligned = kabsch_align(source, destination)
print("Output shape:", aligned.shape)
print("Translation:", translation)
print("Alignment error:", (aligned - destination).norm().item())""",
    'explanation': """### 核心思路
中心化点云后对互协方差做 SVD，通过行列式符号排除反射，得到最优正规旋转，再由两组质心恢复平移。

### 代码步骤拆解
- **Step 1（中心化）**：`src_centered` 与 `dst_centered` 均为 `(N, 3)`。
- **Step 2（旋转求解）**：对 $X^TY$ 分解并构造最后一维可能为 -1 的修正矩阵。
- **Step 3（刚体变换）**：计算 `translation`，并以行向量约定执行 `src @ rotation.T`。

### 重点与易错点分析
- **注意点 1（反射修正）**：若直接用 $VU^T$，退化数据可能得到行列式为 -1 的反射矩阵。
- **注意点 2（乘法约定）**：点按行存放，因此应用旋转时必须乘 `rotation.T`。
- **注意点 3（SVD 梯度）**：奇异值重复时 SVD 梯度可能不稳定，这是可微配准的固有限制。

### 复杂度分析
- **时间复杂度**：$O(N)$，三维 SVD 是常数规模，互协方差累积占主导。
- **空间复杂度**：$O(N)$，主要保存两组中心化点云。""",
}
