"""VQ-VAE Vector Quantizer with Straight-Through Estimator task."""

TASK = {
    'title': 'VQ-VAE Vector Quantizer with Straight-Through Estimator',
    'title_zh': 'VQ-VAE 向量量化器与直通估计器（STE）',
    'difficulty': 'Hard',
    'category': '扩散与流模型',
    'function_name': 'vqvae_quantize',
    'starter_code': """def vqvae_quantize(z_e, codebook, commitment_beta=0.25):
    pass""",
    'hint': """Expand squared norms and use a matrix product for all distances. Index the codebook, form the two stop-gradient losses, then use z_e + (z_q-z_e).detach().""",
    'hint_zh': '展开平方范数并用矩阵乘法计算所有距离。索引码本，构造两个停止梯度的损失，再使用 z_e + (z_q-z_e).detach()。',
    'description_en': """Quantize encoder vectors z_e of shape (B,N,D) against a codebook (K,D), using nearest-neighbor lookup and the VQ-VAE straight-through estimator. Return (z_q_ste,indices,vq_loss).""",
    'description_zh': '将形状 (B,N,D) 的编码器向量通过最近邻查找量化到码本 (K,D)，并使用 VQ-VAE 直通估计器。返回 (z_q_ste,indices,vq_loss)。',
    'theory_en': """Nearest-neighbor selection is discrete. The STE uses quantized values in the forward pass but identity gradients to the encoder. Separate codebook and commitment terms train embeddings and constrain encoder outputs.""",
    'theory_zh': '最近邻选择是离散操作。STE 前向使用量化值，反向向编码器传递恒等梯度。码本损失与承诺损失分别训练嵌入并约束编码器输出。',
    'diagram_en': r"""```text
z_e -> distances to codebook -> argmin -> z_q -> STE -> z_q_ste
                    \-> codebook loss + commitment loss
```""",
    'diagram_zh': r"""```text
z_e -> 到码本的距离 -> argmin -> z_q -> STE -> z_q_ste
                    \-> 码本损失 + 承诺损失
```""",
    "tests": [
        {
            "name": 'shapes and nearest indices',
            "code": """import torch
z=torch.tensor([[[.1,.2],[2.9,3.1]]]); cb=torch.tensor([[0.,0.],[3.,3.],[-2.,1.]])
q,idx,loss={fn}(z,cb)
assert q.shape==z.shape and idx.shape==(1,2) and loss.shape==torch.Size([])
assert torch.equal(idx,torch.tensor([[0,1]])) and torch.equal(q,cb[idx])""",
        },
        {
            "name": 'exact distance and loss reference',
            "code": """import torch
torch.manual_seed(4)
z=torch.randn(2,3,4); cb=torch.randn(5,4)
q,idx,loss={fn}(z,cb,.7)
d=z.square().sum(-1,keepdim=True)+cb.square().sum(-1)-2*(z@cb.t())
ri=d.argmin(-1); raw=cb[ri]
ref=((raw-z.detach())**2).mean()+.7*((z-raw.detach())**2).mean()
assert torch.equal(idx,ri) and torch.allclose(q,raw,atol=1e-7) and torch.equal(loss,ref)""",
        },
        {
            "name": 'exact-code edge case',
            "code": """import torch
cb=torch.tensor([[1.,2.],[-1.,0.]])
z=cb.reshape(1,2,2).clone()
q,idx,loss={fn}(z,cb)
assert torch.equal(q,z) and torch.equal(idx,torch.tensor([[0,1]]))
assert loss.item()==0.0""",
        },
        {
            "name": 'STE and loss gradient flow',
            "code": """import torch
z=torch.randn(2,3,4,requires_grad=True); cb=torch.randn(6,4,requires_grad=True)
q,idx,loss={fn}(z,cb,.25)
(q.sum()+loss).backward()
assert z.grad is not None and cb.grad is not None and idx.dtype==torch.long
assert torch.isfinite(z.grad).all() and torch.isfinite(cb.grad).all()""",
        },
    ],
    'solution': '''def vqvae_quantize(z_e, codebook, commitment_beta=0.25):
    """实现VQ-VAE 向量量化器与直通估计器（STE）（VQ-VAE Vector Quantizer with Straight-Through Estimator）。"""
    # Step 1: 展开平方欧氏距离并为每个编码向量检索最近码本索引
    dists = z_e.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=-1) - 2.0 * (z_e @ codebook.transpose(0, 1))
    indices = torch.argmin(dists, dim=-1)
    z_q = codebook[indices]
    codebook_loss = ((z_q - z_e.detach()) ** 2).mean()
    # Step 2: 分别阻断编码器与码本梯度，构造 codebook 和 commitment 两项损失
    commitment_loss = ((z_e - z_q.detach()) ** 2).mean()
    vq_loss = codebook_loss + commitment_beta * commitment_loss
    z_q_ste = z_e + (z_q - z_e).detach()
    # Step 3: 前向替换为最近码向量、反向对 z_e 使用恒等梯度并返回离散索引
    return (z_q_ste, indices, vq_loss)''',
    'demo': """torch.manual_seed(42)
z_e = torch.randn(2, 8, 16, requires_grad=True)
codebook = torch.randn(32, 16, requires_grad=True)
z_q, indices, loss = vqvae_quantize(z_e, codebook)
print("Quantized / index shapes:", z_q.shape, indices.shape)
print("VQ loss:", loss.item())""",
    'explanation': r"""### 核心思路
对每个编码器向量选择 $k^*=\arg\min_k\|z_e-e_k\|_2^2$，前向使用离散码 $z_q=e_{k^*}$。由于 argmin 不可导，`z_e+(z_q-z_e).detach()` 在数值上等于 $z_q$，但对 $z_e$ 的导数为恒等映射；码本项与 commitment 项再分别训练两侧。

### 代码步骤拆解
- **Step 1（批量最近邻检索）**：利用 $\|z\|^2+\|e\|^2-2z^\top e$ 计算 `(B,N,K)` 距离，`argmin` 得 `(B,N)` 的 `indices`，索引 `(K,D)` 码本得到 `z_q`。
- **Step 2（双向停止梯度损失）**：`(z_q-z_e.detach())²` 只更新码本，`(z_e-z_q.detach())²` 只约束编码器靠近已选码，并乘 `commitment_beta`。
- **Step 3（直通量化输出）**：`z_q_ste` 前向与 `z_q` 完全相同，主重建损失反向时则绕过离散选择直接传给 `z_e`。

### 重点与易错点分析
- **注意点 1（detach 方向）**：两项损失的 detach 方向不能交换；否则码本不会向编码分布移动，或编码器缺少 commitment 约束。
- **注意点 2（STE 不训练码本）**：`z_q_ste` 中差值整体 detach，因此主任务梯度不会经该路径进入 codebook；码本梯度来自 `codebook_loss`。
- **注意点 3（距离矩阵与码本塌缩）**：不显式扩展 `(B,N,K,D)` 可节省一维内存，但 `(B,N,K)` 仍可能很大；未被选中的码本项没有梯度，实践中常用 EMA 或重启死码。

### 复杂度分析
- **时间复杂度**：矩阵乘法计算全部距离为 $O(BNKD)$，最近邻归约为 $O(BNK)$。
- **空间复杂度**：距离矩阵占 $O(BNK)$，量化输出占 $O(BND)$；避免了 $O(BNKD)$ 的显式差值张量。""",
}
