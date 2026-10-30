"""VQ-VAE vector quantization with STE task."""

TASK = {
    "title": "VQ-VAE Vector Quantizer with Straight-Through Estimator",
    "title_zh": "VQ-VAE 向量量化器与直通估计器（STE）",
    "difficulty": "Hard",
    "category": "扩散与流模型",
    "function_name": "vqvae_quantize",
    "starter_code": '''def vqvae_quantize(z_e, codebook, commitment_beta=0.25):
    pass
''',
    "hint": "Expand squared norms and use a matrix product for all distances. Index the codebook, form the two stop-gradient losses, then use z_e + (z_q-z_e).detach().",
    "hint_zh": "展开平方范数并用矩阵乘法计算所有距离。索引码本，构造两个停止梯度的损失，再使用 z_e + (z_q-z_e).detach()。",
    "description_en": "Quantize encoder vectors z_e of shape (B,N,D) against a codebook (K,D), using nearest-neighbor lookup and the VQ-VAE straight-through estimator. Return (z_q_ste,indices,vq_loss).",
    "description_zh": "将形状 (B,N,D) 的编码器向量通过最近邻查找量化到码本 (K,D)，并使用 VQ-VAE 直通估计器。返回 (z_q_ste,indices,vq_loss)。",
    "theory_en": "Nearest-neighbor selection is discrete. The STE uses quantized values in the forward pass but identity gradients to the encoder. Separate codebook and commitment terms train embeddings and constrain encoder outputs.",
    "theory_zh": "最近邻选择是离散操作。STE 前向使用量化值，反向向编码器传递恒等梯度。码本损失与承诺损失分别训练嵌入并约束编码器输出。",
    "diagram_en": "```text\nz_e -> distances to codebook -> argmin -> z_q -> STE -> z_q_ste\n                    \\-> codebook loss + commitment loss\n```",
    "diagram_zh": "```text\nz_e -> 到码本的距离 -> argmin -> z_q -> STE -> z_q_ste\n                    \\-> 码本损失 + 承诺损失\n```",
    "tests": [
        {
            "name": "shapes and nearest indices",
            "code": '''import torch
z=torch.tensor([[[.1,.2],[2.9,3.1]]]); cb=torch.tensor([[0.,0.],[3.,3.],[-2.,1.]])
q,idx,loss={fn}(z,cb)
assert q.shape==z.shape and idx.shape==(1,2) and loss.shape==torch.Size([])
assert torch.equal(idx,torch.tensor([[0,1]])) and torch.equal(q,cb[idx])
''',
        },
        {
            "name": "exact distance and loss reference",
            "code": '''import torch
torch.manual_seed(4)
z=torch.randn(2,3,4); cb=torch.randn(5,4)
q,idx,loss={fn}(z,cb,.7)
d=z.square().sum(-1,keepdim=True)+cb.square().sum(-1)-2*(z@cb.t())
ri=d.argmin(-1); raw=cb[ri]
ref=((raw-z.detach())**2).mean()+.7*((z-raw.detach())**2).mean()
assert torch.equal(idx,ri) and torch.allclose(q,raw,atol=1e-7) and torch.equal(loss,ref)
''',
        },
        {
            "name": "exact-code edge case",
            "code": '''import torch
cb=torch.tensor([[1.,2.],[-1.,0.]])
z=cb.reshape(1,2,2).clone()
q,idx,loss={fn}(z,cb)
assert torch.equal(q,z) and torch.equal(idx,torch.tensor([[0,1]]))
assert loss.item()==0.0
''',
        },
        {
            "name": "STE and loss gradient flow",
            "code": '''import torch
z=torch.randn(2,3,4,requires_grad=True); cb=torch.randn(6,4,requires_grad=True)
q,idx,loss={fn}(z,cb,.25)
(q.sum()+loss).backward()
assert z.grad is not None and cb.grad is not None and idx.dtype==torch.long
assert torch.isfinite(z.grad).all() and torch.isfinite(cb.grad).all()
''',
        },
    ],
    "solution": '''import torch

def vqvae_quantize(z_e, codebook, commitment_beta=0.25):
    dists = (
        z_e.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=-1)
        - 2.0 * (z_e @ codebook.transpose(0, 1))
    )
    indices = torch.argmin(dists, dim=-1)
    z_q = codebook[indices]
    codebook_loss = ((z_q - z_e.detach()) ** 2).mean()
    commitment_loss = ((z_e - z_q.detach()) ** 2).mean()
    vq_loss = codebook_loss + commitment_beta * commitment_loss
    z_q_ste = z_e + (z_q - z_e).detach()
    return z_q_ste, indices, vq_loss
''',
    "demo": '''z_e = torch.randn(2, 8, 16, requires_grad=True)
codebook = torch.randn(32, 16, requires_grad=True)
z_q, indices, loss = vqvae_quantize(z_e, codebook)
print(z_q.shape, indices.shape, loss.item())
''',
    "explanation": "The expanded distance formula avoids materializing (B,N,K,D). Detaches route codebook and commitment gradients to their intended parameters.",
}
