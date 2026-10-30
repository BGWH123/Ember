"""Mixture-of-Transformers (MoT) Asymmetric Cross-Modal Attention task."""

TASK = {
    'title': 'Mixture-of-Transformers (MoT) Asymmetric Cross-Modal Attention',
    'title_zh': 'Mixture-of-Transformers (MoT) 非对称跨模态注意力',
    'difficulty': 'Hard',
    'category': '多模态',
    'function_name': 'mot_asymmetric_attention',
    'starter_code': """def mot_asymmetric_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm):
    pass""",
    'hint': """Concatenate both streams, then combine a causal lower-triangular AR query block with an all-true diffusion query block. Use the boolean mask before softmax.""",
    'hint_zh': '先拼接两条流，再将 AR 查询的因果下三角块与全真的扩散查询块组合，在 softmax 前应用布尔掩码。',
    'description_en': """Compute single-head scaled dot-product attention over concatenated autoregressive and diffusion streams. AR queries are causal and cannot read diffusion tokens; diffusion queries can read every token. Return both split outputs and the boolean mask.""",
    'description_zh': '在拼接的自回归流和扩散流上计算单头缩放点积注意力。AR 查询具有因果性且不能读取扩散 token；扩散查询可读取所有 token。返回拆分输出和布尔掩码。',
    'theory_en': """The asymmetric mask protects autoregressive generation from future and diffusion information while exposing the complete reasoner context to the bidirectional diffusion generator.""",
    'theory_zh': '非对称掩码保护自回归生成不接触未来及扩散信息，同时向双向扩散生成器开放完整推理器上下文。',
    'diagram_en': """```text
keys:     AR          DM
AR q:   causal      blocked
DM q:   all visible all visible
```""",
    'diagram_zh': """```text
键：       AR          DM
AR 查询： 因果可见      禁止
DM 查询： 全部可见      全部可见
```""",
    "tests": [
        {
            "name": 'shapes and mask structure',
            "code": """import torch
a=[torch.randn(2,3,4) for _ in range(3)]; d=[torch.randn(2,5,4) for _ in range(3)]
oa,od,m={fn}(*(a+d))
ref=torch.zeros(8,8,dtype=torch.bool); ref[:3,:3]=torch.tril(torch.ones(3,3,dtype=torch.bool)); ref[3:]=True
assert oa.shape==(2,3,4) and od.shape==(2,5,4) and m.dtype==torch.bool
assert torch.equal(m.cpu(),ref)""",
        },
        {
            "name": 'exact masked attention',
            "code": """import torch
torch.manual_seed(3)
qa,ka,va=[torch.randn(1,2,3) for _ in range(3)]; qd,kd,vd=[torch.randn(1,2,3) for _ in range(3)]
oa,od,m={fn}(qa,ka,va,qd,kd,vd)
Q=torch.cat((qa,qd),1); K=torch.cat((ka,kd),1); V=torch.cat((va,vd),1)
s=(Q@K.transpose(-2,-1))/(3**.5); ref=torch.softmax(s.masked_fill(~m,float('-inf')),-1)@V
assert torch.equal(oa,ref[:,:2]) and torch.equal(od,ref[:,2:])""",
        },
        {
            "name": 'AR output ignores diffusion edge',
            "code": """import torch
qa=torch.zeros(1,2,2); ka=torch.zeros_like(qa); va=torch.tensor([[[1.,2.],[3.,4.]]])
qd=torch.zeros(1,1,2); kd=torch.zeros_like(qd); vd=torch.tensor([[[100.,200.]]])
oa,od,m={fn}(qa,ka,va,qd,kd,vd)
assert torch.equal(oa[:,0],va[:,0]) and torch.equal(oa[:,1],va.mean(1))
assert torch.allclose(od,torch.tensor([[[104/3,206/3]]]))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
a=[torch.randn(1,3,4,requires_grad=True) for _ in range(3)]
d=[torch.randn(1,2,4,requires_grad=True) for _ in range(3)]
oa,od,m={fn}(*(a+d)); (oa.square().sum()+od.square().sum()).backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in a+d)""",
        },
    ],
    'solution': '''def mot_asymmetric_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm):
    """Mixture-of-Transformers (MoT) 非对称跨模态注意力 (Mixture-of-Transformers (MoT) Asymmetric Cross-Modal Attention)。"""
    # Step 1: 沿序列轴拼接 AR 与扩散流的 Q、K、V
    length_ar = q_ar.size(1)
    q_all = torch.cat((q_ar, q_dm), dim=1)
    k_all = torch.cat((k_ar, k_dm), dim=1)
    v_all = torch.cat((v_ar, v_dm), dim=1)
    # Step 2: 构造 AR 因果且屏蔽 DM、DM 查询全可见的非对称布尔掩码
    total_length = q_all.size(1)
    mask = torch.ones(total_length, total_length, dtype=torch.bool, device=q_all.device)
    mask[:length_ar] = False
    mask[:length_ar, :length_ar] = torch.tril(torch.ones(length_ar, length_ar, dtype=torch.bool, device=q_all.device))
    # Step 3: 对掩码后的缩放点积分数归一化，并按原序列边界拆分两路输出
    scores = q_all @ k_all.transpose(-2, -1) / math.sqrt(q_all.size(-1))
    scores = scores.masked_fill(~mask, float('-inf'))
    out = torch.softmax(scores, dim=-1) @ v_all
    return (out[:, :length_ar], out[:, length_ar:], mask)''',
    'demo': """torch.manual_seed(42)
ar = [torch.randn(2, 8, 64) for _ in range(3)]
dm = [torch.randn(2, 32, 64) for _ in range(3)]
out_ar, out_dm, mask = mot_asymmetric_attention(*ar + dm)
print('AR/DM 输出形状:', tuple(out_ar.shape), tuple(out_dm.shape))
print('AR 可见连接数:', int(mask[:8].sum()), 'DM 可见连接数:', int(mask[8:].sum()))
print('首个 AR 查询可见键:', mask[0].nonzero().flatten().tolist())""",
    'explanation': r"""### 核心思路
两种流共享一次注意力计算，但可见域不对称。令拼接序列长度 $L=S_a+S_d$，注意力为 $Y=\operatorname{softmax}(QK^\top/\sqrt D+M)V$。AR 查询行只允许看到不晚于自身的 AR 键，避免未来信息和扩散流泄漏；DM 查询行允许看到全部 AR 与 DM 键，以获得双向生成上下文。

### 代码步骤拆解
- **Step 1（合并两条 token 流）**：`q_all`、`k_all`、`v_all` 从 `(B,S_a,D)` 与 `(B,S_d,D)` 拼为 `(B,L,D)`，`length_ar` 记录之后的拆分边界。
- **Step 2（建立块状可见性）**：掩码左上角是 $S_a\times S_a$ 下三角；右上角 AR→DM 为全假；下方 $S_d\times L$ 的 DM 查询块为全真。
- **Step 3（掩码注意力与拆流）**：不可见分数填为 `-inf`，softmax 后与 `v_all` 相乘得到 `(B,L,D)`，再切成 `out_ar` 和 `out_dm`。

### 重点与易错点分析
- **注意点 1（查询行与键列不能反置）**：`mask[row, col]` 的行是查询、列是键；屏蔽块应是 `mask[:S_a, S_a:]`，而不是反方向，否则会限制 DM 读取 AR。
- **注意点 2（先掩码再 softmax）**：若在 softmax 后把概率置零，行概率和不再为 1；使用 `-inf` 可让不可见位置概率精确为零。
- **注意点 3（与普通 causal attention 的差别）**：整张 $L\times L$ 下三角会让 DM 查询也只能看过去，违背扩散流双向可见要求；这里必须显式构造分块掩码。

### 复杂度分析
- **时间复杂度**：$O(B(S_a+S_d)^2D)$，`QK^\top` 和注意力概率乘 V 是主要计算瓶颈。
- **空间复杂度**：$O(B(S_a+S_d)^2+(S_a+S_d)^2)$，分别保存每批分数/概率与共享布尔掩码。""",
}
