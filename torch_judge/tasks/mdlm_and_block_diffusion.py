"""Masked Diffusion LM (MDLM) Loss & Block-Causal Attention Mask task."""

TASK = {
    'title': 'Masked Diffusion LM (MDLM) Loss & Block-Causal Attention Mask',
    'title_zh': '掩码扩散语言模型（MDLM）损失与块因果掩码',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'block_diffusion_mask_and_loss',
    'starter_code': """def block_diffusion_mask_and_loss(logits, targets, is_masked, alpha_t, dalpha_dt, block_size=4):
    pass""",
    'hint': """Compare integer block IDs to build the mask. Gather target log-probabilities, multiply by the continuous-time schedule weight and masked-position indicator, then normalize by the masked count.""",
    'hint_zh': '比较整数块编号构造掩码。收集目标对数概率，乘以连续时间调度权重和掩码位置指示，再按掩码数量归一化。',
    'description_en': """Construct an (S,S) block-causal attention mask and compute the continuous-time MDLM weighted cross-entropy over only masked tokens. Tokens attend bidirectionally within their block and to every earlier block.""",
    'description_zh': '构造 (S,S) 块因果注意力掩码，并仅在被掩码 token 上计算连续时间 MDLM 加权交叉熵。token 在块内双向注意，并可注意所有更早块。',
    'theory_en': """Block diffusion decodes chunks in causal order while denoising each chunk bidirectionally. The schedule factor -alpha'(t)/(1-alpha(t)) is the continuous-time likelihood weighting for masked diffusion.""",
    'theory_zh': "块扩散按因果顺序解码分块，同时在每个块内双向去噪。调度因子 -alpha'(t)/(1-alpha(t)) 是掩码扩散的连续时间似然权重。",
    'diagram_en': """```text
block 0: [<->]          visible
block 1: [past][<->]    visible
block 2: [past][past][<->]
masked NLL * (-alpha'/(1-alpha)) -> normalized loss
```""",
    'diagram_zh': """```text
块 0：[双向]             可见
块 1：[过去][双向]       可见
块 2：[过去][过去][双向]
掩码 NLL * (-alpha'/(1-alpha)) -> 归一化损失
```""",
    "tests": [
        {
            "name": 'mask and scalar loss shapes',
            "code": """import torch
logits=torch.randn(2,7,5); targets=torch.randint(0,5,(2,7)); masked=torch.rand(2,7)>.5
m,l={fn}(logits,targets,masked,torch.tensor([[.3],[.8]]),torch.tensor([[-.5],[-.2]]),3)
assert m.shape==(7,7) and m.dtype==torch.bool and l.shape==torch.Size([])
assert torch.isfinite(l)""",
        },
        {
            "name": 'exact mask and loss reference',
            "code": """import torch
import torch.nn.functional as F
logits=torch.tensor([[[2.,0.],[0.,2.],[1.,1.],[3.,-1.]]]); targets=torch.tensor([[0,1,1,0]])
masked=torch.tensor([[1,0,1,1]],dtype=torch.bool); a=torch.tensor([[.25]]); da=torch.tensor([[-.75]])
m,l={fn}(logits,targets,masked,a,da,2)
ids=torch.arange(4)//2; rm=ids[None,:]<=ids[:,None]
nll=-F.log_softmax(logits,-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
rl=((-.75*-1)/(1-.25+1e-8)*nll*masked.float()).sum()/masked.float().sum().clamp(min=1.)
assert torch.equal(m,rm) and torch.equal(l,rl)""",
        },
        {
            "name": 'no masked tokens edge',
            "code": """import torch
logits=torch.randn(2,5,3); targets=torch.zeros(2,5,dtype=torch.long); masked=torch.zeros(2,5,dtype=torch.bool)
m,l={fn}(logits,targets,masked,torch.full((2,1),.5),torch.full((2,1),-.5),10)
assert torch.equal(m,torch.ones(5,5,dtype=torch.bool))
assert l.item()==0.0""",
        },
        {
            "name": 'gradient flow only through logits and schedule',
            "code": """import torch
logits=torch.randn(2,4,6,requires_grad=True); targets=torch.randint(0,6,(2,4)); masked=torch.tensor([[1,0,1,0],[0,1,0,1]])
a=torch.full((2,1),.4,requires_grad=True); da=torch.full((2,1),-.3,requires_grad=True)
m,l={fn}(logits,targets,masked,a,da,2); l.backward()
assert logits.grad is not None and a.grad is not None and da.grad is not None
assert torch.isfinite(logits.grad).all()""",
        },
    ],
    'solution': '''def block_diffusion_mask_and_loss(logits, targets, is_masked, alpha_t, dalpha_dt, block_size=4):
    """构造块因果掩码并计算 MDLM 损失 (Block Diffusion MDLM)。"""
    _, S, _ = logits.shape

    # Step 1: 通过块编号比较构造块内双向、块间因果掩码
    block_ids = torch.arange(S, device=logits.device) // block_size
    block_mask = block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)

    # Step 2: 收集目标 token 的负对数似然与时间权重
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    schedule_weight = -dalpha_dt / (1.0 - alpha_t + 1e-8)

    # Step 3: 仅在掩码位置加权并按位置数归一化
    masked = is_masked.to(logits.dtype)
    denominator = masked.sum().clamp(min=1.0)
    loss = (schedule_weight * nll * masked).sum() / denominator
    return block_mask, loss''',
    'demo': """torch.manual_seed(42)
logits = torch.randn(2, 8, 32)
targets = torch.randint(0, 32, (2, 8))
is_masked = torch.rand(2, 8) < 0.4
mask, loss = block_diffusion_mask_and_loss(logits, targets, is_masked, torch.full((2, 1), 0.5), torch.full((2, 1), -0.5))
print("Mask shape:", mask.shape)
print("Masked positions:", is_masked.sum().item())
print("Loss:", loss.item())""",
    'explanation': r"""### 核心思路
块扩散用整数块编号生成可见性规则：查询可看同块和所有更早块；MDLM 损失则以 $-\alpha\prime(t)/(1-\alpha(t))$ 加权被掩码 token 的 NLL。

### 代码步骤拆解
- **Step 1（块掩码）**：广播比较 `block_ids` 得到 `(S, S)` 布尔矩阵。
- **Step 2（目标 NLL）**：log-softmax 后 gather 目标类，得到 `(B, S)`。
- **Step 3（加权归一）**：乘调度权重和 `is_masked`，再按掩码位置总数归一。

### 重点与易错点分析
- **注意点 1（比较方向）**：行代表 query、列代表 key，因此应判断 key 块号不大于 query 块号。
- **注意点 2（空掩码）**：分母 `clamp(min=1)` 让无掩码样本返回精确零而非 NaN。
- **注意点 3（权重广播）**：`alpha_t` 与 `dalpha_dt` 通常是 `(B, 1)`，可沿序列维广播。

### 复杂度分析
- **时间复杂度**：$O(BSV+S^2)$，log-softmax 和块掩码构造占主导。
- **空间复杂度**：$O(BSV+S^2)$，主要保存对数概率和注意力掩码。""",
}
