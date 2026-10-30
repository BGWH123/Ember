"""Speculative Decoding task."""

TASK = {
    'title': 'Speculative Decoding',
    'title_zh': '推测解码',
    'difficulty': 'Hard',
    'category': '采样与解码',
    'function_name': 'speculative_decode',
    'starter_code': """def speculative_decode(target_probs, draft_probs, draft_tokens):
    pass""",
    'hint': """for i in range(K):
  accept_prob = min(1, p_target[i, token] / p_draft[i, token])
  if accepted: append token, continue
  else: resample from max(0, p_target[i] - p_draft[i]) normalized, append, return""",
    'hint_zh': """for i in range(K):
  accept_prob = min(1, p_target[i, token] / p_draft[i, token])
  若接受：追加 token，继续
  若拒绝：从归一化的 max(0, p_target[i] - p_draft[i]) 重采样，追加后返回""",
    'description_en': """Implement speculative decoding for faster LLM inference.

Speculative decoding uses a fast draft model to propose tokens, then verifies them against the target model, accepting or resampling based on probability ratios.

**Signature:** `speculative_decode(target_probs, draft_probs, draft_tokens) -> list[int]`

**Parameters:**
- `target_probs` — target model probabilities (K, V)
- `draft_probs` — draft model probabilities (K, V)
- `draft_tokens` — proposed token IDs (K,)

**Returns:** list of accepted token IDs (length 1 to K)

**Constraints:**
- Accept with prob `min(1, p_target/p_draft)`
- On rejection, resample from `max(0, p_target - p_draft)` normalized
- Stop at first rejection (return accepted so far + resampled token)""",
    'description_zh': """实现推测解码以加速 LLM 推理。

推测解码使用快速草稿模型提议 token，然后根据概率比率与目标模型验证，接受或重新采样。

**签名:** `speculative_decode(target_probs, draft_probs, draft_tokens) -> list[int]`

**参数:**
- `target_probs` — 目标模型概率 (K, V)
- `draft_probs` — 草稿模型概率 (K, V)
- `draft_tokens` — 提议的 token ID (K,)

**返回:** 接受的 token ID 列表（长度 1 到 K）

**约束:**
- 以概率 `min(1, p_target/p_draft)` 接受
- 拒绝时从归一化的 `max(0, p_target - p_draft)` 重新采样
- 在首次拒绝时停止（返回已接受的 + 重采样 token）""",
    "tests": [
        {
            "name": 'Perfect draft: all accepted',
            "code": """import torch
torch.manual_seed(0)
probs = torch.softmax(torch.randn(4, 10), dim=-1)
tokens = torch.tensor([2, 5, 1, 8])
accepted = {fn}(probs, probs, tokens)
assert len(accepted) == 4, f'Perfect draft should accept all, got {len(accepted)}'
for i in range(4):
    assert accepted[i] == tokens[i].item(), f'Token {i} mismatch'""",
        },
        {
            "name": 'Output length bounded',
            "code": """import torch
torch.manual_seed(0)
K = 5
target = torch.softmax(torch.randn(K, 8), dim=-1)
draft = torch.softmax(torch.randn(K, 8), dim=-1)
tokens = torch.randint(0, 8, (K,))
accepted = {fn}(target, draft, tokens)
assert 1 <= len(accepted) <= K, f'Length {len(accepted)} not in [1, {K}]'""",
        },
        {
            "name": 'All tokens valid',
            "code": """import torch
V = 8
for seed in range(20):
    torch.manual_seed(seed)
    target = torch.softmax(torch.randn(3, V), dim=-1)
    draft = torch.softmax(torch.randn(3, V), dim=-1)
    tokens = torch.randint(0, V, (3,))
    for t in {fn}(target, draft, tokens):
        assert 0 <= t < V, f'Token {t} out of range'""",
        },
        {
            "name": 'Rejection occurs when draft prob >> target prob',
            "code": """import torch
# Token 0: draft prob 0.9, target prob 0.1 -> acceptance ~11%
# Run 100 times; rejection must occur at least once
target_probs = torch.tensor([[0.1, 0.8, 0.1], [0.33, 0.33, 0.34]])
draft_probs  = torch.tensor([[0.9, 0.05, 0.05], [0.33, 0.33, 0.34]])
draft_tokens = torch.tensor([0, 1])
torch.manual_seed(99)
rejections = 0
for _ in range(100):
    accepted = {fn}(target_probs, draft_probs, draft_tokens)
    if len(accepted) < 3:
        rejections += 1
assert rejections > 0, f'Token 0 should be rejected at least sometimes (acceptance prob ~11%), got 0 rejections in 100 trials'""",
        },
    ],
    'solution': '''def speculative_decode(target_probs, draft_probs, draft_tokens):
    """实现推测解码（Speculative Decoding）。"""
    # Step 1: 初始化按草稿顺序验证的输出前缀
    num_draft_tokens = len(draft_tokens)
    accepted = []
    # Step 2: 以 min(1, p_target / p_draft) 逐 token 执行接受检验
    for i in range(num_draft_tokens):
        token_id = draft_tokens[i].item()
        ratio = target_probs[i, token_id] / max(draft_probs[i, token_id].item(), 1e-10)
        if torch.rand(1).item() < min(1.0, ratio.item()):
            accepted.append(token_id)
        else:
            # Step 3: 首次拒绝时从截断概率差分布采样替代 token 并立即终止
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            adjusted_sum = adjusted.sum()
            if adjusted_sum > 0:
                adjusted = adjusted / adjusted_sum
            else:
                adjusted = torch.ones_like(adjusted) / adjusted.shape[0]
            accepted.append(torch.multinomial(adjusted, 1).item())
            return accepted
    return accepted''',
    'demo': """torch.manual_seed(42)
target_probs = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
draft_probs = torch.tensor([[0.9, 0.05, 0.05], [0.2, 0.3, 0.5]])
draft_tokens = torch.tensor([0, 2])
decoded = speculative_decode(target_probs, draft_probs, draft_tokens)
print("Draft tokens:", draft_tokens.tolist())
print("Verified output:", decoded)
print("Stopped after rejection:", len(decoded) < len(draft_tokens))""",
    'explanation': r"""### 核心思路
设目标模型分布为 $p$、草稿模型分布为 $q$。草稿 token $x$ 以 $a(x)=\min(1,p(x)/q(x))$ 接受；若拒绝，则从残差分布 $r(v)\propto\max(0,p(v)-q(v))$ 采样替代 token。这个接受-修正机制使最终 token 的边缘分布仍为 $p$，同时可一次验证多个廉价草稿 token。

### 代码步骤拆解
- **Step 1（初始化验证前缀）**：`num_draft_tokens=K`，`accepted` 按位置保存已经通过目标模型检验的 token ID。
- **Step 2（概率比接受检验）**：位置 `i` 只读取提议 `token_id` 在两模型下的概率，以 `min(1, ratio)` 与均匀随机数比较；接受后继续验证下一位置。
- **Step 3（拒绝后的残差重采样）**：首次拒绝时对长度 $V$ 的 `target_probs[i] - draft_probs[i]` 截断负值、归一化并 `multinomial` 采样，追加替代 token 后立即返回。

### 重点与易错点分析
- **注意点 1（不能从目标分布直接重采样）**：拒绝事件已携带关于 $p$ 与 $q$ 的信息，直接从 $p$ 再采样会产生偏差；必须使用正部差分 $(p-q)_+$ 才能恢复目标边缘分布。
- **注意点 2（零草稿概率）**：分母下界避免数值除零；理论上若 $q(x)=0$，该 token 不会由草稿采到，本题保护项主要应对有限精度或外部输入。
- **注意点 3（提前终止与退化分支）**：只在首次拒绝处构造残差分布；若数值输入使残差和为 0，代码退化为均匀分布以保持可采样。全部通过时，本题接口返回恰好 $K$ 个草稿 token，不包含标准算法有时额外生成的 bonus token。

### 复杂度分析
- **时间复杂度**：$O(K+V)$，最多做 $K$ 次标量接受检验，并仅在首次拒绝时扫描一次 $V$ 维残差分布。
- **空间复杂度**：$O(K+V)$，返回前缀最多含 $K$ 个 Python 整数，拒绝分支临时保存一个 $V$ 维 `adjusted` 分布。""",
}
