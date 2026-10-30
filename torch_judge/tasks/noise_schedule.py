"""Diffusion Noise Schedules task."""

TASK = {
    'title': 'Diffusion Noise Schedules',
    'title_zh': '扩散噪声调度',
    'difficulty': 'Easy',
    'category': '扩散与流模型',
    'function_name': 'noise_schedule',
    'starter_code': """def noise_schedule(num_timesteps, schedule_type='cosine'):
    pass""",
    'hint': """linear/sigmoid: compute betas → alpha_bar = torch.cumprod(1-betas, dim=0)
cosine: f(t) = cos(...)² → alpha_bar = f(t)/f(0), clamp to [0.0001, 0.9999]
sigmoid scaling: shift+scale so min(sig)→β_start, max(sig)→β_end""",
    'hint_zh': """linear/sigmoid：计算 betas → alpha_bar = torch.cumprod(1-betas, dim=0)
cosine：f(t) = cos(...)² → alpha_bar = f(t)/f(0)，clamp 到 [0.0001, 0.9999]
sigmoid 缩放：平移+缩放使 min(sig)→β_start，max(sig)→β_end""",
    'description_en': """Implement three noise schedules used in diffusion models.

Each schedule returns `alpha_bar` (ᾱ), the cumulative product of (1 - β) across timesteps. ᾱ_t controls the signal-to-noise ratio: close to 1 means mostly signal, close to 0 means mostly noise.

**Signature:** `noise_schedule(num_timesteps, schedule_type='cosine') -> Tensor`

**Parameters:**
- `num_timesteps` — number of diffusion timesteps T
- `schedule_type` — one of `'linear'`, `'cosine'`, or `'sigmoid'`

**Returns:** `alpha_bar` tensor of shape `(num_timesteps,)`, values in (0, 1], monotonically decreasing

**Formulas:**

`linear`: β_t = β_start + (β_end − β_start) × t/(T−1), then ᾱ = cumprod(1 − β)
- Use β_start=1e-4, β_end=0.02

`cosine`: ᾱ_t = cos²((t/T + s)/(1+s) × π/2) / cos²(s/(1+s) × π/2)
- Use s=0.008, clip to [0.0001, 0.9999]

`sigmoid`: β_t = sigmoid(t/T × 12 − 6), scaled to [β_start, β_end], then ᾱ = cumprod(1 − β)
- Use β_start=1e-4, β_end=0.02""",
    'description_zh': """实现扩散模型中使用的三种噪声调度。

每种调度返回 `alpha_bar`（ᾱ），即各时间步 (1 - β) 的累积乘积。ᾱ_t 控制信噪比：接近 1 表示主要是信号，接近 0 表示主要是噪声。

**签名:** `noise_schedule(num_timesteps, schedule_type='cosine') -> Tensor`

**参数:**
- `num_timesteps` — 扩散时间步数 T
- `schedule_type` — `'linear'`、`'cosine'` 或 `'sigmoid'` 之一

**返回:** 形状为 `(num_timesteps,)` 的 `alpha_bar` 张量，值域 (0, 1]，单调递减

**公式:**

`linear`：β_t = β_start + (β_end − β_start) × t/(T−1)，然后 ᾱ = cumprod(1 − β)
- 使用 β_start=1e-4，β_end=0.02

`cosine`：ᾱ_t = cos²((t/T + s)/(1+s) × π/2) / cos²(s/(1+s) × π/2)
- 使用 s=0.008，截断到 [0.0001, 0.9999]

`sigmoid`：β_t = sigmoid(t/T × 12 − 6)，缩放到 [β_start, β_end]，然后 ᾱ = cumprod(1 − β)
- 使用 β_start=1e-4，β_end=0.02""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(100, stype)
    assert out.shape == (100,), f'{stype}: expected shape (100,), got {out.shape}'""",
        },
        {
            "name": 'All values in (0, 1]',
            "code": """import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(200, stype)
    assert (out > 0).all(), f'{stype}: alpha_bar must be > 0, min={out.min().item()}'
    assert (out <= 1.0).all(), f'{stype}: alpha_bar must be <= 1, max={out.max().item()}'""",
        },
        {
            "name": 'Monotonically decreasing',
            "code": """import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(100, stype)
    diffs = out[1:] - out[:-1]
    assert (diffs <= 1e-6).all(), f'{stype}: alpha_bar must be monotonically decreasing'""",
        },
        {
            "name": 'alpha_bar[0] close to 1 (low noise at start)',
            "code": """import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(1000, stype)
    assert out[0].item() > 0.9, f'{stype}: alpha_bar[0] should be close to 1, got {out[0].item()}'""",
        },
        {
            "name": 'alpha_bar[-1] close to 0 (high noise at end)',
            "code": """import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(1000, stype)
    assert out[-1].item() < 0.05, f'{stype}: alpha_bar[-1] should be close to 0, got {out[-1].item()}'""",
        },
        {
            "name": 'All three schedule types work without error',
            "code": """import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(50, stype)
    assert isinstance(out, torch.Tensor), f'{stype}: must return a torch.Tensor'
    assert out.dtype == torch.float32, f'{stype}: expected float32, got {out.dtype}'
try:
    {fn}(10, 'unknown')
    assert False, 'Should raise ValueError for unknown schedule_type'
except (ValueError, Exception):
    pass""",
        },
        {
            "name": 'Linear schedule exact values',
            "code": r"""import torch
T = 5
out = {fn}(T, schedule_type='linear')
beta_start, beta_end = 1e-4, 0.02
t = torch.arange(T, dtype=torch.float32)
betas = beta_start + (beta_end - beta_start) * t / (T - 1)
alpha_bars = torch.cumprod(1.0 - betas, dim=0)
assert torch.allclose(out, alpha_bars, atol=1e-5), f'Linear schedule mismatch.\nExpected: {alpha_bars}\nGot: {out}'""",
        },
    ],
    'solution': '''def noise_schedule(num_timesteps, schedule_type='cosine'):
    """扩散噪声调度 (Diffusion Noise Schedules)。"""
    # Step 1: 建立 float32 时间轴，作为 beta 或闭式 alpha_bar 的采样位置
    num_steps = num_timesteps
    timesteps = torch.arange(num_steps, dtype=torch.float32)
    # Step 2: 按线性、余弦或 S 形定义生成逐步 beta 或直接生成累计信号率
    if schedule_type == 'linear':
        beta_start, beta_end = (0.0001, 0.02)
        betas = beta_start + (beta_end - beta_start) * timesteps / (num_steps - 1)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
    elif schedule_type == 'cosine':
        cosine_offset = 0.008
        cosine_values = torch.cos((timesteps / num_steps + cosine_offset) / (1 + cosine_offset) * (math.pi / 2)) ** 2
        initial_cosine = math.cos(cosine_offset / (1 + cosine_offset) * (math.pi / 2)) ** 2
        alpha_bar = (cosine_values / initial_cosine).clamp(0.0001, 0.9999)
    elif schedule_type == 'sigmoid':
        beta_start, beta_end = (0.0001, 0.02)
        sigmoid_input = timesteps / num_steps * 12 - 6
        sigmoid_values = 1 / (1 + torch.exp(-sigmoid_input))
        betas = beta_start + (beta_end - beta_start) * (sigmoid_values - sigmoid_values.min()) / (sigmoid_values.max() - sigmoid_values.min())
        alpha_bar = torch.cumprod(1 - betas, dim=0)
    else:
        raise ValueError(f'Unknown schedule_type: {schedule_type}')
    # Step 3: 返回随扩散时间单调下降的 alpha_bar 信号保留系数
    return alpha_bar''',
    'demo': """torch.manual_seed(42)
T = 1000
schedules = ['linear', 'cosine', 'sigmoid']
results = {s: noise_schedule(T, s) for s in schedules}
print('起始 alpha_bar:', {s: round(results[s][0].item(), 4) for s in schedules})
print('中点 alpha_bar:', {s: round(results[s][T // 2].item(), 4) for s in schedules})
print('末端 alpha_bar:', {s: round(results[s][-1].item(), 6) for s in schedules})""",
    'explanation': r"""### 核心思路
扩散前向过程用 $\bar\alpha_t$ 控制保留信号与注入噪声的比例：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。线性与 sigmoid 调度先定义 $\beta_t$，再计算 $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$；余弦调度直接令 $\bar\alpha_t=f(t)/f(0)$，其中 $f(t)=\cos^2\!\left(\frac{t/T+s}{1+s}\frac\pi2\right)$。

### 代码步骤拆解
- **Step 1（离散时间轴）**：`timesteps=torch.arange(T,float32)` 固定输出 dtype，并给三种调度提供 $t=0,\ldots,T-1$ 的采样点。
- **Step 2（选择噪声增长曲线）**：linear 在 $10^{-4}$ 到 $0.02$ 间等距插值；cosine 使用偏移 `0.008` 的闭式信号曲线并按初值归一化；sigmoid 把 $[-6,6)$ 的 S 曲线重新缩放到相同 beta 端点。
- **Step 3（形成累计信号率）**：beta 型调度对 `1-betas` 做 `cumprod`，余弦型裁剪到 `[0.0001,0.9999]`，返回长度 T 的单调下降向量。

### 重点与易错点分析
- **注意点 1（beta 与 alpha_bar 不可混淆）**：返回值不是逐步 $\alpha_t=1-\beta_t$，而是从 0 到 t 的累计乘积；遗漏 `cumprod` 会导致末端仍保留大量信号。
- **注意点 2（余弦时间归一化）**：公式使用 `timesteps / num_steps` 而非 `/ (num_steps-1)`，并除以 `f(0)`；裁剪使首项为 0.9999 而不是精确 1。
- **注意点 3（sigmoid 需重标定端点）**：原始 sigmoid(-6) 并非 0、sigmoid(6) 也并非 1，因此要先减本离散序列最小值再除极差，才能严格映射到指定 beta 范围。

### 复杂度分析
- **时间复杂度**：$O(T)$，三种分支都只需逐时间步的逐元素函数与一次累计乘积或裁剪。
- **空间复杂度**：$O(T)$，时间轴、beta/余弦值及最终 `alpha_bar` 均为长度 T 的向量。""",
}
