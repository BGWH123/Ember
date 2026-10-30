"""KIVI Asymmetric KV Cache Quantization (Per-Channel K, Per-Token V) task."""

TASK = {
    'title': 'KIVI Asymmetric KV Cache Quantization (Per-Channel K, Per-Token V)',
    'title_zh': 'KIVI 非对称 KV Cache 量化（按通道 K / 按 Token V）',
    'difficulty': 'Medium',
    'category': '推理优化',
    'function_name': 'kivi_asymmetric_kv_quant',
    'starter_code': """def kivi_asymmetric_kv_quant(K, V, n_bits=2, eps=1e-8):
    pass""",
    'hint': """Reduce K over sequence dimension 1 and V over channel dimension 2. Use each min as the affine offset and quantize into levels 0 through 2^n_bits-1.""",
    'hint_zh': 'K 沿序列维 1 归约，V 沿通道维 2 归约；以各自最小值为仿射偏移，量化到 0 至 2^n_bits-1。',
    'description_en': """Implement KIVI's asymmetric low-bit KV-cache quantization with per-channel key parameters and per-token value parameters.""",
    'description_zh': '实现 KIVI 非对称低比特 KV 缓存量化：键按通道设置参数，值按 token 设置参数。',
    'theory_en': """Key outliers are structured by feature channel across positions, whereas value outliers are better localized per token. Using different quantization axes reduces error at very low bit width.""",
    'theory_zh': '键异常值通常沿位置共享通道结构，而值异常值更适合按 token 局部处理。采用不同量化轴可降低极低比特下的误差。',
    'diagram_en': """```text
K(B,L,D) --min/max over L--> scale(B,1,D) -> quantize/dequantize
V(B,L,D) --min/max over D--> scale(B,L,1) -> quantize/dequantize
```""",
    'diagram_zh': """```text
K(B,L,D) --沿 L 求 min/max--> scale(B,1,D) -> 量化/反量化
V(B,L,D) --沿 D 求 min/max--> scale(B,L,1) -> 量化/反量化
```""",
    "tests": [
        {
            "name": 'cache and scale shapes',
            "code": """import torch
K = torch.randn(2,5,4); V = torch.randn(2,5,4)
kd, vd, ks, vs = {fn}(K,V,n_bits=2)
assert kd.shape == vd.shape == (2,5,4)
assert ks.shape == (2,4) and vs.shape == (2,5)""",
        },
        {
            "name": 'exact asymmetric reference',
            "code": """import torch
K = torch.tensor([[[-2.,1.],[0.,4.],[3.,2.]]],dtype=torch.double)
V = torch.tensor([[[-1.,2.],[3.,7.],[-4.,5.]]],dtype=torch.double)
kd,vd,ks,vs = {fn}(K,V,n_bits=3,eps=1e-12)
kmn,kmx = K.amin(1,True),K.amax(1,True); rks=(kmx-kmn).clamp(min=1e-12)/7
vk,vx = V.amin(2,True),V.amax(2,True); rvs=(vx-vk).clamp(min=1e-12)/7
rkd=torch.clamp(torch.round((K-kmn)/rks),0.,7.)*rks+kmn
rvd=torch.clamp(torch.round((V-vk)/rvs),0.,7.)*rvs+vk
assert torch.equal(kd,rkd) and torch.equal(vd,rvd)
assert torch.equal(ks,rks.squeeze(1)) and torch.equal(vs,rvs.squeeze(2))""",
        },
        {
            "name": 'constant cache edge case',
            "code": """import torch
K = torch.full((2,3,4),2.5); V = torch.full((2,3,4),-1.5)
kd,vd,ks,vs = {fn}(K,V,n_bits=2,eps=1e-6)
assert torch.equal(kd,K) and torch.equal(vd,V)
assert torch.allclose(ks,torch.full((2,4),1e-6/3))
assert torch.allclose(vs,torch.full((2,3),1e-6/3))""",
        },
        {
            "name": 'gradient through dequantization',
            "code": """import torch
K = torch.randn(2,5,3,requires_grad=True); V = torch.randn(2,5,3,requires_grad=True)
kd,vd,ks,vs = {fn}(K,V,n_bits=4)
(kd.square().sum()+vd.square().sum()+ks.sum()+vs.sum()).backward()
assert kd.requires_grad and vd.requires_grad
assert K.grad is not None and V.grad is not None
assert torch.isfinite(K.grad).all() and torch.isfinite(V.grad).all()""",
        },
    ],
    'solution': '''def kivi_asymmetric_kv_quant(K, V, n_bits=2, eps=1e-8):
    """执行 KIVI 非对称 KV 缓存量化 (KIVI KV Quantization)。"""
    q_max = 2 ** n_bits - 1

    # Step 1: 沿序列维标定并量化键缓存
    key_min = K.amin(dim=1, keepdim=True)
    key_max = K.amax(dim=1, keepdim=True)
    key_scale = (key_max - key_min).clamp(min=eps) / q_max
    key_quantized = torch.clamp(torch.round((K - key_min) / key_scale), 0.0, float(q_max))
    key_dequantized = key_quantized * key_scale + key_min

    # Step 2: 沿通道维标定并量化值缓存
    value_min = V.amin(dim=2, keepdim=True)
    value_max = V.amax(dim=2, keepdim=True)
    value_scale = (value_max - value_min).clamp(min=eps) / q_max
    value_quantized = torch.clamp(torch.round((V - value_min) / value_scale), 0.0, float(q_max))
    value_dequantized = value_quantized * value_scale + value_min

    # Step 3: 返回反量化缓存和去除单例维的尺度
    return key_dequantized, value_dequantized, key_scale.squeeze(1), value_scale.squeeze(2)''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 16, 8
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
K_dequantized, V_dequantized, key_scale, value_scale = kivi_asymmetric_kv_quant(K, V)
print("Output shapes:", K_dequantized.shape, V_dequantized.shape)
print("Scale shapes:", key_scale.shape, value_scale.shape)
print("Mean key error:", (K - K_dequantized).abs().mean().item())""",
    'explanation': r"""### 核心思路
K 沿序列维使用逐通道范围，V 沿通道维使用逐 token 范围，分别映射到 $0$ 至 $2^{n\_bits}-1$ 后反量化。

### 代码步骤拆解
- **Step 1（键量化）**：`key_scale` 为 `(B, 1, D)`，跨全部位置共享每通道参数。
- **Step 2（值量化）**：`value_scale` 为 `(B, S, 1)`，每个 token 独立标定通道范围。
- **Step 3（布局恢复）**：返回缓存原形状，并仅压掉尺度的归约单例维。

### 重点与易错点分析
- **注意点 1（量化轴）**：K 使用 `dim=1`，V 使用 `dim=2`，两者不能互换。
- **注意点 2（常量范围）**：`clamp(min=eps)` 保证常量切片仍有有限尺度，反量化保持原值。
- **注意点 3（低比特边界）**：舍入后仍需 `clamp` 到合法整数等级，避免浮点误差越界。

### 复杂度分析
- **时间复杂度**：$O(BSD)$，标定、量化和反量化均线性遍历缓存。
- **空间复杂度**：$O(BSD)$，主要保存量化临时量和反量化输出。""",
}
