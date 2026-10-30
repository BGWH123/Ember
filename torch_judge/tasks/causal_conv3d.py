"""Video VAE Temporal Causal 3D Convolution task."""

TASK = {
    'title': 'Video VAE Temporal Causal 3D Convolution',
    'title_zh': '视频 VAE 时间因果三维卷积（Causal Conv3D）',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'causal_conv3d',
    'starter_code': """def causal_conv3d(x, weight, bias=None, pad_mode='replicate'):
    pass""",
    'hint': """Prepend k_t-1 frames (copies of the first frame or zeros), then apply symmetric zero padding only in height and width before conv3d with no internal padding.""",
    'hint_zh': '在时间轴前置 k_t-1 帧（首帧副本或零），再仅对高宽做对称零填充，最后调用无内部填充的 conv3d。',
    'description_en': """Apply a stride-one 3D convolution to video x while padding only the past side of time. Replicate or zero temporal padding is selectable; spatial padding is symmetric and zero-valued.""",
    'description_zh': '对视频 x 应用步长为 1 的三维卷积，并仅填充时间轴过去一侧。时间填充可选择复制或零值，空间填充为对称零填充。',
    'theory_en': """Left-only temporal padding aligns each kernel window with the current and earlier frames, preventing future leakage. Replicating the first frame provides a steady boundary condition often used by video VAEs.""",
    'theory_zh': '仅在时间轴左侧填充，使每个卷积窗口只对齐当前及更早帧，从而防止未来信息泄漏。复制首帧提供视频 VAE 常用的稳定边界条件。',
    'diagram_en': """```text
[first ... first | frame0 frame1 ...] -> causal temporal window
                 spatial zero pad -> conv3d -> T preserved
```""",
    'diagram_zh': """```text
[首帧 ... 首帧 | 帧0 帧1 ...] -> 因果时间窗口
                 空间零填充 -> conv3d -> 保持 T
```""",
    "tests": [
        {
            "name": 'output shape',
            "code": """import torch
x=torch.randn(2,3,4,7,9); w=torch.randn(5,3,3,3,3); b=torch.randn(5)
y={fn}(x,w,b)
assert y.shape==(2,5,4,7,9) and torch.isfinite(y).all()""",
        },
        {
            "name": 'exact padding reference',
            "code": """import torch
import torch.nn.functional as F
x=torch.arange(1.,1+1*1*3*2*2).reshape(1,1,3,2,2); w=torch.randn(1,1,2,3,3); b=torch.tensor([.2])
y={fn}(x,w,b,'replicate')
xp=torch.cat((x[:,:,:1],x),2); xp=F.pad(xp,(1,1,1,1,0,0))
ref=F.conv3d(xp,w,b,stride=1,padding=0)
assert torch.equal(y,ref)""",
        },
        {
            "name": 'zero padding and temporal causality',
            "code": """import torch
x=torch.randn(1,1,5,3,3); changed=x.clone(); changed[:,:,3:]+=100
w=torch.ones(1,1,3,1,1)
y={fn}(x,w,pad_mode='zero'); yc={fn}(changed,w,pad_mode='zero')
assert torch.equal(y[:,:,:3],yc[:,:,:3])
assert torch.allclose(y[:,:,0],x[:,:,0])""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x=torch.randn(2,2,4,5,5,requires_grad=True); w=torch.randn(3,2,3,3,3,requires_grad=True)
b=torch.randn(3,requires_grad=True); y={fn}(x,w,b,'replicate'); y.square().mean().backward()
assert x.grad is not None and w.grad is not None and b.grad is not None
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()""",
        },
    ],
    'solution': '''def causal_conv3d(x, weight, bias=None, pad_mode='replicate'):
    """时间因果三维卷积 (Causal 3D Convolution)。"""
    k_t, k_h, k_w = weight.shape[-3:]
    temporal_pad = k_t - 1

    # Step 1: 仅在时间轴过去一侧添加复制或零填充
    if pad_mode == "replicate":
        prefix = x[:, :, :1].expand(-1, -1, temporal_pad, -1, -1)
        x_padded = torch.cat((prefix, x), dim=2)
        x_padded = F.pad(
            x_padded, (k_w // 2, k_w // 2, k_h // 2, k_h // 2, 0, 0)
        )
    elif pad_mode == "zero":
        x_padded = F.pad(
            x,
            (k_w // 2, k_w // 2, k_h // 2, k_h // 2, temporal_pad, 0),
        )
    else:
        raise ValueError("pad_mode must be 'replicate' or 'zero'")

    # Step 2: 使用无内部 padding 的三维卷积保持时间长度
    return F.conv3d(x_padded, weight, bias, stride=1, padding=0)''',
    'demo': """torch.manual_seed(42)
B, C, T, H, W = 2, 3, 8, 16, 16
video = torch.randn(B, C, T, H, W)
weight = torch.randn(4, C, 3, 3, 3)
out = causal_conv3d(video, weight)
print("Output shape:", out.shape)
print("First-frame mean:", out[:, :, 0].mean().item())
print("Sample values:", out[0, 0, 0, :2, :2])""",
    'explanation': """### 核心思路
在时间轴仅左填充 $k_t-1$ 帧，使每个输出时间点只依赖当前与过去；空间轴则使用对称零填充。

### 代码步骤拆解
- **Step 1（时间填充）**：复制首帧或补零，并保持未来一侧不填充。
- **Step 2（空间填充）**：高宽各补 `k_h//2`、`k_w//2`，不改变时间边界语义。
- **Step 3（卷积计算）**：调用 `F.conv3d` 且内部 `padding=0`，避免重复填充。

### 重点与易错点分析
- **注意点 1（因果方向）**：时间填充只能位于过去一侧，否则会造成未来信息泄漏。
- **注意点 2（模式隔离）**：`replicate` 只作用时间轴，空间边界仍需零填充。
- **注意点 3（核大小）**：保持高宽尺寸的公式默认空间核为奇数；偶数核会改变输出大小。

### 复杂度分析
- **时间复杂度**：$O(BT H W C_{out}C_{in}k_tk_hk_w)$，卷积是主要瓶颈。
- **空间复杂度**：$O(BCTHW)$，主要是填充输入与输出激活。""",
}
