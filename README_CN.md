[English](./README.md) | [中文](./README_CN.md)

<p align="center">
  <h1 align="center">🔥 Ember</h1>
  <p align="center">
    从零实现 Transformers、vLLM、TRL 等系统的核心模块。
  </p>
  <p align="center">
    <em>读完论文，写出代码。无需 GPU。</em>
  </p>
  <p align="center">
    <a href="https://star-history.com/#BGWH123/Ember&Date">
      <img src="https://img.shields.io/github/stars/BGWH123/Ember?style=social" alt="GitHub stars" />
    </a>
  </p>
</p>

---

## 💡 项目理念

Ember 是一个为「**通过动手构建来理解 AI 系统**」而设计的实战训练场。

它将 101 道编码练习与本地 FastAPI 评测引擎结合，让你可以在自己的电脑上完成编写、测试、迭代的完整闭环。除此之外，还有 13 章理论面试题（八股文）和模拟面试模式，随机抛出编码题与理论题的组合挑战。

无需云账号，无需租用 GPU。只需你、浏览器和代码。

**适合谁：**
- 正在准备 ML 系统面试的工程师
- 用过 `nn.MultiheadAttention` 但从未手写过的研究者
- 比起看视频，更习惯通过敲键盘来学习的人

---

## 🛠️ 能做什么

| 能力 | 说明 |
|---|---|
| **浏览器内编码** | 内置 Monaco 编辑器，支持 Python 语法高亮，无需配置本地 IDE |
| **实时评测** | 提交代码后秒出隐藏测试用例的通过情况 |
| **对照学习** | 完成自己的实现后，解锁带逐行注释的参考答案 |
| **理论题库（八股文）** | 13 章结构化面试题，覆盖大模型架构、对齐技术、分布式训练、推理优化 |
| **模拟面试** | 随机组合编码题与理论题，模拟真实面试的紧张节奏 |
| **系统化路线** | 从注意力机制到图神经网络、扩散模型，跟着预设路径循序渐进 |
| **进度持久化** | 做题记录与尝试次数本地保存，刷新浏览器或换设备不丢失 |
| **AI 辅助** | 按需接入任意 OpenAI 兼容接口，卡住时获得个性化提示 |
| **隐私优先** | 全链路本地运行，代码和数据不离开你的电脑（AI 辅助除外） |

---

## 📚 内容体系

### 编码练习 — 101 题，22 个方向

每道题均包含 3–5 个隐藏测试用例，由本地评测服务验证。

| 方向 | 题数 | 内容 |
|---|---|---|
| **注意力机制** | 14 | 缩放点积、多头、因果、交叉、GQA、滑动窗口、线性、Flash、差分注意力、MLA、注意力掩码、CLIP、Einsum、余弦相似度 |
| **基础组件** | 14 | ReLU、GELU、SwiGLU、Leaky ReLU、Softmax、Dropout、Embedding、Linear、Kaiming 初始化、线性回归、MLP、残差连接、LayerScale、广播 |
| **损失函数** | 11 | 交叉熵、标签平滑、Focal Loss、对比损失、二元交叉熵、MSE、KL 散度、L1/L2 正则、DPO、GRPO、PPO |
| **图神经网络** | 7 | GCN、图读出、GAT、GIN、MPNN、GraphSAGE、链接预测、图自编码器 |
| **激活函数** | 7 | ReLU、GELU、SwiGLU、Leaky ReLU、Sigmoid（BCE）、Softmax |
| **Transformer 组件** | 6 | GPT-2 Block、ViT Patch、ViT Block、Transformer 编码器、Transformer 解码器、CLIP 模型 |
| **优化器与学习率** | 5 | Adam、余弦学习率、梯度裁剪、L1/L2 正则 |
| **采样与解码** | 5 | Top-k、束搜索、推测解码、BPE、多 Token 预测 |
| **正则化** | 5 | Dropout、L1、L2、标签平滑、LayerScale |
| **高效训练** | 4 | 混合精度、梯度累积、激活检查点、FSDP |
| **归一化** | 4 | LayerNorm、BatchNorm、RMSNorm、adaLN-Zero |
| **位置编码与嵌入** | 3 | 正弦编码、RoPE、ALiBi、NTK-aware RoPE |
| **混合专家** | 3 | MoE、MoE 负载均衡、多 Token 预测 |
| **扩散与流匹配** | 2 | 噪声调度、DDIM 步骤、流匹配 |
| **量化** | 2 | INT8 量化、QLoRA |
| **推理优化** | 2 | KV Cache、推测解码 |
| **训练技巧** | 2 | 梯度累积、激活检查点 |
| **分词** | 1 | BPE |
| **多模态** | 1 | CLIP 模型 |
| **参数高效微调** | 1 | LoRA |
| **状态空间模型** | 1 | Mamba SSM |
| **强化学习** | 1 | PPO 损失 |

### 学习路线

根据目标选择适合自己的路径：

| 路线 | 题数 | 覆盖内容 |
|---|---|---|
| **Transformer 内部机制** | 12 | 激活函数 → 归一化 → 注意力 → GPT-2 Block |
| **注意力与位置编码** | 13 | 所有注意力变体 + RoPE、ALiBi、NTK-RoPE |
| **从零训练 GPT** | 15 | Embedding → 架构 → 损失 → 优化器 → 训练技巧 |
| **推理与分布式** | 9 | KV Cache、量化、采样、张量并行、FSDP |
| **对齐与推理搜索** | 6 | 奖励模型 → DPO → GRPO → PPO → MCTS |
| **ViT 全流程** | 7 | 卷积 → Patch Embedding → ViT Block |
| **扩散模型与 DiT** | 5 | 噪声调度 → DDIM → 流匹配 → adaLN-Zero |
| **LLM 前沿架构** | 7 | GQA、差分注意力、MLA、MoE、多 Token 预测 |
| **图神经网络** | 8 | GCN → GAT → GIN → MPNN → GraphSAGE → 链接预测 → GAE |

```
路线导航：

基础 ──→ Transformer 内部机制 ──→ 从零训练 GPT
                 │                        │
                 ▼                        ▼
        注意力与位置编码            推理与分布式
                 │                        │
                 ▼                        ▼
        LLM 前沿架构             对齐与推理搜索
                 │
          ┌──────┼──────┐
          ▼      ▼      ▼
     ViT 全流程  扩散   图神经网络
```

### 面试准备 — 八股文 & 模拟面试

除了编码练习，Ember 还内置了完整的理论面试题库和模拟面试模式：

**八股文 — 13 章、200+ 道题：**

| 章节 | 内容 |
|---|---|
| LLM 基础面 | 架构（Encoder-Decoder / Causal / Prefix）、训练目标、涌现能力 |
| LLM 进阶面 | 复读机问题、LLaMA 系列、长文本处理 |
| 评测面 | POPE、MME、CHAIR、AMBER、评测基准设计 |
| 推理面 | KV Cache、量化、解码策略、vLLM |
| 微调面 | LoRA、QLoRA、全量微调、指令微调 |
| 训练集面 | 数据采集、清洗、去重、领域混合 |
| Agent 面 | 工具调用、ReAct、规划、多智能体 |
| RLHF / PPO 面 | 奖励建模、PPO、DPO、GRPO、偏好优化 |
| LangChain 面 | Chains、Agents、Memory、RAG |
| 增量预训练 | 领域适配、灾难性遗忘 |
| 蒸馏篇 | 知识蒸馏、自蒸馏 |
| 分布式训练 | 数据并行、张量并行、流水线并行、FSDP |
| 显存问题 | 梯度检查点、激活重计算、Offloading |

**模拟面试：**
- 随机从八股文抽取理论题 + 从题库抽取编程题
- 模拟真实面试节奏（理论 → 编码 → 追问）
- 跨会话追踪面试表现

---

## 🚀 快速上手

### 环境要求

| 依赖 | 最低版本 | 检查命令 |
|------|---------|---------|
| Python | 3.11 | `python --version` 或 `python3 --version` |
| Node.js | 18 | `node --version` |
| npm | （随 Node.js 捆绑） | `npm --version` |
| Git | 任意 | `git --version` |

**平台说明**
- **macOS**：系统可能未预装 Python。推荐通过 [Homebrew](https://brew.sh/) 安装（`brew install python node`）或使用 [uv](https://docs.astral.sh/uv/)。
- **Windows**：推荐使用 PowerShell，CMD 亦可。确保 Python 和 Node.js 已加入系统 `PATH`。
- **Linux**：大多数发行版自带 Python 3.11+。如果没有，使用包管理器或 [pyenv](https://github.com/pyenv/pyenv) 安装。

---

### 安装

提供四种启动方式，选择最适合你的即可。

#### 方式 A — 一键脚本（推荐）

最快的方式。脚本会自动创建 Python 虚拟环境（优先使用 `uv`，没有则回退到 `python -m venv`），安装所有 Python 和 Node.js 依赖，并打印启动命令。

**macOS / Linux**

```bash
# 1. 克隆仓库
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. 执行安装脚本
./setup.sh

# 3. 启动开发服务器
npm run dev
```

**Windows (PowerShell)** — 如遇权限问题请以管理员身份运行：

```powershell
# 1. 克隆仓库
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. 执行安装脚本
.\setup.ps1

# 3. 启动开发服务器
npm run dev
```

**Windows (CMD)**

```cmd
git clone https://github.com/BGWH123/Ember.git
cd Ember
setup.bat
npm run dev
```

<details>
<summary>安装脚本做了什么？（点击展开）</summary>

1. 检查是否安装了 `uv`（高性能 Python 包管理器）。如果已安装，使用 `uv` 创建 `.venv` 并安装依赖；否则回退到 `python -m venv`。
2. 以可编辑模式安装项目（`pip install -e ".[dev]"`）。
3. 执行 `npm install` 安装前端依赖。
4. 打印成功消息和下一步提示。

</details>

> **注意**：当 `.venv` 存在时，`npm run dev` 会自动优先使用项目内的 Python；如果 `.venv` 不存在，则回退到当前 shell 的 `python`。

---

#### 方式 B — Conda

如果你已经在使用 Conda 管理环境：

```bash
# 1. 克隆仓库
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. 创建并激活环境
conda create -n pyre python=3.11 -y
conda activate pyre

# 3. 安装 Python 依赖
pip install -e ".[dev]"

# 4. 安装 Node.js 依赖
npm install

# 5. 启动
npm run dev
```

> 每次运行前记得 `conda activate pyre`。

---

#### 方式 C — 手动（venv）

如果你希望完全掌控环境配置：

**macOS / Linux**

```bash
# 1. 克隆仓库
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. 创建虚拟环境
# 使用 uv（推荐）：
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 或使用标准 venv：
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -e ".[dev]"

# 3. 安装前端依赖
npm install

# 4. 启动
npm run dev
```

**Windows (PowerShell / CMD）**

```powershell
git clone https://github.com/BGWH123/Ember.git
cd Ember

python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

npm install
npm run dev
```

---

#### 方式 D — Docker

如果你偏好容器化部署，或不想在本地安装 Python/Node.js：

```bash
# 1. 克隆仓库
git clone https://github.com/BGWH123/Ember.git
cd Ember

# 2. 构建并运行
docker compose up --build
```

然后打开 `http://localhost:3000`。

- **评测服务** 运行在 `grading` 容器内，暴露端口 `8000`。
- **Web 应用** 运行在 `web` 容器内，暴露端口 `3000`。
- 进度数据持久化在 Docker 卷 `pyre_data` 中。

停止服务：
```bash
docker compose down
```

重置所有数据（包括进度）：
```bash
docker compose down -v
```

> **Docker 排错**：
> - 如遇端口冲突，请检查是否有其他服务占用了 `3000` 或 `8000` 端口。
> - Windows 上执行前请确保 Docker Desktop 正在运行。

---

### 验证安装

启动后，你应该能看到两个服务：

| 服务 | 地址 | 检查内容 |
|------|------|---------|
| 评测服务 | `http://localhost:8000` | 浏览器打开 → 应显示 FastAPI 文档 |
| Web 应用 | `http://localhost:3000` | 应加载题目列表 |

如果两者均可访问，说明安装成功。选一道题开始写代码吧！

---

### 配置

| 环境变量                | 默认值                    | 说明              |
| ----------------------- | ------------------------- | ----------------- |
| `GRADING_SERVICE_URL` | `http://localhost:8000` | 评测服务地址      |
| `DB_PATH`             | `./data/ember.db`        | SQLite 数据库路径 |

在 `web/.env.local` 中设置即可覆盖。

---

### AI 辅助（可选）

如需启用服务端 AI 提示，复制示例环境文件并填写你的 API 信息：

```bash
cp web/.env.example web/.env
```

编辑 `web/.env`：

```bash
AI_HELP_BASE_URL=https://api.openai.com/v1
AI_HELP_API_KEY=sk-...
AI_HELP_MODEL=gpt-4o-mini
```

支持任意 OpenAI 兼容接口（OpenAI、Anthropic 代理、Ollama 等）。如果未配置服务端，用户也可以在 UI 中自行填写 API key。

---

### 常见问题 / FAQ

<details>
<summary><strong>Q：安装脚本提示 "python not found"</strong></summary>

请确认 Python 3.11+ 已安装且加入了 `PATH`。Windows 上可能需要使用 `py` 代替 `python`。也可以手动创建 `.venv`（见方式 C）。

</details>

<details>
<summary><strong>Q：npm install 失败 / 卡住</strong></summary>

- 检查 Node.js 版本：`node --version`（必须 >= 18）。
- 尝试清除 npm 缓存：`npm cache clean --force`。
- 如果你在使用代理，配置 npm：`npm config set proxy http://...`。

</details>

<details>
<summary><strong>Q：后端已启动，但前端显示连接失败</strong></summary>

检查评测服务是否确实在 `8000` 端口运行：
```bash
curl http://localhost:8000
```
如果没有运行，检查 `logs/backend.log` 日志，或手动启动后端：
```bash
python -m grading_service.main
```

</details>

<details>
<summary><strong>Q：Docker 构建失败，提示找不到 public 目录</strong></summary>

确保 `web/` 目录下存在 `public/` 文件夹。如果没有，创建一个空的：
```bash
mkdir -p web/public
```
Next.js 静态导出有时需要这个目录存在。

</details>

<details>
<summary><strong>Q：可以不用 Docker 运行吗？</strong></summary>

可以 — 方式 A、B、C 均支持原生运行，无需 Docker。

</details>

<details>
<summary><strong>Q：需要 GPU 吗？</strong></summary>

不需要。所有题目均在 CPU 上运行。只有当你想在 Ember 之外训练大模型时，才需要 GPU。

</details>

---

## 🏗️ 技术架构

| 层级     | 技术                                                                   |
| -------- | ---------------------------------------------------------------------- |
| 前端     | Next.js + Monaco Editor + Tailwind CSS                                 |
| 后端     | FastAPI 评测服务                                                       |
| 评测引擎 | [torch_judge](https://github.com/duoan/TorchCode) — 执行并验证 101 道题，每题 3–5 个测试用例 |
| 存储     | SQLite（进度持久化）                                                   |

---

## 📁 项目目录

```
pyre-code/
├── web/                  # Next.js 前端
│   ├── src/app/          # 页面与 API 路由
│   ├── src/components/   # UI 组件
│   └── src/lib/          # 工具函数、题目数据
├── grading_service/      # FastAPI 评测后端
├── torch_judge/          # 评测引擎（题目定义 + 测试执行）
├── scripts/              # 构建与测试脚本
└── package.json          # 开发脚本（前后端并行启动）
```

---

## 📢 更新日志

- **[2026/05/01]** 扩展至 **101 道题**，覆盖 **22 个分类** — 从注意力机制到 SSM、MoE、图神经网络。🔥
- **[2026/04/28]** **八股文面试题** — 13 章、200+ 道理论题，覆盖 LLM 基础、RLHF、蒸馏、分布式训练。🔥
- **[2026/04/25]** **模拟面试模式** — 随机组合理论题和编程题，模拟真实面试。🔥
- **[2026/04/20]** 新增 GNN 学习路径 — 8 道题覆盖 GCN、GAT、GIN、MPNN、GraphSAGE、链接预测、图自编码器。🔥
- **[2026/04/20]** 全新 UI 设计，采用 OKLch 色彩系统、暗色模式和 Geist 字体。🔥
- **[2026/04/13]** 提交历史 — 查看每道题的所有历史提交记录。
- **[2026/04/10]** AI 辅助 — 可选的 AI 提示功能，支持任意 OpenAI 兼容 API。🔥
- **[2026/04/09]** 正式发布 🎉

---

## 🤝 参与贡献

- **出新题** — 在 `torch_judge/` 里添加题目定义和测试用例，提 PR
- **报 Bug** — [开 issue](https://github.com/BGWH123/Ember/issues)，附上复现步骤
- **修 Bug** — fork → 修复 → PR
- **改文档** — 错别字、表述不清、翻译，都欢迎

大改动建议先开 issue 聊聊思路。

---

## ⭐ Star History

![GitHub stars](https://img.shields.io/github/stars/BGWH123/Ember?style=social)

[![Star History Chart](https://api.star-history.com/svg?repos=BGWH123/Ember&type=Date)](https://star-history.com/#BGWH123/Ember&Date)

---

## 🙏 致谢

题库和评测引擎基于 [duoan](https://github.com/duoan) 的 [TorchCode](https://github.com/duoan/TorchCode)，MIT 协议。

前端 UI、学习路径及扩展功能参考了 [whwangovo](https://github.com/whwangovo) 的 [pyre-code](https://github.com/whwangovo/pyre-code)，感谢其出色的参考实现。

---

## 📄 许可证

MIT License，详见 [LICENSE](LICENSE)。
