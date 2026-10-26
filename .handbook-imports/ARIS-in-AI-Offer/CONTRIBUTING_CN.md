# 给 ARIS-in-AI-Offer 贡献

[English](CONTRIBUTING.md) | 中文版

谢谢你愿意看到这里 🌱 —— 这个仓库存在的初衷就是让 2026 年 AI 秋招稍微轻松一点，也给以后的同学留个底子。一个人的力量有限，PR 和 issue 都欢迎。

## 你可以怎么贡献

- **加一篇新 cheat sheet**：当前 collection 没覆盖的方向（Diffusion Post-Training、Flow-DPO/OPD、Audio Gen、LLM-as-Judge、Long-Video Diffusion …）
- **更新现有 cheat sheet**：前沿动了（SWE-bench 新 SOTA、MCP spec 新 revision 等）
- **修数学 / 代码 / 引用 bug**：每篇都有跨模型 `codex GPT-5.5 xhigh` 审查 trail 在 `*.review.json` 里，但 reviewer 也会漏
- **改进 render / 排版**：`tools/render_html.py` 和 `tools/templates/` 都在这个仓库里，直接 PR
- **翻译**：当前大部分教程是中文 + 英文术语原文混排，可以加纯英文版

## 怎么加新 cheat sheet —— ARIS workflow

仓库里所有教程都是用 ARIS 的 `/interview-cheatsheet` + `/render-html` 两个 skill 生成的。为了保持一致，**请也用这套 workflow**：

1. **安装 ARIS**（skill 源码在 `skills/`；canonical 来源是 [ARIS 主仓库](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)）。Claude Code 用户：`bash <(curl -s https://raw.githubusercontent.com/wanshuiyin/Auto-claude-code-research-in-sleep/main/tools/install_aris.sh)`
2. **在本仓库目录里调** `/interview-cheatsheet "<TOPIC>"`。skill 自动跑跨模型 codex review gate + render-html。
3. **自己 review** 生成的 `.md` / `.html` / `.review.json` 再 push。
4. **开 PR**：三个文件都带上（MD 源 / HTML 渲染 / review JSON），README.md / README_CN.md 对应分类下加一行。

不想用 ARIS 手写也行 —— 照着现有教程的结构写（例如 `docs/tutorials/attention_tutorial.md`），跑 `tools/render_html.py` 渲染就行。

## 风格指南（严格）

这些规则是前 16 篇 review-loop 踩出来的：

| 规则 | 原因 | 怎么办 |
|---|---|---|
| 标题：`## §N Title`（§N 后面留空格） | 老版本 `§NTitle` 粘连 | 始终留空格 |
| 表格里的数学：`\lvert x \rvert` 不要 `\|x\|` | 表格里的 `\|` 会被当作单元格分隔 | 用 LaTeX 显式写法 |
| 数学里的 `<` 用 `\lt`，`>` 用 `\gt` | 原生 `<t` 会被 parser 当 HTML 起始 tag，整段坏 | 用 `\lt` / `\gt` 转义 |
| Callout (`> 💡 / ⚠️ / ✅ / ❌`) 后接 list：**中间留空行** | 否则 renderer 会把 list 吃进 callout | 强制空行分隔 |
| 章节：§0 TL;DR · §1 直觉 · §2-§9 公式+代码+变体 · §10 25 高频题（L1 必会 × 10 / L2 进阶 × 10 / L3 顶级 lab × 5） · §A 可选附录 | 匹配 ARIS pilot pattern | 按编号结构走 |
| 不放个人信息 | 隐私 + 通用性 | byline 走 `--author` flag 在 render 时加；ban：SJTU JHC / Server5 / job market / `/Users/...` / 具体实验室 |

完整契约见 `skills/render-html/SKILL.md` 和 `skills/interview-cheatsheet/SKILL.md`。

## PR 流程

1. 改动要文档化（带上 `.review.json` 让 reviewer 看你的 audit trail）
2. 加新教程要 **同时** 更新 `README.md` 和 `README_CN.md`，按分类放（General / Post-Training & Reasoning / LLM Architecture & Systems / Generative Models — Theory & Tokenizers / Generation Systems / Multimodal / Agents）
3. PR 聚焦单一主题 —— 一篇教程一个 PR
4. Commit 信息清晰 —— 规范是 `docs(tutorials): add <Topic> cheat sheet (rendered via /render-html)`

### PR checklist

- [ ] `.md` 源符合风格指南
- [ ] `.html` 是用 `tools/render_html.py` 重新渲染的
- [ ] `.review.json` 包含跨模型 codex audit trail（math/code review verdict + render review verdict + thread IDs）
- [ ] byline 之外没个人信息
- [ ] `README.md` + `README_CN.md` 都更新了
- [ ] 引用真实可验证（不能 hallucinate 作者 / 年份 / venue）

## 坦白说

现有教程是尽力而为，但领域更新太快。如果你发现：
- 作者 / 年份 / venue 标错了
- 数学推导漏一项
- 代码实际跑不通
- benchmark / SOTA 已经过时
- 或者就是写得不清楚

…… 请提 issue 或 PR，一起修。Reviewer（含 codex 跨模型审）也会漏，社区的眼睛能补上。

## 行为准则

- 礼貌 + 建设性 —— 假设对方善意
- 反馈对事不对人
- 帮别人学 —— 这首先是个学习资源

## 有问题？

开 issue，或者去 [ARIS 主仓库](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) 找微信/Discord 群（二维码 + 链接在那个仓库的 README）。

## License

提交贡献即默认你接受 MIT license —— 和仓库其它部分一致。
