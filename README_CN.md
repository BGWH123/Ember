[English](./README.md) | 中文

# Ember

从零实现现代 AI 系统的核心模块。Ember 是本地优先的桌面训练场：101 道 PyTorch 练习、学习路线、理论题库与模拟面试，基于 Tauri 2、React 和 Rust 核心。

浏览题库、参考答案和学习路线不需要 Python。需要执行代码时，再安装独立本地评测模块。

## 从源码启动

准备 Node.js 22+、Rust stable 和 [Tauri 桌面环境](docs/fern/pages/desktop/installation/install-desktop.mdx)。Windows 使用 MSVC Rust 工具链与 Visual Studio C++ Build Tools。

    npm ci
    npm run dev

Noto Sans、Noto Sans SC 和 Fira Code 随应用打包，离线使用。字体许可证位于 apps/desktop/public/licenses。

## 开发时使用本地评测器

Windows PowerShell：

    uv venv --python 3.12 .venv
    uv pip install --python .venv/Scripts/python.exe -r modules/evaluator/requirements.txt
    uv pip install --python .venv/Scripts/python.exe pytest
    $env:EMBER_DEV_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
    npm run dev

Linux/macOS 使用 .venv/bin/python，并按终端语法设置同名环境变量。源码解释器只用于 debug 构建；正式安装包使用独立签名评测模块。详见[源码开发](docs/fern/pages/developer-guide/development/build-from-source.mdx)。

## 检查与构建

    npm run check:core
    npm run test:python
    npm run build:ui
    cargo build --release -p ember-core --features developer-tools
    npm --prefix apps/desktop exec -- playwright install chromium
    npm --prefix apps/desktop test
    npm run build:windows

Windows 安装包输出到 target/release/bundle；macOS 使用 npm run build:macos。平台工具链与 CI 见[打包指南](docs/fern/pages/developer-guide/development/package-installers.mdx)。

## 目录与服务

| 目录 | 用途 |
| --- | --- |
| apps/desktop | React 界面和 Tauri 桌面壳 |
| crates/ember-core | 题库、SQLite 学习记录和 AI 请求 |
| crates/ember-evaluator | 本地评测协议、模块安装和签名 |
| modules/evaluator | Python 独立评测器 |
| content/catalog | 随应用分发的题库与参考答案 |
| torch_judge/tasks | 可执行任务定义 |
| scripts/desktop | 导出、构建、验收与性能工具 |

旧 Next.js 网页端和重复的 grading_service 后端已删除。桌面应用通过 Tauri IPC 调用 Rust 核心，评测通过本地子进程协议执行，不需要 HTTP 服务。

线上评测暂不可用。仅在 crates/ember-evaluator/src/remote.rs 保留一个不会发送请求的 HTTP 接口，并标注“TODO：线上评测预留功能”。Python 远程服务及其 Web 服务依赖均已移除，提交代码不会进入线上评测链路。

题库更新运行 npm run build:problems 和 npm run build:solutions；展示元数据保留在 content/catalog。原版学习记录的只读导入功能仍在设置中。

## 文档与贡献

[桌面文档](docs/README.md)包含安装、模块协议、架构和性能工具。运行 npm run docs:install、npm run docs:dev 本地预览，npm run docs:check 验证。

基于 [TorchCode](https://github.com/duoan/TorchCode)；原始版权声明保留于 [LICENSE](LICENSE)。MIT License。
