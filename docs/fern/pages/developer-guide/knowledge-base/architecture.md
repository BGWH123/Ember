---
# SPDX-License-Identifier: MIT
title: 系统架构
subtitle: 理解桌面界面、Rust 核心、存储和独立评测器之间的调用关系。
---

Ember 把学习数据与执行环境分开管理。Rust 核心拥有题库、学习记录和模块选择权；评测器只接收代码与测试选择，返回结果，不直接写入业务数据库。

## 请求路径

```mermaid
flowchart TB
  UI[React 界面] --> IPC[Tauri 命令]
  IPC --> App[Rust Application]
  App --> Catalog[静态题库]
  App --> DB[SQLite 学习数据]
  App --> Port[Evaluator 接口]
  Port --> Local[本地协议进程]
  Port --> Remote[远程 HTTP 服务]
  Local --> Worker[单次执行进程]
```

浏览题目走题库路径，读取进度和历史走 SQLite 路径；两者都不需要启动评测器。提交代码经过输入校验、单任务准入和持久化，再调用所选评测器。结果回到 Rust 后，提交结果与学习进度在同一事务中更新。

## 模块职责

| 路径 | 职责 |
| --- | --- |
| `apps/desktop` | 路由、编辑器、设置与 Tauri 壳 |
| `crates/ember-core` | 题库、应用用例、SQLite 和 AI 请求适配 |
| `crates/ember-evaluator` | 协议、本地/远程适配器、模块安装与验证 |
| `modules/evaluator` | Python 协议进程、预算检查和单次执行 |
| `content/catalog` | 随壳打包的展示与测试元数据 |
| `torch_judge/tasks` | 执行任务与测试的源码 |
| `scripts/desktop` | 导出、构建、签名、验收和性能工具 |

原版 `web` 和 `grading_service` 仍保留，用于兼容开发与对比；它们不是新桌面应用的请求路径。

## 依赖方向

`ember-core` 依赖评测接口，接口不依赖前端或业务数据库。前端通过 Tauri 命令调用应用用例，不能直接启动 Python 或修改 SQLite。模块管理器负责下载与签名校验，执行适配器负责协议和进程生命周期，两者不互相承担学习状态更新。

这种边界允许模块与壳独立升级，也允许远程服务实现同一[评测协议](../../reference/evaluator/protocol.mdx)。独立升级仍须满足[兼容性约束](../../reference/compatibility.md)。

## 一致性边界

一次只接纳一个评测任务。评测与模块切换共用准入锁，避免运行途中换掉执行环境。基础设施失败留下可诊断的提交记录，但不改变学习进度。具体规则见[存储](storage.md)与[评测器生命周期](evaluator-lifecycle.md)。

实现入口：[Application](https://github.com/BGWH123/Ember/blob/main/crates/ember-core/src/application.rs)与[Tauri 命令](https://github.com/BGWH123/Ember/blob/main/apps/desktop/src-tauri/src/main.rs)。
