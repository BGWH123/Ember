---
# SPDX-License-Identifier: MIT
title: 参与贡献
subtitle: 按模块验证行为，并更新与改动对应的公开文档。
---

先选择改动所在的边界：桌面交互、Rust业务、评测传输、Python执行或题库。用最小路径验证行为，再扩大到跨模块验收。

## 选择入口

| 改动 | 先阅读 | 主要检查 |
| --- | --- | --- |
| 界面与编辑器 | [源码开发](../../developer-guide/development/build-from-source.mdx) | UI构建、Playwright及必要的原生IPC验收 |
| SQLite与进度 | [存储](../../developer-guide/knowledge-base/storage.md) | Rust事务、恢复和兼容回归 |
| 本地/远程评测 | [评测协议](../../reference/evaluator/protocol.mdx) | Python协议、模块安装与运行时兼容 |
| 性能改动 | [压力测试](../../developer-guide/testing/performance.md) | 同负载A/B、尾延迟、拒绝和内存 |
| 文档 | [本地预览](documentation/local-preview.mdx) | 导航、链接、MDX、实际页面 |

## 保持契约

修改结果或模块字段时，同步Rust类型、Python适配器、界面读取方和参考页。协议字段改变不等于模块版本号更新；是否需要协议升级，应依据现有消费者能否兼容解释。

修改题目测试时重新导出catalog，并在打包模块前核对执行测试一致性。不要通过伪造hash掩盖旧模块与新题库的差异。

## 提交可复核的证据

说明具体触发条件、改动后的行为与运行过的检查。性能结果写清数据规模、到达方式、成功/拒绝/丢弃和统计范围，不只给一个最快值。文档中的安装命令、字段与限制应能从同次改动的源码核对。

公开文档使用[写作规范](documentation/style-guide.md)。内部设计过程与本机原始数据继续保留在项目wiki或ignored产物目录，不把它们变成读者完成任务的前置依赖。
