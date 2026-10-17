---
# SPDX-License-Identifier: MIT
title: 兼容性
subtitle: 核对平台、协议、题库和独立版本的兼容条件。
---

桌面壳和评测模块使用独立版本号。安装模块时，版本号不同并不构成不兼容；协议、题库和平台不匹配则会被拒绝。

## 当前验收范围

| 范围 | 状态 |
| --- | --- |
| Linux x86_64 桌面 | 已在本机真实WebKit、Tauri IPC与release .deb链路验收 |
| Linux CPU 评测模块 | 已验证独立ZIP、签名安装、升级与回退 |
| Windows、macOS | 已提供MSI/NSIS、APP/DMG命令与手动原生打包CI；尚未完成真实机器验收 |
| 公共模块下载渠道 | 需要维护者配置；文档不提供未上线地址 |
| 桌面 updater | 已有构建与运行时配置路径；正式签名和服务端渠道需维护者配置 |
| GPU/CUDA性能 | 本轮未测；CPU版PyTorch结果不代表GPU能力 |

## 模块兼容矩阵

| 字段 | 检查方式 |
| --- | --- |
| `protocolVersion` | 必须等于壳支持的协议版本，当前为1 |
| `catalogVersion` | 必须等于壳加载的 `problems.json` 原始字节 SHA-256 |
| `target` | 必须等于Rust运行时的架构与OS，例如 `x86_64-linux` |
| 清单签名 | 必须由壳中受信的Ed25519公钥验证通过 |
| 运行时 `hello` | 实际进程再次报告匹配的协议与题库版本 |

打包脚本会检查任务测试与catalog的语法树一致。手动把清单改成壳期望的hash，不能使不匹配的执行代码兼容。

## 独立升级

`engineVersion` 与模块 `version`描述评测器发行版本，不要求与桌面版本相同。协议与题库保持兼容时，模块可以独立升级。题库变化后，需要重新构建壳资产与对应模块。

活动模块失配时，题库和草稿仍可使用，评测器标记为不可用。安装旧版应用与回退模块是不同操作；数据库不随模块回退。字段定义见[模块清单](evaluator/module-manifest.md)。

桌面安装包和Mac Universal构建流程见[打包安装程序](../developer-guide/development/package-installers.mdx)。
