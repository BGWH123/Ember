---
# SPDX-License-Identifier: MIT
title: 排查问题
subtitle: 从可见错误定位环境、评测器和学习记录问题。
---

先保留完整错误文本，再检查问题发生在构建、安装、运行还是迁移阶段。反馈时附上应用版本、评测器版本、操作系统和最小复现代码。

## 桌面窗口没有出现

如果 Cargo 报告 `pkg-config`、GTK 或 WebKit 缺失，重新检查[平台依赖](../installation/install-desktop.mdx)。只运行核心测试不能证明原生桌面依赖已经就绪。若只打开了 Vite 的浏览器地址，Tauri IPC 不可用；请通过 Tauri 开发命令启动窗口。

## 无可用评测器

在设置中检查模式和模块状态。浏览题目可以成功，并不表示 Python 评测环境已经安装。先按[安装评测器](../evaluation/install-evaluator.mdx)准备模块，再运行题目。

## 公钥或签名错误

`No evaluator release public key configured` 表示应用构建时没有嵌入模块公钥，需要维护者重新构建。`Invalid module signature` 表示签名无法用受信公钥验证。检查清单来源、文件配对和应用版本；不要通过修改 payload、关闭校验或重写版本号处理。

## 协议或题库不兼容

应用和模块的 `protocolVersion`、`catalogVersion` 必须匹配，模块的 `target` 必须适用于本机。安装维护者为当前题库构建的模块；修改清单中的 hash 会破坏签名，也不能改变模块内的实际测试。

## 下载失败

检查清单和 ZIP 都使用受支持的 HTTPS 直链。当前下载器不跟随 HTTP 重定向，仓库网页地址不能代替 ZIP 地址。离线文件可用时，使用设置中的离线安装。

## Busy、超时或评测器崩溃

Busy 表示另一个评测或切换操作占用当前评测器，不会自动进入无限队列。等待当前操作结束。超时时检查源码循环和复杂度；崩溃时确认模块状态后再提交。源码问题与基础设施失败的区别见[协议结果](../../reference/evaluator/protocol.mdx)。

## 学习状态与预期不符

样例通过不会标记题目已解决；需要完整提交全部测试。导入只读取所选用户和已完成记录。若迁移数量不符，先检查旧库用户与来源路径，不要反复改名导入。

## 更新检查报错

开发构建未配置 updater 时，错误会直接显示在设置中。维护者需要配置[应用更新参数](../../reference/configuration.md)；用户不能靠重装评测模块修复桌面更新渠道。
