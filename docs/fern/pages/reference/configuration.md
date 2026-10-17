---
# SPDX-License-Identifier: MIT
title: 开发与发布配置
subtitle: 查找桌面构建、评测进程和开发工具读取的环境变量。
---

本页覆盖桌面与本地评测配置。应用设置中的本地模块选择与构建时嵌入的信任配置作用不同。

## 桌面构建与运行

| 变量 | 读取阶段 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `EMBER_EVALUATOR_PUBLIC_KEY` | 桌面Rust构建 | 未配置 | 32字节Ed25519公钥的64位十六进制字符串；缺失时不能安装签名模块 |
| `EMBER_DEV_PYTHON` | debug桌面启动 | 未配置 | 开发解释器绝对路径，直接使用仓库源码评测器；release桌面不读取 |
| `EMBER_SHELL_UPDATE_URL` | `build-release.mjs` | 必填 | HTTPS updater端点，允许Tauri目标/架构/版本模板 |
| `EMBER_SHELL_UPDATE_PUBLIC_KEY` | `build-release.mjs` | 必填 | Tauri updater公钥，格式不同于模块公钥 |
| `TAURI_SIGNING_PRIVATE_KEY` | Tauri发行构建 | 必填 | Tauri签名私钥，按Tauri工具链格式传入 |

`build-release.mjs`还要求模块公钥，生成独立的updater配置覆盖层。未配置updater的开发构建可以启动，但检查更新会返回配置错误。不要将模块Ed25519密钥与updater密钥混用。

## 评测器进程

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `EMBER_CATALOG_PATH` | 打包hash文件或仓库catalog | Python握手计算题库hash时使用的JSON路径，主要用于开发联调 |
| `OMP_NUM_THREADS` | Rust启动器设为1 | 限制本地评测器的OpenMP线程数 |
| `MKL_NUM_THREADS` | Rust启动器设为1 | 限制本地评测器的MKL线程数 |

`runner.py`负责设置Python模块路径；正式自包含模块使用清单中的 `-I runner.py` 入口。修改环境变量不会绕过壳侧兼容性检查。

## 开发工具

| 变量 | 使用范围 | 默认值 |
| --- | --- | --- |
| `EMBER_ROOT` | headless/stress仓库根路径 | 必填 |
| `EMBER_DATA` | headless/stress独立数据目录 | 必填 |
| `EMBER_DEV_PYTHON` | headless/stress解释器路径 | 评测模式需要 |
| `EMBER_EVALUATOR_PUBLIC_KEY` | headless模块安装检查 | 可选，运行时读取 |
| `STRESS_MODE` | Rust驱动模式 | `mixed` |
| `STRESS_ARRIVAL` | 到达方式 | 非`burst`值按固定间隔处理 |
| `STRESS_RATE` | 每秒计划到达 | `500` |
| `STRESS_SECONDS` | 到达窗口秒数 | `30` |
| `STRESS_HISTORY` | 计时前历史夹具条数 | `10000` |

优先使用[压测脚本](../developer-guide/testing/performance.md)的参数，它负责创建临时数据目录、设置环境和回收测试进程。`--profiler`只有显式选择后才向自己的测试程序传入采样配置。

## 设置页

| 设置 | 持久性 | 约束 |
| --- | --- | --- |
| 本地活动模块 | 保存到app-data | 安装与切换都须验证兼容性与握手 |
| 学习记录 | 本地SQLite | 不自动同步到其他设备 |

实现入口：[桌面初始化](https://github.com/BGWH123/Ember/blob/main/apps/desktop/src-tauri/src/main.rs)、[发行脚本](https://github.com/BGWH123/Ember/blob/main/scripts/desktop/build-release.mjs)。
