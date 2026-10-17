---
# SPDX-License-Identifier: MIT
title: 设计与运行压测
subtitle: 用开放到达模型定位排队、历史查询和内存增长。
---

单用户串行对比反映一次交互的延迟，不能推导持续负载或突发到达下的行为。`ember-stress` 直接调用生产 Application，用独立到达线程发起并发请求，避免串行 headless 协议掩盖排队。

## 准备驱动

从仓库根目录构建开发工具，并安装采集依赖：

```bash
cargo build --release -p ember-core --features developer-tools --bin ember-stress
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python psutil
```

混合读取不需要 Python 评测环境；evaluate/fault 模式还需要[源码评测器依赖](../development/build-from-source.mdx)。驱动以4个 Tokio 工作线程运行，结果序列化后计入完成延迟。

## 固定工作负载

| 场景 | 到达方式 | 观察目标 |
| --- | --- | --- |
| mixed | 每秒500次，50%历史、20%目录、20%进度、10%草稿 | 读写锁、fsync与历史增长 |
| catalog | 每秒2000次目录请求 | JSON构造、分配和序列化 |
| evaluate | 总100次/秒，其中20%评测 | 准入拒绝与读取响应 |
| fault | 首次评测无限循环，其余继续到达 | 10秒超时、读写可用性和后续恢复 |
| burst | 每秒同时投递100次 evaluate 混合请求 | 同一时刻的锁竞争和 Busy 尾延迟 |

历史夹具在计时前创建，其中1%属于被查询题目。夹具使用短源码；结果不能代表100份64KB源码同时返回的上限场景。

```bash
.venv/bin/python scripts/desktop/stress.py --mode mixed --history 100000 --rate 500 --seconds 300 --output artifacts/desktop/stress/mixed-new
.venv/bin/python scripts/desktop/stress.py --mode evaluate --arrival burst --history 100000 --rate 100 --seconds 30 --output artifacts/desktop/stress/burst-new
.venv/bin/python scripts/desktop/stress.py --mode fault --history 100000 --rate 100 --seconds 30 --output artifacts/desktop/stress/fault-new
```

## 统计规则

延迟从计划到达时刻开始，包含排队。驱动最多保留256个在途请求；超过上限或发生器落后超过100ms，计为 `generatorDropped`。Busy 是业务拒绝，两者分开报告，不算成功评测。

每轮保存 `rows.jsonl`、`resources.json`、`summary.json`。比较时固定题库、数据规模、到达率、编译配置和预热策略，用 A/B/B/A 顺序交叉运行。合并原始样本计算 P95/P99，不平均各轮百分位。将采样定位与无 profiler 的验收分开，运行期间不要并行构建。

外部 psutil 每200ms采集进程树 RSS/PSS，包含驱动和包装器，不包含 WebView。短命 worker 的瞬时峰值可能落在两次采样之间。GNU time 的 CPU 回收链对被终止的 forkserver 孙进程不保证完整，评测模式的累计 CPU 只能作为下界。

## 已验证的退化

本机 i5-12490F / WSL2 Linux、十万条历史的开发快照中，补索引前后两轮混合负载的历史 P99 为2079.94 → 7.49ms，丢弃13688/30000 → 0/30000。分离 WAL 读写连接后，两轮集中突发的历史 P99 为139.41 → 18.27ms。单任务评测策略与 FULL 同步保持不变。

索引版本五分钟完成150000次请求，零丢弃；最终读写分离版本另做60秒、30000次混合请求复测，历史 P99 为0.88ms，采样峰值 PSS 约11.40MiB。这些是后台驱动的本机结果，不能当成完整桌面内存或其他机器的性能保证。

完整桌面另一次原生验收的 PSS 快照约为主页228MiB、编辑器与Torch评测后577MiB，未做完整UI对原版浏览器的内存 A/B。复现现象后，再按[采集样本](profiling.mdx)定位机制。
