---
# SPDX-License-Identifier: MIT
title: 模块清单
subtitle: Ed25519签名封装、发布字段和归档边界。
---

清单使用外层签名封装加内层JSON字符串。签名覆盖 `payload` 的原始UTF-8字节，不是重新格式化后的对象。

## 外层封装

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `payload` | string | 内层清单的原始JSON字符串 |
| `signature` | string | Ed25519签名的十六进制表示 |

解析前先验证签名；任何空格、转义或字段顺序变化都可能改变被签名字节。签名验证成功后，安装器才按下面的字段检查兼容性。

## Payload字段

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `version` | string | 规范semver，作为不可变版本目录名 |
| `target` | string | 目标架构与OS，例如 `x86_64-linux` |
| `protocolVersion` | integer | 当前为1，必须与壳一致 |
| `catalogVersion` | string | catalog原始文件的SHA-256，必须与壳一致 |
| `url` | string | ZIP直链；HTTPS或回环开发HTTP，不跟随重定向 |
| `sha256` | string | ZIP内容的64位十六进制SHA-256 |
| `size` | integer | ZIP精确字节数，必须大于0且不超过2GiB |
| `entrypoint` | string | ZIP内可执行入口的安全相对路径 |
| `args` | string[] | 可省略，默认空数组；最多16项，每项最多4096 UTF-8字节 |

清单还受到64KiB下载上限约束。解压后的数据总量最多6GiB、条目最多100000；安装器拒绝绝对路径、路径越界与符号链接。入口路径不接受反斜线或冒号。

## 激活与回退

安装器验证归档大小与hash后写入版本目录。应用探测入口的真实 `hello`，成功后才原子替换 `active.json`。已安装目录内的 `module.json`保存签名清单，切换时再次读取与验证。

离线导入同样验证签名、大小、hash、平台与协议，不存在生产“跳过签名”开关。发布工具和签名流程见[打包模块](../../developer-guide/evaluator/package-and-release.mdx)。

源码：[ModuleManager](https://github.com/BGWH123/Ember/blob/main/crates/ember-evaluator/src/installer.rs)。
