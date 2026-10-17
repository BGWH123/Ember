# Ember：20 人容量与时延诊断

默认目标是 20 人在线，正式负载约 20 分钟。脚本分开统计读接口、异步入队、状态轮询、排队、进程启动、代码加载、测试和持久化。第 14 分钟还会同时发起 20 个完整评测，验证并发提交；202、429、超时和错误判题均单独计数。

## 运行

需要 Bash、Python 3.10+ 和 Docker，或本机 k6 1.x。应用服务要求 Linux / Python 3.11+。默认使用固定镜像 `grafana/k6:1.6.0`，Docker 发生器使用 Linux host 网络。

```bash
docker compose -p ember-loadtest up --build -d

# 约 45 秒；加上 20 个同时提交，先验证四类正确代码和并发链路。
PROFILE=smoke BURST_USERS=20 OBSERVE_COMPOSE_PROJECT=ember-loadtest \
bash scripts/loadtest/run.sh

# 正式负载 20 分钟，另有正确性预检和 HTTP 收尾。
USERS=20 OBSERVE_COMPOSE_PROJECT=ember-loadtest \
bash scripts/loadtest/run.sh
```

已有本地服务也可以使用。前端用生产构建，后端关闭 `--reload`；压测会创建匿名用户和提交，使用独立测试数据库。

```bash
TARGET_URL=http://127.0.0.1:3000 \
OBSERVE_DB_PATH=/absolute/path/to/test.db \
OBSERVE_PIDS=1234,5678 \
bash scripts/loadtest/run.sh
```

脚本不自动启动目标服务，不访问外部 AI 推理接口。预检会逐一验证 ReLU、交叉熵、LayerNorm、Attention 的完整评测，失败就停止。`SUITE=read` 只测阅读，`SUITE=judge` 只测评测并保留恢复探针。Ctrl+C 清理本轮发生器与观察进程；已经入队的任务继续按服务规则执行。

## 负载假设

20 人在线不等于 20 人持续发请求。默认每人平均 20 秒读一次、120 秒评测一次，即 **1 次读请求/秒、10 次评测/分钟**。70% 是网页同样的前两个样例测试，30% 跑完整测试；代码包含网页模板的合法导入。读请求覆盖题目列表、详情、进度、历史、HTML 页面。

发生器采用开放到达率模型，服务变慢仍继续按计划发起请求，无法启动的迭代记作 `dropped_iterations`。20 个普通会话复用固定身份；突发测试使用另一个 20 人测试组，并继续背景流量。k6 VU 是发生器执行资源，不等于在线人数。

| 阶段 | 时间 | 读请求/秒 | 评测/分钟（起始 → 结束） | 用途 |
|---|---|---:|---:|---|
| warmup | 0–2 分钟 | 1 | 1 → 1 | 预热 |
| baseline | 2–5 分钟 | 1 | 1 → 3 | 低负载参考 |
| ramp | 5–8 分钟 | 1 | 3 → 10 | 升载 |
| steady | 8–14 分钟 | 1 | 10 → 10 | 目标负载 |
| spike | 14–16 分钟 | 1 → 2 | 10 → 20 | 加倍负载，起点另发 20 个同时提交 |
| recovery | 16–18 分钟 | 2 → 1 | 20 → 1 | 恢复 |
| drain | 18–20 分钟 | 1 | 停止新增评测 | 排空 |

执行器到达率会取整且至少为 1，因而 20 人配置的早期阅读负载没有继续降低。轮询另外计入 HTTP 流量：默认每个等待任务每秒约 4 次。压测保持 250ms 轮询和 30s 客户端等待；网页前 5s 用 250ms，之后改为 1s，最多等待 5 分钟，后台任务不会被客户端超时取消。

## 计时点

后端每个请求返回 `Server-Timing` 和 `X-Request-ID`；Next.js 保留它们，并补充自身计时。浏览器开发者工具和 k6 都能读取。日志以 JSON 记录 `http.completed`、`judge.completed`，不写 session token 或用户代码。任务结果的 `timings` 与 `resources` 保存到 SQLite。

| 位置 | 字段 | 口径 |
|---|---|---|
| HTTP | `backend_ms` / `http_total_ms` | 请求进入到响应头 / 完整响应发送；后者在日志里 |
| HTTP | `dispatch_ms` | 请求参数校验与进入同步处理线程前的时间 |
| 会话 | `session_ms` | 身份查找/创建；已有 GET 会话每分钟才更新一次活动时间 |
| SQLite | `db_connect_ms` / `db_lock_ms` / `db_sql_ms` / `db_commit_ms` / `db_close_ms` | 建连接与 PRAGMA、BEGIN 等待、语句执行、COMMIT、关闭连接 |
| 业务 | `admission_ms` / `result_read_ms` / `progress_read_ms` / `history_read_ms` | 各服务操作的完整时间 |
| Web | `proxy_upstream` / `proxy_overhead` / `web` | fetch 与读取上游响应、自身处理、路由处理总时间 |
| 调度 | `queue_wait_ms` / `claim_ms` | 入队到状态转为 running / 领取事务；领取 DB 字段带 `claim_` 前缀 |
| 子进程 | `process_launch_ms` / `worker_startup_ms` / `runtime_import_ms` | start() 调用、父进程开始到子进程入口、入口后的运行时导入 |
| 评测 | `setup_ms` / `parse_compile_ms` / `user_code_ms` / `test_execution_ms` | 配置与准备、AST/编译、顶层用户代码、全部选中测试；每个测试另有 `execTimeMs` |
| IPC | `result_encode_ms` / `result_transfer_ms` / `result_decode_ms` / `process_reap_ms` | 编码、发送后接收等待、解码、进程回收 |
| 评测总计 | `worker_total_ms`（兼容 `total_ms`） | 父进程开始启动到结果接收和子进程回收，成功/失败采用同一口径 |
| 落库 | `persist_ms` / `persist_db_lock_ms` / `persist_db_commit_ms` | 主结果事务完整时间，包含 COMMIT；不含随后保存计时元数据的小事务 |
| 完成 | `server_e2e_ms` / `job_e2e_ms` | 入队到服务端完成提交 / 客户端开始 POST 到轮询看到终态 |

时间保留小数毫秒。span 有包含关系，不应相加各项 P95。SQLite SQL span 计 execute()，行读取/映射的剩余成本在业务 span 里；transfer 是发送标记到父进程可读，不代表纯网络时延。计时元数据在结果 COMMIT 后更新，极短窗口内首次查询可能尚未带完整落库计时；JSON 日志保存完整计时。

`admission_cpu_ms`、`result_read_cpu_ms`、`progress_read_cpu_ms`、`history_read_cpu_ms` 和 `db_*_cpu_ms` 用当前线程 CPU 时钟记录执行成本，不计锁等待与磁盘等待；它们是评估语言收益的补充，不能当成纯 Python CPU 时间，因为也包含 SQLite 等原生库。

资源字段记录解释器/Torch 版本、Torch 线程数、启动方法、运行时 VMS/RSS 基线、任务峰值 RSS 和子进程 CPU 毫秒。额外的 `resources.jsonl` 每 5 秒记录选定容器 cgroup CPU/节流、内存、真实队列与错误分布；采样会漏掉很短的排队尖峰，应同时看逐任务 queue_wait。容器 CPU 包含注入观察程序的开销，PID 采样无法完整累计已经退出的短命子进程。

## 已修复的链路问题

- 正确代码被 1GiB 绝对虚拟地址上限误杀：使用独立预热 forkserver，Torch/BLAS 单线程；以总 RSS 1GiB 为预算，每 10ms 检查，辅以按剩余 RSS 预算计算的虚拟地址增长限制和容器 4GiB 总上限。没有取消内存限制。
- 网页合法导入被拒绝：允许 torch、torch.nn、torch.nn.functional、numpy、math、typing，保留受限导入规则。
- 先 join 再收结果可能阻塞：运行时读取 IPC，再回收进程；异常、EOF、超时均形成终态，单次故障不会杀死调度线程。
- 输出先无限累积、序列化 JSON 被直接切断：stdout/stderr 共用 16KB 字节预算，保存完整 JSON。
- 纯查询抢写锁、每次轮询写会话、新会话与提交分两次 fsync：读 WAL 快照、活动时间节流、首次会话和入队合并为一个事务；进程内写入锁减少 SQLite 各连接的 busy 重试。保持 SQLite FULL 同步级别。
- 请求结束后数据库全部断开会重复 checkpoint/清理 WAL：应用生命周期保留一个空闲连接，不持有读事务，也不关闭自动 checkpoint。[SQLite WAL 说明](https://www.sqlite.org/wal.html#the_wal_file)
- 容器控制台入口在 forkserver 子进程里重复导入 Uvicorn：Docker 改用 `python -m uvicorn`，保留预热进程和每次评测的独立子进程。
- 进程内通知丢失/重启遗留 running：数据库是队列来源，重启重新领取未完成任务；独占文件锁防止两个本地调度器重复恢复。`/health` 检查调度线程。
- Web 覆盖幂等 key、查询错误伪装为空成功、跨用户读提交：保留幂等 key、返回真实状态并限定提交所有者。重复 key 改变请求会返回 409。
- 空测试/提交只跑部分用例也能算通过：入口拒绝非法索引；正式提交必须跑全部测试。
- SQL 占位符、Python 3.11 的 GQA f-string、排队和落库计时错误已修复。

本地调度仍限制一个 API 进程/数据库，默认租户在途上限 32、每用户在途上限 1。不要增加 `uvicorn --workers`；独立队列与多调度器租约是后续扩容改造。RSS 检查是采样，代码执行也仍需生产环境的独立权限/网络隔离；当前进程限制与 allowlist 不构成面向恶意代码的安全沙箱。

## 产物与验收

默认保存到 `artifacts/loadtest/<run-id>/`，不进 Git：`report.md`、可直接打开的 `dashboard.html`、`summary.json`、`config.json`、`execution.json`、`samples.json.gz`、`resources.jsonl`、`context.json` 和日志。上下文记录 commit、dirty diff、应用/脚本 SHA256、版本与资源范围；`worktree.patch` 保留已跟踪文件的本轮改动。远程目标的资源采样应在目标服务器独立运行再合并报告。

默认门槛：读 P95<200ms/P99<500ms、入队 P95<300ms、轮询 P95<200ms、任务 E2E P95<5s、异常率<1%、429<5%、已准入任务客户端成功率/终态正确率>99%、发生器零丢弃。稳态单独验收；突发 20 个任务必须全部准入且正确完成。返回码 0 表示通过，99 表示阈值失败，其他非零表示预检/配置/执行失败。E2E 只涵盖看到终态的任务，必须同时检查超时、拒绝和排空。

| 参数 | 默认 | 用途 |
|---|---:|---|
| `USERS` | 20 | 普通会话池，派生目标到达率 |
| `BURST_USERS` | 正式 20 / smoke 0 | 额外同时提交数；0 关闭 |
| `READ_RPS` / `JOBS_PER_MINUTE` | 1 / 10 | 覆盖行为假设 |
| `READ_PRE_VUS` / `READ_MAX_VUS` | 5 / 30 | 阅读发生器预算 |
| `JUDGE_PRE_VUS` / `JUDGE_MAX_VUS` | 正式 20 / 40 | 提交发生器预算；smoke 预分配 5 |
| `POLL_INTERVAL_MS` / `JOB_TIMEOUT_SECONDS` | 250 / 30 | 观察轮询与等待上限 |
| `PROFILE` / `MODE` / `SUITE` | capacity / web / all | smoke、grading、read/judge 可单独定位 |
| `OBSERVE_COMPOSE_PROJECT` | 空 | 只采样指定项目的容器 |
| `OBSERVE_DB_PATH` / `OBSERVE_PIDS` | 空 | 本地 DB 与指定进程及后代 |

## 2026-10-04 实测

已实际运行完整 20 分钟，预设门槛全部通过。后端容器限制 3 核 / 4GiB，使用 Python 3.11.17、PyTorch 2.14.1+cpu、SQLite FULL 同步；Next.js 15.5.14 生产服务和 k6 在同一主机运行。工作负载为上述四类题目，普通负载之外包含 20 人同时完整提交。

| 指标 | 实测 |
|---|---:|
| 提交 / 准入 / 正确完成 | 158 / 158 / 158 |
| 拒绝 / 异常 / 客户端超时 / 丢弃 | 全部为 0 |
| 稳态读请求 P95 / P99 | 15.12 / 19.70ms |
| 稳态入队 P95 | 24.35ms |
| 稳态客户端完成 P95 | 291.05ms |
| 20 人同时完整提交，正确完成 | 20 / 20 |
| 同时提交客户端完成 P95 / P99 | 1222.90 / 1236.58ms |
| 全程入队业务线程 CPU P95 | 1.70ms |
| 全程评测启动 / 评测总计 P95 | 7.25 / 37.90ms |
| 后端容器累计 CPU / 节流时间 | 25.63s / 0s |

稳态阶段持续 6 分钟，有 60 次评测和 360 次读请求。服务端完成时间中位数为 40.75ms，客户端观察到完成的中位数为 281ms；两者分别统计，不能直接相减估算每个任务的开销。全程入队写锁 P95 为 96.41ms、COMMIT P95 为 12.49ms；API 业务 CPU 明显小于这些等待时间。

这些结果支持先保留 Python，20 人容量暂不需要为降低时延重写 Go / Rust。优先验证更快的结果通知、SQLite 写入策略与评测并行度；仍需保持评测异步和进程隔离。没有实测 Go / Rust 的同业务实现，不能据此给出语言加速倍数。

另做相同 45 秒 smoke 工作负载的轮询间隔对照，两轮均 30 次提交全部正确。250ms → 50ms 后，稳态客户端完成 P95 从 278.70ms 降到 83.40ms，但突发完成 P95 从 1258.75ms 升到 1448.30ms，轮询次数从 95 次增至 309 次。每轮稳态仅 4 个任务且没有重复试验，不能外推 P99 或语言收益；结果提示应验证完成通知或自适应轮询，而不是全局缩短间隔。对照报告分别在 `validation-20-final-smoke/` 与 `validation-20-poll-50ms/`。

完整证据在 [`artifacts/loadtest/capacity-20-20261004/report.md`](../../artifacts/loadtest/capacity-20-20261004/report.md)，同目录保留原始样本、仪表盘、后端日志与回归日志。数据库的 162 个正确结果包括 4 次预检，不应当成 162 次正式负载提交。末次资源采样记录了一次发生器退出与 `docker stats` 之间的竞态，目标后端采样正常；已修复该观察程序竞态并加入回归测试，原始记录保留。

压测镜像固定为该次成功链路版本。压测结束后另修复用户 `SystemExit` 的终态映射，以及错误结果发送后子进程提前被杀的问题；这两处异常分支和采样竞态通过回归测试，不属于上述 20 分钟镜像的覆盖范围。四类题目、单机测试只说明本轮容量与时延，完整题库、浏览器体验与上线隔离需要分别验收。

## Go / Rust 决策

先看慢在哪一段，再做相同硬件、题目、数据和到达率的多轮对照。SQL 写锁、磁盘 COMMIT、评测串行排队、子进程创建和轮询间隔都应分别判断。换 API 语言仍然要付数据库和 Python/PyTorch 评测的成本；不能用一个空 HTTP 接口的 QPS 推断整体收益。

如果稳态 API 的分发、编码、业务 CPU 时间很小，优先保留 FastAPI 与现有 React/Next.js 界面；若 SQL/COMMIT 主导，先验证存储与事务设计；若 worker 主导，先验证任务模型和独立 worker 并行度。只有 API CPU 在相同负载下持续饱和且排除了这些因素，再为 Go 或 Rust 做保持业务语义与正确性门槛的 A/B。20 分钟与四类题目只是容量基线，不代表全题库或上线就绪。

直连后端或轮询间隔对照：

```bash
docker compose -p ember-loadtest -f docker-compose.yml \
  -f scripts/loadtest/compose.override.yml up --build -d
MODE=grading SUITE=judge USERS=20 OBSERVE_COMPOSE_PROJECT=ember-loadtest \
bash scripts/loadtest/run.sh
PROFILE=smoke BURST_USERS=20 POLL_INTERVAL_MS=50 \
OBSERVE_COMPOSE_PROJECT=ember-loadtest bash scripts/loadtest/run.sh
```

测试覆盖真实 SQLite WAL 并发、同时创建会话、幂等/权限、错误恢复、重启、真实子进程默认内存预算/超时/大输出，以及模拟接口上的真实 k6 429、异常、超时、错判计数与报告导出。

```bash
python -m pytest grading_service/test_grading_service.py scripts/loadtest/test_loadtest.py -q
EMBER_TEST_K6=1 python -m pytest grading_service/test_grading_service.py scripts/loadtest/test_loadtest.py -q
```

参考：[Python 进程启动方式](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)、[资源限制](https://docs.python.org/3/library/resource.html)、[SQLite 事务](https://www.sqlite.org/lang_transaction.html)、[k6 开放到达率](https://grafana.com/docs/k6/latest/using-k6/scenarios/executors/ramping-arrival-rate/)。
