# Ember 数据库模式契约

## 1. 契约状态

本文档是 Ember v0 的数据库模式初步契约，供迁移脚本、仓储层、API 用例、评测调度器和测试共同使用。实现可以先使用 SQLite，服务化部署可以迁移到 PostgreSQL，但业务语义、租户隔离和状态约束不得改变。

字段名统一使用 `snake_case`。API 层可以转换为现有前端使用的 `camelCase`。所有时间字段使用 UTC 存储，推荐使用带时区的数据库类型；SQLite 模式使用 ISO 8601 UTC 字符串。

## 2. 核心实体关系

```text
tenant
  ├── membership ── user
  ├── progress ─── problem_version
  ├── submission ── user
  │       └── judge_job
  │               └── judge_result
  └── quota_policy
```

内容题目本身可以由版本化构建产物提供，但所有用户产生的数据必须带有 `tenant_id`。任何查询都不得只依赖前端传入的 user_id 来完成租户隔离。

## 3. 表和字段

### 3.1 tenant

租户代表个人工作区、组织、课程或团队。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | UUID/TEXT | PK | 租户标识 |
| slug | TEXT | UNIQUE, NOT NULL | URL 和日志中的稳定标识 |
| name | TEXT | NOT NULL | 租户名称 |
| status | TEXT | NOT NULL | `active`、`suspended`、`deleted` |
| plan | TEXT | NOT NULL | 配额策略名称 |
| created_at | TIMESTAMP | NOT NULL | 创建时间 |
| updated_at | TIMESTAMP | NOT NULL | 更新时间 |

### 3.2 user

用户是跨租户的身份实体。本地匿名会话也必须映射到 user。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | UUID/TEXT | PK | 用户标识 |
| external_subject | TEXT | NULLABLE | 外部登录系统的 subject |
| display_name | TEXT | NULLABLE | 展示名 |
| status | TEXT | NOT NULL | `active`、`disabled`、`deleted` |
| created_at | TIMESTAMP | NOT NULL | 创建时间 |
| updated_at | TIMESTAMP | NOT NULL | 更新时间 |

`external_subject` 如果存在，应与身份提供方名称组成唯一约束。不得把 session token 直接作为 user_id。

### 3.3 membership

记录用户与租户的关系。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| tenant_id | UUID/TEXT | PK/FK | 租户 |
| user_id | UUID/TEXT | PK/FK | 用户 |
| role | TEXT | NOT NULL | `owner`、`admin`、`member`、`viewer` |
| status | TEXT | NOT NULL | `active`、`invited`、`revoked` |
| created_at | TIMESTAMP | NOT NULL | 加入时间 |
| updated_at | TIMESTAMP | NOT NULL | 更新时间 |

主键为 `(tenant_id, user_id)`。所有业务请求必须先获得 active membership。

### 3.4 session

保存本地会话或服务端 session 映射，不把明文 token 写入业务日志。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | UUID/TEXT | PK | session 标识 |
| token_hash | TEXT | UNIQUE, NOT NULL | token 哈希 |
| tenant_id | UUID/TEXT | FK, NOT NULL | 当前租户 |
| user_id | UUID/TEXT | FK, NOT NULL | 当前用户 |
| expires_at | TIMESTAMP | NOT NULL | 过期时间 |
| created_at | TIMESTAMP | NOT NULL | 创建时间 |
| last_seen_at | TIMESTAMP | NOT NULL | 最近使用时间 |

### 3.5 problem_version

记录题目构建产物的可引用版本，不保存不必要的大型题目正文时，可以只保存 content manifest 和 hash。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| problem_id | TEXT | PK 组成部分 | 稳定题目 ID |
| version | TEXT | PK 组成部分 | 内容和评测版本 |
| content_hash | TEXT | NOT NULL | 内容校验值 |
| runtime_key | TEXT | NOT NULL | 评测运行时版本 |
| status | TEXT | NOT NULL | `draft`、`published`、`retired` |
| metadata_json | JSON/TEXT | NOT NULL | 标题、难度、分类等元数据 |
| created_at | TIMESTAMP | NOT NULL | 构建时间 |

已创建的 submission 必须固定引用一个 problem_version，不能因为题目发布新版本而改变历史结果。

### 3.6 submission

代表用户的一次提交请求，是用户可见的业务对象。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | UUID/TEXT | PK | submission ID |
| tenant_id | UUID/TEXT | FK, NOT NULL | 租户 |
| user_id | UUID/TEXT | FK, NOT NULL | 提交用户 |
| problem_id | TEXT | NOT NULL | 稳定题目 ID |
| problem_version | TEXT | NOT NULL | 固定题目版本 |
| mode | TEXT | NOT NULL | `sample`、`submit` |
| source_code | TEXT | NOT NULL | 用户代码，需有大小限制 |
| idempotency_key | TEXT | NOT NULL | 客户端请求幂等键 |
| status | TEXT | NOT NULL | `accepted`、`rejected`、`queued`、`running`、`completed`、`cancelled` |
| created_at | TIMESTAMP | NOT NULL | 创建时间 |
| accepted_at | TIMESTAMP | NULLABLE | 准入时间 |
| completed_at | TIMESTAMP | NULLABLE | 完成时间 |

唯一约束建议为 `(tenant_id, user_id, idempotency_key)`。source_code 的最大长度由租户策略和运行时策略共同限制。

### 3.7 judge_job

代表 submission 在调度系统中的任务。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | UUID/TEXT | PK | job ID |
| submission_id | UUID/TEXT | UNIQUE/FK | 对应提交 |
| tenant_id | UUID/TEXT | FK, NOT NULL | 调度隔离键 |
| priority | INTEGER | NOT NULL | 逻辑优先级 |
| queue_name | TEXT | NOT NULL | 队列名称 |
| status | TEXT | NOT NULL | `queued`、`leased`、`running`、`succeeded`、`failed`、`timeout`、`cancelled`、`expired` |
| attempt | INTEGER | NOT NULL | 当前执行次数 |
| available_at | TIMESTAMP | NOT NULL | 最早可执行时间 |
| lease_until | TIMESTAMP | NULLABLE | worker 租约截止时间 |
| worker_id | TEXT | NULLABLE | 当前 worker |
| queued_at | TIMESTAMP | NOT NULL | 入队时间 |
| started_at | TIMESTAMP | NULLABLE | 开始执行时间 |
| finished_at | TIMESTAMP | NULLABLE | 完成时间 |
| last_error_code | TEXT | NULLABLE | 最近失败类型 |

`attempt` 递增必须有上限。worker 失联后只能通过租约过期恢复，不允许多个 worker 无约束地同时执行同一 job。

### 3.8 judge_result

保存评测摘要和计时结果。单个测试用例明细可以单独保存或放入受大小限制的 JSON。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | UUID/TEXT | PK | 结果 ID |
| job_id | UUID/TEXT | UNIQUE/FK | 评测任务 |
| tenant_id | UUID/TEXT | FK, NOT NULL | 租户 |
| outcome | TEXT | NOT NULL | `passed`、`failed`、`timeout`、`crashed`、`cancelled` |
| passed_count | INTEGER | NOT NULL | 通过数 |
| total_count | INTEGER | NOT NULL | 总数 |
| error_code | TEXT | NULLABLE | 结构化错误类型 |
| queue_wait_ms | INTEGER | NOT NULL | 排队耗时 |
| worker_startup_ms | INTEGER | NOT NULL | worker 启动/预热耗时 |
| setup_ms | INTEGER | NOT NULL | 任务准备耗时 |
| user_code_ms | INTEGER | NOT NULL | 用户代码耗时 |
| test_execution_ms | INTEGER | NOT NULL | 测试耗时 |
| persist_ms | INTEGER | NOT NULL | 持久化耗时 |
| total_ms | INTEGER | NOT NULL | 总耗时 |
| details_json | JSON/TEXT | NULLABLE | 测试明细，需限制大小 |
| created_at | TIMESTAMP | NOT NULL | 结果时间 |

### 3.9 progress

保存用户在租户范围内的题目进度。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| tenant_id | UUID/TEXT | PK 组成部分 | 租户 |
| user_id | UUID/TEXT | PK 组成部分 | 用户 |
| problem_id | TEXT | PK 组成部分 | 题目 |
| status | TEXT | NOT NULL | `todo`、`attempted`、`solved` |
| attempts | INTEGER | NOT NULL | 尝试次数 |
| best_time_ms | INTEGER | NULLABLE | 最佳通过耗时 |
| solved_at | TIMESTAMP | NULLABLE | 首次通过时间 |
| updated_at | TIMESTAMP | NOT NULL | 更新时间 |

主键为 `(tenant_id, user_id, problem_id)`。已 solved 的题目不能被失败提交降级为 attempted。

### 3.10 quota_policy

定义租户和用户的资源预算。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| tenant_id | UUID/TEXT | PK/FK | 租户 |
| max_running_jobs | INTEGER | NOT NULL | 最大并发任务数 |
| max_queued_jobs | INTEGER | NOT NULL | 最大排队任务数 |
| max_user_running_jobs | INTEGER | NOT NULL | 单用户最大并发 |
| rate_limit_per_minute | INTEGER | NOT NULL | 单位时间提交上限 |
| max_source_bytes | INTEGER | NOT NULL | 代码大小上限 |
| max_cpu_ms | INTEGER | NOT NULL | CPU 时间上限 |
| max_memory_bytes | INTEGER | NOT NULL | 内存上限 |
| updated_at | TIMESTAMP | NOT NULL | 更新时间 |

## 4. 状态和事务规则

### 4.1 提交状态

```text
accepted → rejected
accepted → queued → running → completed
queued → cancelled
running → cancelled
running → completed
running → failed
running → timeout
```

提交只有一个最终状态。重复请求使用幂等键返回原 submission，不重复创建 judge_job。

### 4.2 事务边界

创建提交时，在一个事务内完成：

1. 校验 tenant、user 和 membership。
2. 检查幂等键。
3. 检查租户和用户准入配额。
4. 创建 submission。
5. 创建 judge_job。
6. 更新必要的租户队列计数或写入待处理事件。

评测完成时，在一个事务内完成：

1. 检查 job 租约和当前状态。
2. 写入 judge_result。
3. 更新 judge_job 最终状态。
4. 更新 submission 状态。
5. 如通过，更新 progress。

状态更新必须带旧状态条件，防止超时清理、worker 回报和用户取消同时写入时产生覆盖。

## 5. 索引要求

至少需要以下索引：

- `tenant(slug)` 唯一索引。
- `session(token_hash)` 唯一索引。
- `membership(user_id, status)`。
- `progress(tenant_id, user_id, updated_at)`。
- `submission(tenant_id, user_id, created_at)`。
- `submission(tenant_id, problem_id, created_at)`。
- `judge_job(status, available_at, priority)`。
- `judge_job(tenant_id, status, queued_at)`。
- `judge_job(lease_until)`。
- `judge_result(tenant_id, created_at)`。

所有索引都需要结合 SQLite 和 PostgreSQL 的查询计划验证，不能仅凭字段名称添加索引。

## 6. 数据隔离与安全规则

所有带 `tenant_id` 的表都必须在 repository 层强制要求 tenant context。禁止提供只按资源 ID 查询而不校验租户的公共方法。

日志中不得记录明文 session token、API key 或完整 source code。提交代码和 details_json 必须有大小上限；过大内容应被拒绝或转移到受控对象存储。

## 7. 迁移和兼容规则

现有本地 SQLite 的 `users`、`progress` 和 `submissions` 数据需要提供一次性迁移：

- 为旧数据创建默认 local tenant。
- 为旧 session 创建或关联 user。
- 为旧 progress 和 submissions 补齐 tenant_id。
- 保留旧记录的时间和状态语义。
- 迁移必须可重复执行，不能产生重复数据。

数据库迁移使用版本号管理，启动时只执行未完成的迁移。迁移失败时应用应停止启动并报告明确错误，不能静默使用部分 schema。

## 8. 契约变更规则

新增字段优先使用 nullable 或带默认值的方式，避免旧 worker 无法读取新任务。删除字段或改变状态语义需要经过双写/双读兼容阶段。任何影响评测状态、租户隔离或幂等性的变更都必须配套迁移测试、回滚方案和并发测试。
