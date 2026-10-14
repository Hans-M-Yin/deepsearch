# SGLang idle keepalive

长期轮询 router 压力；持续空闲后给每个 worker 补充随机生成请求，真实请求到来时按 `rid` 取消补充请求。支持 Ctrl-C / SIGTERM、网络故障暂停、失败退避、请求 ID 持久化与重启清理。不会修改或重启模型服务。

## 前提

- 所有真实推理请求必须经过所监控的 router，补充请求必须直连普通完整模型 worker。此脚本不支持 PD 分离部署。
- 每个 worker 启动时加 `--enable-metrics`，并提供 `/generate`、`/abort_request`。
- Router 提供真实推理请求的在途 gauge。脚本自动识别 `sgl_router_running_requests`、`smg_worker_requests_active`、`sgl_router_worker_inflight_requests`；不同版本可能需要显式指定指标名。不能把累计请求 counter 当成在途 gauge。
- 地址必须能从脚本所在机器访问。不要把 router 的推理端口或 Prometheus 端口作为 `--worker-url`。
- 如果 router 会在分配 worker 前排队，应监控包含入口等待请求的 gauge，或者提供到达计数器；只统计 worker 已分配请求的 gauge 存在入口队列盲区。

## 启动

依赖为 `aiohttp` 和 `prometheus-client`，不需要在脚本所在机器安装 SGLang 或 CUDA：

```bash
pip install aiohttp prometheus-client
```

先确认端口与指标（以下地址只是示例；co-launch worker 地址可从 router 启动日志获取）：

```bash
curl -s http://127.0.0.1:6661/metrics
curl -s http://WORKER_1:30000/metrics
curl -s http://WORKER_2:30000/metrics
```

先只监控，确认日志里 `real_active` 随业务请求变化：

```bash
python utils/sglang_idle_keepalive.py \
  --router-metrics-url http://127.0.0.1:6661/metrics \
  --worker-url http://WORKER_1:30000 \
  --worker-url http://WORKER_2:30000 \
  --poll-interval 1 --idle-grace 20 \
  --dry-run
```

正常后去掉 `--dry-run` 即开始补充负载。需要后台长期运行时：

```bash
nohup python -u utils/sglang_idle_keepalive.py \
  --router-metrics-url http://127.0.0.1:6661/metrics \
  --worker-url http://WORKER_1:30000 \
  --worker-url http://WORKER_2:30000 \
  --poll-interval 1 --idle-grace 20 \
  --concurrency-per-worker 2 --prompt-words 64 --max-tokens 256 \
  --state-file /tmp/sglang_idle_keepalive_state.json \
  > /tmp/sglang_idle_keepalive.log 2>&1 &
```

记录 shell 返回的 PID；`kill -TERM PID` 会执行取消和状态保存。`nohup` 防止终端退出影响进程，但不负责崩溃自动重启；需要跨机器重启运行时，把同一命令放到已有的进程管理器，并把状态文件放到持久化目录。

## 参数与行为

| 参数 | 默认 | 含义 |
|---|---|---|
| `--poll-interval` | 1 秒 | 每轮查询结束后的等待时间 |
| `--idle-grace` | 20 秒 | 业务结束/监控恢复后持续空闲等待时间 |
| `--concurrency-per-worker` | 2 | 每个 worker 的补充并发上限 |
| `--prompt-words` | 64 | 输入随机单词数量，不是精确 token 数 |
| `--max-tokens` | 256 | 单次生成上限 |
| `--http-timeout` | 3 秒 | 指标与取消请求超时 |
| `--request-timeout` | 120 秒 | 单个补充请求最长 HTTP 生命周期 |
| `--error-backoff` | 30 秒 | 生成失败后的暂停时间 |
| `--report-interval` | 10 秒 | 状态摘要间隔 |
| `--router-active-metric` | auto | 真实请求在途 gauge 名称 |
| `--router-arrival-metric` | 未设置 | 可选的推理请求到达 counter |
| `--background-priority` | 未设置 | 补充请求优先级，需 worker 已启用 priority scheduling |

优先从每 worker 并发 1～2 开始，根据 GPU 利用率和真实业务首 token 延迟逐步调整。脚本通过并发补充负载，不闭环控制 GPU 利用率，也不保证某个利用率百分比。

`--router-arrival-metric` 仅能指定在推理请求**到达时**递增的 counter。指标必须只覆盖推理流量；若混入 `/metrics` 和健康检查，监控本身会导致脚本一直认为有业务。累计值变化会延长空闲等待，可检测两次采样之间已经结束的短请求，但无法追溯取消发生过的计算。不要用请求完成时才递增的指标替代及时的到达检测。

鉴权通过环境变量传入，避免把密钥放入命令行：`--api-key-env SGLANG_API_KEY`；指标和取消接口使用独立管理员 token 时加 `--admin-key-env SGLANG_ADMIN_KEY`。所有 worker 与 router 需接受相应 token，否则监控会暂停。

日志 mode 含义：`business-busy` 为真实请求活动，`idle-grace` 为空闲等待，`filling` 为补充负载，`monitor-error` 为指标查询失败，`cleanup-or-backoff` 为取消/故障清理。workers 摘要为 `(运行数, 排队数, 生成token/s)`；吞吐指标缺失会显示 `None`，不阻止基于请求数的监控。

## 取消、恢复与边界

- 单次轮询并非实时监听。正常发现延迟约为轮询间隔加 HTTP 查询耗时；短请求可能在两次采样之间完成。要求请求进入时立即让路，需要入口代理。
- 只取消脚本记录的 `idle-fill-*` 请求，始终发送 `abort_all: false`。业务或监控故障触发后持续取消，直到补充流返回结束事件；因此提交与取消交错时仍会重试，不会因业务刚结束而停止取消。
- 指标缺失、NaN、负值、HTTP 错误都会暂停补充负载并取消已登记请求。不会将监控失败解释为 GPU 空闲。
- 保存 ID 后才提交请求。终止流缺少结束事件时保留 ID，重复取消；重启时同样先清理。没有可用流确认的 ID，需要连续成功取消且所有 worker 持续空闲一个 `idle-grace` 窗口后才清除，再等待空闲窗口恢复负载。这是保守恢复判断，不是服务端按 ID 查询的完成确认。
- `/abort_request` 返回 200 表示取消已发送，不保证 GPU 当场停止。后台仍可能完成当前 batch/prefill，有限并发和短输入有助于减小影响。
- 同一组 worker 只运行一个控制器，使用固定状态文件；相同状态文件有进程锁。重启不要删除状态和 lock 文件，也不要同时使用不同状态文件启动多个控制器。
- 脚本不会自动发现动态新增/移除的 worker。调整 worker 列表前先正常停止并确认遗留请求已清理。

验证：`python -m unittest utils.test_sglang_idle_keepalive`，使用本机模拟 HTTP 服务，不需要模型或 GPU。
