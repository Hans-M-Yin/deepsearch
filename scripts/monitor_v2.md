# monitor_v2：通过 HTTP 控制 GPU holder

在每台需要控制的 GPU 机器上运行一个服务。它只响应外部命令，不轮询 GPU 利用率，也不自动检测业务请求。

`holder.py` 本身会为每张可见 GPU 创建计算子进程，因此 monitor 启动一份 `python -u holder.py`，通过 `CUDA_VISIBLE_DEVICES` 限定管辖 GPU。`/stop` 会终止这份 holder 的整个进程组，包括它创建的 GPU 子进程。

## 启动服务

```bash
nohup python -u scripts/monitor_v2.py \
  --host 0.0.0.0 --port 18090 \
  --gpus 0,1,2,3,4,5,6,7 \
  > /tmp/monitor_v2.log 2>&1 &
```

- 默认 holder：`/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/verl/holder.py`。可用 `--target-script /path/to/holder.py` 修改。
- 默认用启动 monitor 的 Python 运行 holder。若环境不同，加 `--python /path/to/venv/bin/python`；该解释器必须有 PyTorch 和 CUDA。
- 不指定 `--gpus` 时，优先沿用 `CUDA_VISIBLE_DEVICES`，否则通过 `nvidia-smi` 查询全部本机 GPU UUID。GPU 范围在服务启动时确定；HTTP 请求不能修改执行脚本或 GPU 范围。
- 服务启动后 holder **不会自动启动**，等待 `/start`。
- Holder 输出追加到 `/tmp/monitor_v2_holder.log`；HTTP 服务日志在 `/tmp/monitor_v2.log`。

## HTTP 接口

| 请求 | 作用 |
|---|---|
| `POST /start` | 启动 holder；已在运行时返回 `already_running`，不会重复启动 |
| `POST /stop` | 停止 holder 进程组；已经停止时返回 `already_stopped` |
| `GET /status` | 返回 `running`、PID、管辖 GPU 列表、退出码和日志路径 |
| `GET /health` | 同 `/status` |

请求不需要 JSON body。成功返回 HTTP 200 和 JSON；失败返回非 200。`/start` 成功表示进程已创建，不代表 PyTorch 初始化完成；可通过 `/status` 与 holder 日志核对。`/stop` 等待进程组退出后才返回，先 SIGTERM，默认 5 秒后仍未退出则 SIGKILL。

```bash
curl -X POST http://GPU机器IP:18090/start
curl http://GPU机器IP:18090/status
curl -X POST http://GPU机器IP:18090/stop
```

## Python 远程调用

使用 `requests`：

```python
import requests

MONITOR_URL = "http://10.124.100.11:18090"  # 换成运行 monitor_v2 的 GPU 机器 IP

def start_holder():
    response = requests.post(f"{MONITOR_URL}/start", timeout=30)
    response.raise_for_status()
    return response.json()

def stop_holder():
    response = requests.post(f"{MONITOR_URL}/stop", timeout=30)
    response.raise_for_status()
    return response.json()

def holder_status():
    response = requests.get(f"{MONITOR_URL}/status", timeout=30)
    response.raise_for_status()
    return response.json()

print(start_holder())
print(holder_status())
print(stop_holder())
```

实际业务中，空闲时调用 `start_holder()`；处理业务前调用 `stop_holder()` 并等待成功，再向模型发送真实请求。多个业务请求并发时，由调用方统一管理忙闲状态，避免一个请求结束就启动 holder，而其他请求仍在执行。

多台 GPU 机器分别部署服务，Python 并发控制所有节点：

```python
from concurrent.futures import ThreadPoolExecutor
import requests

MONITORS = ["http://10.124.100.11:18090", "http://10.124.100.12:18090"]

def control_all(action):
    if action not in {"start", "stop"}:
        raise ValueError("action must be start or stop")
    def call(url):
        response = requests.post(f"{url}/{action}", timeout=30)
        response.raise_for_status()
        return response.json()
    with ThreadPoolExecutor(max_workers=len(MONITORS)) as pool:
        futures = [pool.submit(call, url) for url in MONITORS]
        results = []
        errors = []
        for url, future in zip(MONITORS, futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
        return results

control_all("start")
control_all("stop")
```

每台 monitor 只管理自己启动的 holder。不会扫描或停止其他服务启动的 `holder.py`，也不会停止训练或模型服务。避免在相同 GPU 上同时运行旧版自动 monitor 和这个服务。

## 可选鉴权与服务退出

若设置环境变量 `MONITOR_V2_TOKEN`，所有接口都要求 `Authorization: Bearer TOKEN`。Python 请求可加 `headers={"Authorization": "Bearer " + os.environ["MONITOR_V2_TOKEN"]}`。未设置该变量时不要求 token。

Ctrl-C 或 `kill -TERM MONITOR_PID` 会先停止管理的 holder，再退出 HTTP 服务。`nohup` 不负责崩溃自动重启；SIGKILL 或机器异常退出无法执行清理，也不支持跨服务重启接管遗留进程。服务运行期间不要删除 lock 文件。

测试：`python -m unittest scripts.test_monitor_v2`，使用 CPU 模拟 holder，不会占用 GPU。

## RL rollout 自动控制

DeepResearch 公共启动脚本 `qwen3-vl-8b-2node.sh` 及其继承脚本默认设置
`RLLM_HOLDER_MONITOR_URL=http://10.124.139.215:18090`。先在该 GPU 机器上运行上述 monitor_v2 服务。

训练 driver 在每次 `generate_trajectories()` 开始前调用 `/stop`，确认 `running=false` 后才执行 rollout；整个 rollout（包含奖励计算与后处理）成功返回后调用 `/start`。训练、验证和重复采样都经过这个边界。更新和保存 checkpoint 期间 holder 可持续运行；训练正常结束或异常进入 shutdown 时尝试 `/stop`。

停止失败会重试 3 次，仍无法确认就报错并阻止 rollout；启动失败重试后记录警告，更新继续进行。Rollout 本身异常退出时不会重新启动 holder。日志关键字为 `[RL_HOLDER]`。SIGKILL/机器故障不能执行退出清理。

- 改地址：`export RLLM_HOLDER_MONITOR_URL=http://其他机器:18090`
- 关闭：`export RLLM_HOLDER_MONITOR_URL=''`
- 请求超时：`export RLLM_HOLDER_MONITOR_TIMEOUT=30`
- 若 monitor 已设置 `MONITOR_V2_TOKEN`，训练端设置相同值到 `RLLM_HOLDER_MONITOR_TOKEN`。该变量会转发到 Ray TaskRunner，不会写入控制日志。

此控制适用于该远端服务器在所有 rollout 之外没有其他业务请求的场景；多个训练任务不要同时控制同一个 monitor。代码更新不会热加载到已运行的训练 driver，需要下一次启动训练后生效。
