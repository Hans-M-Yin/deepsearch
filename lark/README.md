# Linux 集群飞书命令机器人

这是一个单用户、轻量版的飞书机器人实现：

- 使用飞书应用机器人的长连接接收单聊消息；
- 只允许白名单中的 `open_id`；
- 只处理单聊文本消息；
- 在机器人进程所在的 Linux 主机执行 `/bin/bash -lc` 命令；
- 命令成功时回复 `Done!`；
- 运行超过 1 分钟时，每分钟回复一次最新 30 行输出；
- 命令以 `SLICENCE` 前缀开头时，关闭中间进度，只在结束时回复 `Done!` 或最终错误；
- 命令退出码非 0、超时或执行异常时，回复原消息并附上有限长度的错误输出；
- 使用 SQLite 按 `message_id` 去重，避免飞书重复推送导致命令重复执行。

## 飞书后台配置

在飞书开放平台创建企业自建应用，然后：

1. 开启机器人能力。
2. 添加消息权限，至少允许接收用户发给机器人的单聊消息，并允许应用发送消息。
3. 在事件订阅中选择“使用长连接接收事件”。
4. 订阅 `im.message.receive_v1`（接收消息 v2.0）。
5. 把应用可用范围限制为你自己的账号。
6. 创建并发布应用版本。

长连接模式下，集群进程主动连接飞书云，因此不需要向公网暴露 HTTP 端口。集群需要允许 DNS、HTTPS 和长期 WebSocket 出站连接。

## 集群部署

以下命令在 Linux 集群中执行，当前开发机不会自动连接集群：

```bash
cd lark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 编辑项目中的 .lark_env
```

编辑 `lark/.lark_env`，至少填写：

- `LARK_APP_ID`
- `LARK_APP_SECRET`
- `LARK_ALLOWED_OPEN_IDS`
- `LARK_COMMAND_CWD`
- `LARK_STATE_DB`

第一次不知道自己的 `open_id` 时，可以暂时只填写 App ID/Secret，不填写
`LARK_ALLOWED_OPEN_IDS`。程序会拒绝命令，但会在日志中打印被拒绝消息的
`tenant_key` 和 `open_id`。把自己的 `open_id` 填入 `.lark_env` 后重启程序。

启动：

```bash
source .venv/bin/activate
python bot.py
```

如果集群没有 systemd，可以使用集群允许的常驻方式，例如 `tmux`：

```bash
tmux new -s lark-bot
source .venv/bin/activate
python bot.py
# 按 Ctrl-b，再按 d，退出 tmux 但保持进程运行
```

不要把 `.lark_env`、App Secret 或 `lark_state.sqlite3` 提交到 Git。

## 命令执行位置

当前代码默认在机器人进程所在主机执行命令。如果机器人运行在集群登录节点，命令也会在登录节点执行；它不会自动进入计算节点。

如果集群使用 Slurm、Kubernetes 或其他调度系统，需要把 `bot.py` 中的
`run_shell_command` 替换为对应的执行适配器。例如 Slurm 场景通常需要通过
`srun --wait` 或作业 API 等待最终退出码，而不能只判断 `sbatch` 是否提交成功。

## 运行时注意事项

- 这是有意开放 shell 的远程执行程序，只应在个人、受控环境中运行。
- 建议使用专用的普通 Linux 用户运行，禁止 root 和 `sudo`。
- `LARK_MAX_WORKERS=1` 会串行执行命令，第一版建议保持不变。
- 长命令超过 `LARK_COMMAND_TIMEOUT_SECONDS` 会被终止并返回超时错误。
- 成功命令会回复 `Done!`；运行时间较长的命令会每分钟回复最新 30 行输出。
- 在命令前加 `SLICENCE` 可以关闭长命令的中间进度，例如：
  `SLICENCE python train.py` 或 `SLICENCE: python train.py`。
- 服务进程停止时不会接收新消息，也不会恢复正在执行的任务。

## 高层伪代码

```text
启动机器人
  加载 App ID、App Secret 和用户白名单
  建立 SQLite 去重表
  通过飞书长连接订阅单聊消息

收到消息
  解析 tenant_key、open_id、message_id 和文本内容
  如果不是白名单用户、不是单聊或不是文本消息：忽略
  如果 message_id 已处理：忽略
  记录 message_id
  解析可选的 SLICENCE 前缀
  把命令放入后台执行队列
  立即返回事件处理结果

后台执行命令
  在配置的 Linux 工作目录运行 bash 命令
  如果未启用 SLICENCE 且到达 1 分钟间隔：回复最新 30 行输出
  如果成功：通过飞书 API 回复 Done!
  如果失败或超时：回复错误和截断后的输出
```
