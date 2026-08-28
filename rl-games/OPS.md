# rl-games 运维手册

> 目标主机：VMware 虚拟机 `think@192.168.19.129`（Ubuntu 24, kernel 6.8）
> 部署目录：`~/rl-games/`
> 首次部署日期：2026-08-28

---

## 1. 环境信息

| 项 | 值 |
|---|---|
| 主机 | myserver (192.168.19.129) |
| 系统 | Ubuntu 24.04, kernel 6.8.0-138 |
| SSH | `ssh think@192.168.19.129`（已配免密） |
| Docker | 29.7.2 |
| Docker Compose | v5.5.0 |
| 镜像 | `rl-games:latest`（python:3.12-slim 基础 + torch CPU + mlflow） |
| 部署路径 | `/home/think/rl-games/` |

---

## 2. 服务拓扑

三容器，共用 `rl-games:latest` 镜像，通过命名卷 `models` / `data` 共享数据（训练/Web 解耦）。

| 服务 | 端口 | 命令 | 用途 |
|---|---|---|---|
| web-othello | 8001 | `uvicorn othello.serve:app --host 0.0.0.0 --port 8001` | 黑白棋驾驶舱 |
| web-snake | 8002 | `uvicorn snake.serve:app --host 0.0.0.0 --port 8002` | 贪吃蛇驾驶舱 |
| worker | - | `python othello/train.py --episodes 2000 --checkpoint models/othello.pt` | 黑白棋自对弈训练 |

命名卷：
- `rl-games_models` → `/app/models`（模型权重，worker 写、web 读）
- `rl-games_data` → `/app/data`（训练曲线/进度 JSON、MLflow db，worker 写、web 读）

---

## 3. 日常运维命令

> 以下命令均在 VM 上 `~/rl-games/` 目录执行（先 `ssh think@192.168.19.129`，再 `cd ~/rl-games`）。

### 3.1 启动 / 停止

```bash
# 启动全部服务（镜像已构建时，秒起）
docker compose up -d

# 停止并移除容器（数据卷保留，训练成果不丢）
docker compose down

# 停止但保留容器（不移除，重启快）
docker compose stop
docker compose start

# 查看状态
docker compose ps
```

### 3.2 查看日志

```bash
# Web 服务日志
docker compose logs -f web-othello
docker compose logs -f web-snake

# worker 日志（注意：train.py 进度写文件非 stdout，logs 多半为空）
docker compose logs -f worker
```

### 3.3 查看训练进度（推荐）

worker 的训练进度写到 `data/othello_progress.json`，不打印到 stdout。两种查看方式：

```bash
# 方式一：通过 Web API 读（最直观，返回 JSON）
curl http://localhost:8001/api/train/status | python3 -m json.tool

# 返回示例：
# {
#   "status": "running",
#   "episode": 200,
#   "episodes": 2000,
#   "win_rate": 0.7,
#   "epsilon": 0.6701,
#   "opponent_pool": 2,
#   "running": true
# }

# 方式二：直接读卷里文件
docker compose exec worker cat /app/data/othello_progress.json | python3 -m json.tool

# 训练曲线
curl http://localhost:8001/api/curve | python3 -m json.tool
```

### 3.4 数据卷管理

```bash
# 查看卷
docker volume ls | grep rl-games

# ⚠️ 删除卷（会清空所有模型和训练数据，谨慎！）
# docker compose down -v   # 停服时连带删卷
```

### 3.5 访问服务

在 VM 本机浏览器（如有）：
- http://localhost:8001 — 黑白棋
- http://localhost:8002 — 贪吃蛇

从 Windows 访问（三种方式，详见第 6 节）：
- 直连：http://192.168.19.129:8001 （需 VM 防火墙放行）
- SSH 隧道：`ssh -N -L 8001:localhost:8001 -L 8002:localhost:8002 think@192.168.19.129`，再访问 http://localhost:8001

---

## 4. 代码更新流程（重要）

项目代码更新后，重新部署的步骤：

### 4.1 传输新代码到 VM

```bash
# 方式 A：本地打包传（推荐，适合大改）
# 在 Windows 本地项目根目录：
tar -czf rl-games.tar.gz rl-games/
scp rl-games.tar.gz think@192.168.19.129:/tmp/
ssh think@192.168.19.129 "cd ~/rl-games && tar -xzf /tmp/rl-games.tar.gz --strip-components=1"

# 方式 B：只传改动的文件（适合小改）
scp othello/serve.py think@192.168.19.129:~/rl-games/othello/serve.py
```

### 4.2 重建并重启

```bash
ssh think@192.168.19.129 "cd ~/rl-games && docker compose up -d --build"
```

`--build` 会重新构建镜像。由于 Dockerfile 先 `COPY requirements.txt` 再 `COPY . .`，**如果只改了代码没动依赖，依赖层命中缓存，重建只需几秒**；改了 `requirements.txt` 才会重装依赖（约 6-9 分钟）。

### 4.2.1 部署补丁（每次代码更新后必须重加）

新版本的 `docker-compose.yml` 和 `.dockerignore` 是原始版本，每次用新代码覆盖后需要重加两个部署补丁：

```bash
ssh think@192.168.19.129 "cd ~/rl-games && \
  # 补丁1: PYTHONPATH (compose 三服务加 environment: PYTHONPATH /app)
  # 见 ~/rl-games/docker-compose.yml.orig 备份,或用 sed 补丁
  # 补丁2: .dockerignore 排除 mlruns/ (MLflow 产物 14MB+,不该进镜像)
  grep -q mlruns .dockerignore || echo 'mlruns/' >> .dockerignore"
```

> 补丁详情见第 5 节。建议开发者把这两处修到源码里根治。

### 4.3 一键更新脚本（可选）

如果想一条命令搞定，可在 VM 上创建 `~/rl-games/update.sh`：

```bash
#!/usr/bin/env bash
set -e
cd ~/rl-games
echo "[1/3] 停旧容器..."
docker compose down
echo "[2/3] 解压新代码（需先 scp 到 /tmp/rl-games.tar.gz）..."
tar -xzf /tmp/rl-games.tar.gz --strip-components=1
echo "[3/3] 重建并启动..."
docker compose up -d --build
echo "完成。状态："
docker compose ps
```

用法：
```bash
# Windows 侧
scp rl-games.tar.gz think@192.168.19.129:/tmp/
ssh think@192.168.19.129 "bash ~/rl-games/update.sh"
```

---

## 5. 部署侧补丁说明（重要）

### 5.1 PYTHONPATH 补丁 —— ✅ 已根治，无需再打

**历史问题**：旧版 `python othello/train.py` 直接运行脚本时，Python 把脚本所在目录 `/app/othello` 放进 `sys.path[0]`，而非工作目录 `/app`，导致 `from othello.dqn import ...` 失败（`othello/` 无 `__init__.py`，是命名空间包）。

**当前状态（2026-08-28 第二版已根治）**：开发 agent 已在 `othello/train.py` 和 `snake/train.py` 顶部加上与 `serve.py` 一致的 `_ROOT + sys.path.insert(0, str(_ROOT))`，本地直接运行和容器内运行都正常。**VM 上 `docker-compose.yml` 已是原始版本，无需 PYTHONPATH 环境变量。**

验证命令（无 PYTHONPATH 直接跑）：
```bash
docker run --rm rl-games:latest python othello/train.py --help
docker run --rm rl-games:latest python snake/train.py --help
```

**遗留说明**：`mlruns/`（MLflow 产物，14MB+）仍需在 `.dockerignore` 排除——那是训练产物，不该进镜像。若新包不含 `mlruns/` 则无需处理。

### 5.2 预训练模型进卷 —— ✅ 已由 .dockerignore 修复解决

**历史问题**：旧版 `.dockerignore` 排除 `models/*`，命名卷首次挂载时空目录覆盖镜像内目录，前端"暂无模型登记"。曾用手动 `docker compose cp` 灌卷。

**当前状态**：新版 `.dockerignore` 已放行 `models/*.pt` 和 `registry.json`，预训练模型随镜像构建，命名卷首次挂载自动复制进卷。**无需再手动灌卷。**

**旧的手动灌卷命令（仅历史参考，不再需要）**：

```bash
ssh think@192.168.19.129 "cd ~/rl-games && \
  docker compose cp models/othello.pt web-othello:/app/models/othello.pt && \
  docker compose cp models/snake.pt web-othello:/app/models/snake.pt && \
  docker compose cp models/registry.json web-othello:/app/models/registry.json"
```

验证：
```bash
curl http://localhost:8001/api/models   # 黑白棋，应返回 3 个模型
curl http://localhost:8002/api/models   # 贪吃蛇，应返回 1 个模型
```

**注意**：`docker compose down -v`（删卷）或重建卷后需重新灌入。建议把这个操作加进 `update.sh` 脚本，或在 Dockerfile 里去掉 `.dockerignore` 的 `models/*` 排除（但会让镜像变大，且卷挂载仍会覆盖——治标不治本，灌卷是正解）。

### 5.3 根治建议（可选，改源码）

PYTHONPATH 问题任选其一：
1. **Dockerfile 加一行**：`ENV PYTHONPATH=/app`（最小改动，推荐）
2. **给包加 `__init__.py`** + worker command 改成 `python -m othello.train`
3. **worker command 改成** `sh -c "cd /app && python -m othello.train ..."`（不推荐，绕弯）

模型预灌问题：在 `docker-compose.yml` 加一个 init 容器，首次启动时把 `/app/models-seed/`（镜像内打包的预训练模型）拷到 `models` 卷。或更简单——改 `.dockerignore` 放行 `models/registry.json` 和 `models/*.pt`，再用一个 entrypoint 脚本：卷为空时从镜像拷入。

---

## 6. 从 Windows 访问 VM 服务

### 方式一：直连（最简单）

compose 端口映射已是 `0.0.0.0:8001:8001`，VM 局域网内可直接访问。前提是 VM 防火墙放行：

```bash
# 在 VM 上（首次配置）
sudo ufw allow 8001/tcp
sudo ufw allow 8002/tcp
```

然后 Windows 浏览器访问 http://192.168.19.129:8001

### 方式二：SSH 本地端口转发（推荐，安全）

无需开放防火墙端口：

```bash
# 在 Windows Git Bash 执行（窗口保持开着）
ssh -N -L 8001:localhost:8001 -L 8002:localhost:8002 think@192.168.19.129
```

然后访问 http://localhost:8001 / http://localhost:8002。Ctrl+C 关闭隧道。

### 方式三：后台 SSH 隧道

```bash
ssh -fN -L 8001:localhost:8001 -L 8002:localhost:8002 think@192.168.19.129
```

---

## 7. 常见问题

### Q: `docker compose logs worker` 没输出？
A: 正常。train.py 把进度写到 `data/othello_progress.json`，不打 stdout。用 `curl http://localhost:8001/api/train/status` 看进度。

### Q: worker 启动报 `ModuleNotFoundError: No module named 'othello'`？
A: `docker-compose.yml` 的 `PYTHONPATH: /app` 补丁丢了，参考第 5 节重新加上。

### Q: 重启后训练进度没了？
A: 不会。`docker compose down` 不删卷，`docker compose up -d` 后卷自动挂回。只有 `docker compose down -v` 才删卷。

### Q: 想重新从零训练？
A: `docker compose down -v && docker compose up -d`，`-v` 清空 models/data 卷。

### Q: 镜像构建太慢？
A: torch CPU 包 191MB 是大头。只要不改 `requirements.txt`，依赖层命中缓存，重建只跑 `COPY . .` 那层，几十秒完成。

---

## 8. 关键文件位置速查

| 内容 | 位置 |
|---|---|
| 部署目录 | VM `~/rl-games/` |
| Compose 配置 | VM `~/rl-games/docker-compose.yml`（含 PYTHONPATH 补丁） |
| Compose 原始备份 | VM `~/rl-games/docker-compose.yml.bak` |
| 训练进度 JSON | 容器内 `/app/data/othello_progress.json`（卷 `rl-games_data`） |
| 训练曲线 | 容器内 `/app/data/othello_curve.json` |
| 模型权重 | 容器内 `/app/models/othello.pt`（卷 `rl-games_models`） |
| MLflow db | 容器内 `/app/data/mlflow.db` |
