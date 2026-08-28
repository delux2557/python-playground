# python-playground

Python 机器学习 / 深度学习学习项目的聚合仓库。

一个仓库收纳所有小练习，每个项目一个子目录，目录即索引。
（大型项目如 rl-games 有完整 Docker 部署流程，也以子目录形式托管在这里，保持独立可运行。）

## 目录

| 子目录 | 项目 | 说明 |
|---|---|---|
| [rl-games/](./rl-games/) | RL 强化学习平台 | 黑白棋 / 贪吃蛇 DQN 训练 + Web 对战驾驶舱 |

## 使用约定

- 每个子项目**相互独立**：各自有 `requirements.txt` / venv，互不共享依赖。
- 新增项目：拷贝进一个新子目录 → `git add . && git commit -m "add: <项目名>" && git push`。
- 大文件（模型权重、训练产物）按各项目 `.gitignore` / `.dockerignore` 规则处理。

## 快速开始（rl-games）

```bash
cd rl-games
# 本地直接跑测试
python -m pytest tests/ -q
# Docker 一键起全套（web 8001/8002 + 训练 worker）
docker compose up -d --build
```

部署细节、运维命令、常见问题见 [rl-games/OPS.md](./rl-games/OPS.md)；
架构设计与评估见 [rl-games/ARCHITECTURE_REVIEW.md](./rl-games/ARCHITECTURE_REVIEW.md)。

## 环境要求

- Python 3.12+（本地运行）
- Docker + Docker Compose（容器部署）
