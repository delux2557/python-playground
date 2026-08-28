# rl-games 架构评估

> 评估基准：基于实际源码（shared/ 8 模块 + othello/ 7 模块 + snake/ 5 模块 + tests/ 14 文件 1595 行），不依赖 README 描述。  
> 评估时间：2026-08-28

---

## 一、整体评分

| 维度    | 评分    | 说明                                                |
| ----- | ----- | ------------------------------------------------- |
| 分层与抽象 | ★★★★★ | shared/protocol + platform + dqn 三层抽象干净，游戏插件化做得彻底 |
| 可扩展性  | ★★★★☆ | 加游戏成本低，但训练/对战/评测三套引擎耦合度可再降                        |
| 代码质量  | ★★★★☆ | 注释极详尽（几乎每行都有"为什么"），类型标注齐全，docstring 规范            |
| 工程化   | ★★★☆☆ | 有测试、Docker、MLflow，但缺 CI、lint 配置、类型检查              |
| 并发安全  | ★★★★☆ | 原子写、线程锁、会话隔离都考虑到了，跨进程靠文件解耦                        |
| 部署友好度 | ★★★☆☆ | .dockerignore 与卷挂载冲突导致预训练模型丢失（已遇坑）                |
| 训练健壮性 | ★★★☆☆ | 无 checkpoint 续训、无早停、无中间存档，崩了从头来                   |

**总评：这是一个设计思路清晰、抽象层次分明的"教学级 RL 平台"原型。架构理念（协议化、平台化、训练/Web 解耦）达到了生产级的思维水准，但工程化收尾（CI、续训、监控、模型初始化）还停在原型阶段。**

---

## 二、架构亮点（值得学习的设计）

### 2.1 三层协议抽象（shared/protocol.py）—— 本项目最亮眼的设计

```
StepRequest（请求格式）  →  AgentService（6 个抽象方法的基类）  →  build_app()（包装成标准 FastAPI 路由）
```

**为什么好**：

- 加一个新游戏 = 写一个 AgentService 子类（实现 6 个方法）+ 提供前端 static/，**路由零改动、前端框架零改动**
- `build_app()` 是高阶函数：传 service 进去，出标准 FastAPI app。所有游戏共用同一套 URL（`/api/meta`、`/api/step`...），前端只需读一份 `/api/meta` 就能渲染任何游戏
- 会话隔离通过 `X-Session-Id` header 透传到 service 方法，多标签页互不干扰——这个设计很轻量，不引入 Redis/Session 中间件

这&#x662F;**"协议优于实现"**&#x7684;教科书式落地，和 OpenAI Gym 的 `Env` 抽象、Stable-Baselines3 的 `BaseAlgorithm` 抽象是同一思路。

### 2.2 对局平台三件套（shared/platform.py）—— 环境/智能体分离

```
Player（参与方：人/模型/随机/训练器）
   ↓ decide(obs, legal_mask) → action
GameAdapter（游戏规则插件，无状态）
   ↓ new_game / observe / apply / done / result
Match（对局仲裁：回合驱动 + 棋谱记录）
```

**为什么好**：

- 严格遵守 RL 经典原则：**环境无状态、智能体无状态、对局有状态**
- `GameAdapter` 是无状态的，可以同时服务多场对局（线程安全的本质）
- 训练热路径（自对弈）不走 `Match`，而是直接调 `play_game()`——`platform.py` 注释明确写了"平台负责编排，不充当训练热路径"。这个边界划得非常清醒，避免了"为了统一抽象而拖慢训练"
- `HumanPlayer` 用 `queue.Queue` 接收输入，虽然 Web 对战实际走 `Match.act()`，但这个类在异步房间/测试场景有意义——**预留了扩展点而不是过度设计**

### 2.3 训练/Web 解耦（通过文件而非消息队列）

## 需要做什么？已经做完了——改源码根治，而不是只打补丁

**根因**（与 agent 分析一致，且是项目自身缺陷而非部署环境问题）：直接运行 `python <子目录>/train.py` 时，Python 只把**脚本所在目录**（`othello/`、`snake/`）放进 `sys.path[0]`，项目根不在。而 `othello/`、`snake/` 是无 `__init__.py` 的命名空间包，必须从项目根才能 import。讽刺的是 [serve.py](file:///workspace/rl-games/othello/serve.py) 顶部**早有**这个 `sys.path` 处理，两个 `train.py` 却漏了。pytest 能过是因为它会自动把 rootdir 加进 sys.path，掩盖了此问题。

**修复**：在 [othello/train.py](file:///workspace/rl-games/othello/train.py#L24-L36) 和 [snake/train.py](file:///workspace/rl-games/snake/train.py#L21-L33) 顶部加上与 serve.py 一致的 `_ROOT` + `sys.path.insert(0, str(_ROOT))`，本地包导入移到其后。

**为什么不用 Dockerfile 补丁方案**：agent 推荐的 `ENV PYTHONPATH=/app` 只能救容器内，救不了 README 里 `python othello/train.py` 的本地用法（本地直接跑仍然报错）。代码内处理一劳永逸，任何环境都通。

## 验证结果

- ✅ `python othello/train.py --help` / `python snake/train.py --help` 直接运行正常
- ✅ 全量测试 107 个用例通过，无回归
- ✅ 已登记到 [BUGFIX.md](file:///workspace/rl-games/BUGFIX.md#L111-L129)（第二轮修复日志）

## 给部署 agent 的交接

- 新包 **`/workspace/rl-games.tar.gz`**（339K）已重新打好，包含修复后的源码
- VM 上那份 `PYTHONPATH: /app` 补丁现在**冗余但无害**——可保留，或下次更新时去掉
- `docker-compose.yml` 无需任何改动，无需再加 `ENV PYTHONPATH=/app

  `

**为什么好**：

- 零中间件：不依赖 Redis/RabbitMQ/Celery，两个进程通过文件系统通信
- **原子写**（`tmp + os.replace`）保证 Web 永远不会读到"写了一半"的坏 JSON——这是跨进程文件通信的正确姿势
- `threading.Lock` 保护进程内并发，`os.replace` 保证跨进程安全，两层防护各司其职
- 训练崩了不影响 Web 服务（`restart: "no"`），Web 重启不影响训练

**trade-off**：文件通信无法做实时推送（Web 端只能轮询），但 RL 训练进度本来就是秒级更新，轮询足够。这个 trade-off 选得合理。

### 2.4 统一评测基准（shared/eval.py）

`register_evaluator("othello")` 装饰器 + `benchmark()` 一次性重测所有模型，用**固定种子**保证可比性。

**为什么好**：

- 解决了"训练时记的 eval_score 不可比"的问题——不同训练 run 用了不同种子/局数，直接比分不公平
- 统一评测用独立 `np.random.default_rng(seed)`，不污染全局随机状态——细节到位
- `_benchmark_lock` 保证同一时间只跑一场（并发跑会互相践踏随机序列，破坏基准）

### 2.5 模型注册表（shared/registry.py）

- `resolve()` 有**路径穿越防护**：`../` 之类的路径会被拒绝，防止用户请求把任意文件喂给 `torch.load`——安全意识到位
- key 带微秒级时间戳防并发覆盖
- 原子写 + 整本重写（读-改-写全在锁内）

### 2.6 DQN 实现的工程细节（shared/dqn.py）

- `replace_online()` 换网络后**重建 optimizer**——注释点出了"直接赋新值会让 optimizer 持有旧参数引用"的坑，这是很多人踩过的雷
- `save()` 用原子写（`tmp + os.replace`），因为"训练保存时 Web 可能正在加载同一个文件"——并发意识贯穿到底
- `select_action()` 的 `legal_mask` 把非法动作 Q 值压成 `-inf` 而非直接过滤——这样 argmax 天然合法，且保留了 Q 值用于可视化

---

## 三、架构问题与改进建议

### 3.1 [P1] 训练无 checkpoint 续训能力

**现状**：`train.py` 的 `trainer.run()` 跑完才 `save + register`。中途崩了（OOM、断电、容器被杀）= 全部重来。

**影响**：2000 局训练可能跑几小时，中途失败成本极高。实际部署已踩坑——首次部署后 worker 训练到 400 局时如果容器重启，进度全丢。

**建议**：在 `SelfPlayTrainer.run()` 的评估回调里加定期 checkpoint：

```python
if ep % self.cfg.checkpoint_freq == 0:
    self.agent.save(self.cfg.checkpoint_path)        # 中间存档
    # 同时更新 progress.json（已有）
```

启动时检查 checkpoint 是否存在，存在则 `DQNAgent.load()` 续训。这是 RL 工程化的标配。

### 3.2 [P1] .dockerignore 与卷挂载冲突导致模型丢失

**现状**：`.dockerignore` 排除 `models/*`，但 `docker-compose.yml` 用命名卷 `models:/app/models` 挂载——空卷覆盖了镜像里本就没有的目录，容器内 `/app/models/` 为空。

**影响**：首次部署后前端显示"暂无模型登记"，已手动灌卷修复。

**建议**（任选）：

- **Dockerfile 加 entrypoint**：启动时若卷为空，从镜像内 `/app/models-seed/`（构建时打包）拷入卷
- **init 容器**：compose 加一个 one-shot 容器，首次启动灌种子模型
- *去掉 .dockerignore 的 models/ 排除** + 改用 bind mount（治标不治本，卷仍会覆盖）

### 3.3 [P1] othello/ 和 snake/ 缺 `__init__.py`，依赖 PYTHONPATH 黑魔法

**现状**：`othello/`、`shared/`、`snake/` 都是命名空间包（无 `__init__.py`，只有 `shared/__init__.py` 和 `snake/__init__.py`）。`python othello/train.py` 直接运行时 `sys.path[0]` 是脚本目录，`from othello.dqn import` 失败。靠 `serve.py` 手动 `sys.path.insert(0, _ROOT)` 或部署时加 `PYTHONPATH=/app` 绕过。

**影响**：部署踩坑（已修复）。`serve.py` 里的 `sys.path.insert` 是 code smell——说明包结构本身不完整。

**建议**：

- 给 `othello/` 加 `__init__.py`（哪怕是空文件）
- 把 worker command 改成 `python -m othello.train`（模块模式，`sys.path[0]` 是 cwd 而非脚本目录）
- 或在 Dockerfile 加 `ENV PYTHONPATH=/app`（最小改动）

### 3.4 [P2] 跨进程并发安全靠"原子替换 + 整本重写"，高并发下有丢更新风险

**现状**：`registry.py` 注释自己说了——`threading.Lock` 只保护进程内，跨进程靠"原子替换"。但**两个进程同时读-改-写** registry.json 时，仍可能丢更新（A 读到旧数据，B 写入，A 再写入覆盖 B 的记录）。

**影响**：当前场景（训练进程写、Web 进程只读）没问题。但如果未来多个 worker 并发训练 + 同时 register，会丢模型登记。

**建议**：

- 短期：文档标注"registry 写入不支持跨进程并发，多 worker 训练需串行 register"
- 长期：改用 SQLite（`registry.db`），天然支持跨进程锁。或用 `fcntl.flock` 文件锁

### 3.5 [P2] 无 CI / lint / 类型检查配置

**现状**：有 `pyproject.toml` 但只有 113 字节，无 ruff/mypy/pytest 配置。有 1595 行测试但无 CI 自动跑。

**建议**：

- `pyproject.toml` 加 ruff + mypy 配置（项目已有完整类型标注，上 mypy 成本低）
- GitHub Actions 跑 `ruff check + mypy + pytest`，PR 合并门槛
- 作为 Python 全栈工程化最佳实践，这也是基本盘

### 3.6 [P2] serve.py 体量过大（othello/serve.py 27KB / 700+ 行）

**现状**：`othello/serve.py` 承担了服务定义 + 会话管理 + 扩展路由（duel/arena/benchmark）+ 模型加载 + 前端元数据，单文件 27KB。

**影响**：可维护性下降，新游戏照抄时不知道哪些是必须的、哪些是 othello 特有的扩展。

**建议**：拆分

```
othello/
  serve.py          # 只留 OthelloService + build_app 调用
  routes_duel.py    # /api/duel/* 扩展路由
  routes_arena.py   # /api/arena/* 扩展路由
  session.py        # 会话管理（SessionManager）
```

### 3.7 [P3] MLflow 是可选依赖但 requirements.txt 强制安装

**现状**：`experiment.py` 做了优雅降级（`_mlflow()` 返回 None 时所有方法静默跳过），但 `requirements.txt` 里 `mlflow` 是硬依赖。降级逻辑永远不会触发——除非手动卸载。

**影响**：镜像体积被 mlflow 全家桶拖大（pandas/pyarrow/scikit-learn/matplotlib 都是 mlflow 拉进来的）。

**建议**：如果 MLflow 是核心功能，保留硬依赖；如果定位为可选，移到 `requirements-optional.txt`，`experiment.py` 的降级逻辑才有意义。

### 3.8 [P3] 前端是原生 JS + Canvas，未模块化

**现状**：`static/app.js` + `static/index.html`，无构建工具、无框架、无模块化。

**trade-off**：对教学项目是优点（零依赖、易读）；对产品化是缺点（复用难、状态管理靠全局变量）。当前定位下可接受，但如果要加第三个游戏，前端复用会成痛点。

---

## 四、架构图（实际依赖关系）

```
┌─────────────────────────────────────────────────────────┐
│                    浏览器 (Canvas + 原生 JS)              │
│              X-Session-Id  /  HTTP 轮询                  │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│ web-othello:8001 │      │ web-snake:8002   │
│ OthelloService   │      │ SnakeService     │
│ (AgentService子类)│      │ (AgentService子类)│
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         │  共用协议层              │
         ▼                         ▼
┌─────────────────────────────────────────────────────────┐
│              shared/protocol.py (build_app)              │
│  /api/meta /api/state /api/step /api/reset /api/models   │
└─────────────────────────┬───────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ shared/dqn   │ │shared/registry│ │ shared/eval  │
│ QNetwork     │ │ registry.json │ │ benchmark    │
│ ReplayBuffer │ │ 原子写+锁      │ │ 统一评测基准  │
│ DQNAgent     │ └──────────────┘ └──────────────┘
└──────┬───────┘
       │ 模型文件读写
       ▼
┌─────────────────────────────────────────────────────────┐
│           命名卷 models/ + data/ (跨进程共享)            │
│  othello.pt / snake.pt / registry.json / progress.json  │
└─────────────────────────▲───────────────────────────────┘
                          │ 只写
                 ┌────────┴────────┐
                 │   worker 容器    │
                 │ SelfPlayTrainer  │
                 │ (othello/selfplay)│
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │ shared/platform  │ ← 对局平台（编排，非热路径）
                 │ Player/Adapter/  │
                 │ Match            │
                 └─────────────────┘
```

**关键边界**：

- 实线（HTTP）：浏览器 ↔ Web 服务
- 虚线（文件）：worker ↔ 命名卷 ↔ Web（跨进程解耦）
- platform.py 不在训练热路径上，只服务于 duel/arena（对战/打榜）

---

## 五、与业界实践对照

| 实践            | 本项目                    | 业界标杆                          | 评价             |
| ------------- | ---------------------- | ----------------------------- | -------------- |
| 环境抽象          | GameAdapter（无状态）       | OpenAI Gym `Env`              | ✅ 思路一致，更轻量     |
| 智能体抽象         | Player.decide()        | SB3 `BaseAlgorithm.predict()` | ✅ 合理           |
| 训练/Web 解耦     | 文件 + 原子写               | Celery + Redis / Ray          | ✅ 轻量级方案，够用     |
| 实验跟踪          | MLflow（可选降级）           | MLflow / W\&B / TensorBoard   | ✅ 标准选择         |
| 模型注册表         | JSON + 原子写             | MLflow Model Registry / 自建 DB | ⚠️ 原型级，够用但不可扩展 |
| Checkpoint 续训 | 无                      | 每 N 步存档 + 断点续训                | ❌ 缺失，生产级硬伤     |
| 统一评测          | 固定种子 benchmark         | OpenAI Baselines eval harness | ✅ 思路到位         |
| 自对弈对手池        | 历史快照轮换                 | AlphaGo League Training       | ✅ 简化版正确实现      |
| 会话隔离          | X-Session-Id + 内存 dict | Redis Session / JWT           | ✅ 轻量够用         |

---

## 六、总结

**这个项目最值得称道的是"抽象层次"**——protocol / platform / dqn 三层划分干净利落，游戏插件化、训练/Web 解耦、统一评测基准三个理念贯穿始终，注释质量极高（几乎每行都解释"为什么"而非"是什么"）。作为 RL 学习/教学项目，架构认知水准超出平均水平。

**最大的短板在工程化收尾**：

1. 训练无续训能力（崩了重来）
2. 部署的 .dockerignore/卷挂载冲突（已踩坑）
3. 包结构不完整（靠 sys.path hack）
4. 无 CI/lint/type check

**如果只做三件事优先改进**：

1. 加训练 checkpoint 续训（P1，生产可用性）
2. 修包结构 + `python -m` 启动（P1，消除 sys.path hack）
3. 加 ruff + mypy + GitHub Actions CI（P2，工程化基本盘）

做完这三件，这个项目从"优秀的教学原型"就迈进"可复用的 RL 平台底座"。
