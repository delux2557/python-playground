# rl-games · 强化学习入门项目

> 用 DQN 训练会玩游戏的强化学习模型，并把"模型到底在想什么"实时可视化，最后演进到人和模型同屏对战。

* 技术路线：**贪吃蛇（入门热身）→ 黑白棋（人机对战）**，全程手写 DQN（PyTorch）+ Web 可视化。

* 详细设计文档见 [DESIGN.md](./DESIGN.md)；问题追踪与修复登记见 [BUGFIX.md](./BUGFIX.md)。

* 阅读顺序：先读本 README 建立全局认知 → 需要深入理解设计时读 DESIGN.md → 排查问题时查 BUGFIX.md。

***

## 1. 项目简介

这个项目是一个"可复用的 RL 学习/训练/对战平台"，核心目标有三层：

1. **训练**：手写 DQN 训练智能体学会玩游戏（贪吃蛇、黑白棋）。
2. **可视化**：Web 驾驶舱实时展示模型每一步的"内心活动"（Q 值、观察输入、ε 等）。
3. **平台化**：同一套 Web 界面能服务任何游戏——换游戏、换算法、换模型时前端零改动。

**平台化的核心**是 [shared/protocol.py](file:///workspace/rl-games/shared/protocol.py) 定义的三层契约：`StepRequest`（请求格式）、`AgentService`（游戏只需实现 6 个方法的抽象基类）、`build_app()`（把任意服务包装成标准路由）。

***

## 2. 架构

### 2.1 分层结构

```
浏览器 (HTML5 Canvas + 原生 JS)
   │  HTTP 请求 / X-Session-Id 头
   ▼
shared/protocol.py  ← 统一 Agent 服务协议（标准路由，所有游戏共用）
   │  实现 AgentService 抽象基类
   ├── snake/serve.py     贪吃蛇服务（会话隔离 + 换模型）
   └── othello/serve.py   黑白棋服务（会话隔离 + 悔棋 + 对战/竞技场扩展路由）
         │  共用
         ▼
shared/platform.py  对局平台核心（Player / GameAdapter / Match）
                    训练模型、人类、随机基线都以"参与方"身份统一接入
shared/static/frontend/  共享前端层（common.js + common.css，挂在 /shared）
                    所有游戏驾驶舱共用的逻辑与样式，消除跨游戏重复代码
shared/dqn.py       通用 DQN 组件（经验回放 / Q 网络 / 目标网络）
shared/registry.py  模型注册表（登记 / 列出 / 切换模型）
shared/eval.py      统一评测基准（固定种子给全体模型打分）
shared/arena.py     循环赛打榜引擎（全体模型两两对战排榜）
shared/experiment.py MLflow 实验跟踪
```

### 2.2 目录与模块职责

| 路径                                                                      | 职责                                                                 |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [shared/platform.py](file:///workspace/rl-games/shared/platform.py)     | 对局平台核心：`Player`（参与方统一接口）/ `GameAdapter`（游戏插件）/ `Match`（对局仲裁引擎） |
| [shared/static/frontend/](file:///workspace/rl-games/shared/static/frontend/common.js) | 共享前端层：`common.js`（会话ID/api/曲线/模型切换/训练轮询）+ `common.css`（tokens/公共组件），经 `build_app` 挂到 `/shared` |
| [shared/protocol.py](file:///workspace/rl-games/shared/protocol.py)     | 统一协议层：`StepRequest` / `AgentService` / `build_app()`，标准路由 + 会话隔离透传 |
| [shared/dqn.py](file:///workspace/rl-games/shared/dqn.py)               | 通用 DQN：经验回放、Q 网络、目标网络、`select_action`、`replace_online`、原子保存        |
| [shared/registry.py](file:///workspace/rl-games/shared/registry.py)     | 模型注册表：`register` / `list_models` / `resolve`，原子写 + 路径穿越防护          |
| [shared/eval.py](file:///workspace/rl-games/shared/eval.py)             | 统一评测基准：固定种子打分、排行榜原子写、并发保护                                          |
| [shared/arena.py](file:///workspace/rl-games/shared/arena.py)           | 循环赛引擎：两两对战（黑白各打一遍消除先后手偏差）、积分排榜、头对头统计                               |
| [shared/experiment.py](file:///workspace/rl-games/shared/experiment.py) | MLflow 实验跟踪：训练超参数 / 指标 / 模型 artifact 自动记录                          |
| snake/                                                                  | 游戏一：贪吃蛇（env / dqn / train / serve / static 前端，按 js/{main,board,state} 拆分） |
| othello/                                                                | 游戏二：黑白棋（env / adapter / selfplay / train / serve / duel / static 前端，按 js/{main,board,features,state} 拆分） |
| tests/                                                                  | pytest 测试套件（109 个用例）                                                |
| models/                                                                 | 训练产出的 checkpoint + `registry.json`（注册表数据）                          |
| data/                                                                   | 训练进度 / 曲线 JSON（训练与 Web 之间通过文件解耦通信）                                 |
| mlruns/ + mlflow\.db                                                    | MLflow 实验数据库                                                       |

### 2.3 数据流

```
训练进程(train.py)              Web 进程(serve.py)
────────────────────           ─────────────────────
环境 + DQN 自对弈               协议层标准路由
   │ 原子写 data/*.json ───────►  /api/curve、/api/train/status 只读
   │ 存模型 + registry.register ─► /api/models 列出、/api/models/load 切换
                                ├─► /api/step 模型推理
                                └─► 前端 Canvas 渲染
```

训练与 Web 是**两个独立进程**，不共享内存，只通过文件契约通信（详见"机制"）。

***

## 3. 机制

### 3.1 会话隔离（最重要的机制）

**解决的问题**：多人同时开多个浏览器标签页，之前所有标签页共用同一个游戏环境，按键互相覆盖——贪吃蛇会"漂移"、黑白棋互相踩棋。

**实现方式**：

* 前端用 `crypto.randomUUID()` 生成会话 ID，存在 `sessionStorage` 中，每次请求带 `X-Session-Id` 头。

* 后端协议层把 `X-Session-Id` 透传给服务方法（`snapshot/reset/step/load_model` 都有 `session` 参数）。

* 每个服务维护一个**会话表** `session_id -> 游戏环境`：每个会话独立一局。

**全局资源 vs 会话状态的分界**：

* **按会话隔离**：对局局面（蛇的状态、棋盘、悔棋历史、人执哪色）。

* **全局共享**：模型权重、训练曲线、模型注册表、对战（duel）、竞技场（arena）、评测（benchmark）。

**生命周期管理**（防内存泄漏/恶意攻击）：

* `_MAX_SESSIONS = 256`：会话数上限，超限返回 `429`。

* `_SESSION_TTL = 30 分钟`：无活动自动回收（每次请求刷新 `seen`）。

* 不带会话头的请求（如 curl 测试）落到 `default` 会话。

**关键注意**：`GET /api/state` 有"惰性创建会话"的副作用（为让首屏拿到状态），这是刻意的设计取舍，不是 bug。

### 3.2 统一协议层

任何游戏实现 `AgentService` 的 6 个方法（`meta/snapshot/reset/step/curve/config` + 可选的 `train_status` + `_load_weights`）后，`build_app()` 生成一套**完全一致**的标准路由：

```
GET  /api/meta          元数据（棋盘/动作/观察含义）+ 已登记模型
GET  /api/state         当前局面 + 模型看到的一切（观察 + Q 值 + ε）
POST /api/reset         开局
POST /api/step          走一步 {"ai":true} 或 {"action":0~3}
GET  /api/curve         训练曲线
GET  /api/config        训练超参数
GET  /api/model         当前加载的模型信息
GET  /api/models        模型注册表列表
POST /api/models/load   切换模型 {"name": "模型key或路径"}
GET  /api/train/status  训练进程实时状态
```

前端启动时先读 `/api/meta`，一切渲染由元数据驱动——这就是"换游戏前端零改动"的秘密。黑白棋额外加了扩展路由：

```
POST /api/player        设置人执黑/白/纯AI对战
POST /api/undo          悔棋
GET  /api/history       当前局棋谱
POST /api/duel/start    发起模型对战（黑 vs 白任选两模型）
GET  /api/duel/status   对战进度
GET  /api/duel/game/{index}  单局复盘
GET  /api/duel/regions  分区胜率热力图
POST /api/arena/start   发起循环赛
GET  /api/arena/status  循环赛进度
GET  /api/arena/leaderboard  排行榜
POST /api/benchmark     对全部模型统一评测
GET  /api/benchmark     读取评测结果
```

### 3.3 训练/Web 解耦

训练（几小时）和 Web（常驻）是独立进程，**只通过文件通信，不共享内存**：

* `train.py` 通过 `ProgressReporter` 把状态原子写入 `data/*.json`（状态机 `starting → running → done/error`）。

* Web 端 `GET /api/train/status`、`GET /api/curve` 只读这些文件。

* **原子写**（先写 `.tmp` 再 `os.replace`）：Web 任何时刻读取都是完整 JSON。

* 前端按训练状态自适应轮询：训练中 2 秒一次，空闲 15 秒一次。

好处：训练进程挂了不影响 Web；Web 重启不打断训练。

### 3.4 原子文件操作（并发安全的核心）

共享资源（注册表、模型 checkpoint、训练曲线、排行榜）一律采用"**临时文件 + os.replace 原子替换**"：

* 写数据先写 `xxx.json.tmp`，再 `os.replace` 覆盖原文件。

* 跨进程并发写（训练进程 + Web 进程同时写注册表）靠原子替换 + 整本重写兜底（`threading.Lock` 只保护单进程）。

* 读损坏文件返回空结构，不崩溃。

### 3.5 模型注册表与切换

* 训练脚本存模型后调 `registry.register()` 自动登记，key 用 `{game}-{时间戳到微秒}` 保证唯一。

* `resolve(key_or_path)` 把 key 或路径解析成**项目根内**的绝对路径；路径必须落在项目根内，防止路径穿越。

* 切换模型入口是 `_load_weights`，必须做**维度校验**（防止误加载别的游戏的模型导致每步 500）。

* 换模型时用 `DQNAgent.replace_online()` 重建网络，同步重建 optimizer 和目标网络（否则复用训练会出错）。

### 3.6 DQN 训练

三个关键技巧（详见 DESIGN.md 第 4 章）：

1. **经验回放**：`(s, a, r, s')` 存大缓冲池随机抽样，打破样本相关性。
2. **目标网络**：延迟更新的参数算目标 Q 值，避免"自己追自己"发散。
3. **ε-greedy 探索**：以概率 ε 随机走、否则按 Q 值贪心，保证见过足够多样的局面。

贪吃蛇是单人决策（每步有奖励）；黑白棋是双人零和博弈，升级为**自对弈（self-play）**：让模型跟"对手池"（历史 checkpoint）下棋，避免两个最新版互相 exploit 同一个漏洞而原地打转。黑白棋只在终局给奖励（+1 赢 / -1 输 / 0 平），配合法动作掩码。

### 3.7 对战 / 竞技场 / 统一评测

* **统一评测**（`shared/eval.py`）：所有模型面对同一把"尺子"——固定种子 + 固定局数，得分可横向对比。黑白棋评测器用 `evaluate_win_rate(seed=2026)` 的独立随机源，不污染全局随机状态。

* **模型对战**（`othello/duel.py`）：任选两个模型打一架，记录单局棋谱、分区（角落/边/中央）胜率统计。对局引擎已上提到平台层（见 3.8）。

* **循环赛**（`shared/arena.py`）：注册表所有模型两两对战，**黑方白方各打一遍**消除先手优势，积分制排总榜。

### 3.8 对局平台抽象（Player / GameAdapter / Match）

这是把"游戏服务"升级成"对战平台"的核心一步，对应业内"环境/智能体分离"的最佳实践（Gymnasium Env / ML-Agents Behavior / PettingZoo AEC）。

**三个抽象**（[shared/platform.py](file:///workspace/rl-games/shared/platform.py)）：

* **Player（参与方）**：人 / 模型 / 训练器 / 随机基线，统一接口 `decide(obs, legal_mask) -> action`。内置 `ModelPlayer`（包 DQNAgent）、`RandomPlayer`（评测地板）、`HumanPlayer`（动作外部注入）。

* **GameAdapter（游戏插件）**：纯规则、无状态——`new_game / current_player / observe / legal_mask / apply / done / result`。[othello/adapter.py](file:///workspace/rl-games/othello/adapter.py) 是第一个实现。

* **Match（一场对局）**：把 N 个 Player 放到一个 GameAdapter 上，负责回合仲裁与棋谱记录。`play()` 同步打完整局（对战/竞技场）；`act()/auto_step()` 逐步驱动（人机对战）。人类座位不放 Player 时 `auto_step()` 抛 KeyError——这是回合秩序的天然防线。

**分工原则**：新增游戏 = 写一个 `GameAdapter`；新增参与方 = 写一个 `Player`；平台只做仲裁与编排，不懂具体游戏规则。

**训练热路径约定**：训练时的高频自对弈在进程内直接调用，不走网络——平台负责"编排"，不当训练热路径。

**现状**：`duel` 与竞技场的 `match_fn` 已改走 `Match + ModelPlayer`；人机对战的 HTTP 会话仍由服务直接驱动 env（逐步交互需展示 Q 值），是下一步"通用对局观看器"的工作。

### 3.9 前端工程化（共享层 + 模块拆分）

驾驶舱之前是"每游戏一份独立 static + 原生 JS 单文件"，两个游戏有约 300 行通用逻辑（api 封装/会话 ID/曲线绘制/模型选择/训练轮询）逐行复制。已做轻量工程化（**零构建、零新依赖**）：

* **共享前端层** [shared/static/frontend/](file:///workspace/rl-games/shared/static/frontend/common.js)：`common.js`（`SESSION_ID`/`api`/`setupCanvas`/`drawCurve`/`renderConfig`/`renderModelPicker`/`startTrainPolling`）+ `common.css`（`:root` 设计 tokens + 公共组件）。协议层 `build_app` 把它们挂到 `/shared`，两个游戏 `<script type="module">` 直接 import。
* **JS 按功能拆分**（浏览器原生 ES Modules）：othello 拆 `js/{main,board,features,state}.js`（入口/渲染/高级功能/状态），snake 拆 `js/{main,board,state}.js`。每个文件职责单一，改一处通用逻辑只改一处。
* **CSS 拆 tokens + 公共布局**：公共部分进 `common.css`，游戏特有的棋盘/对战/竞技场样式留在各自 `style.css`。

**开发守则**：新增共享逻辑进 `shared/static/frontend/`；游戏特有的画布/面板渲染进各自的 `board.js`/`features.js`；避免模块间循环依赖（状态收敛到 `state.js`，流程编排留在 `main.js` 事件里）。

***

## 4. 之前遇到的问题（重点摘要）

> 完整的问题清单、修复说明和登记见 [BUGFIX.md](./BUGFIX.md)。这里只保留"最容易再犯"的高价值教训。

### 4.1 根源问题：多标签页串局 → 蛇漂移

* **现象**：人玩模式下啥都不按，蛇不走直线，自己"漂移"。

* **根因**：服务端全局单例游戏环境，多个浏览器标签页共用同一条蛇/同一盘棋，按键互相覆盖。

* **修复**：会话隔离（见 3.1），每个会话独立一局。这是本项目最重要的一次架构改造。

### 4.2 状态一致性类（最容易踩的坑）

| 问题         | 根因                                    | 修复                                         |
| ---------- | ------------------------------------- | ------------------------------------------ |
| 蛇死后"复活"    | `step()` 不检查 done，终局后仍可移动             | 环境加 `done` 标志，终局后 step 抛错，服务层转 400         |
| 换模型后串到别的会话 | `/api/models/load` 不读会话头，返回默认会话快照     | `load_model` 透传 session，路由读 `X-Session-Id` |
| 黑白棋回合秩序乱   | 服务端不校验回合归属，curl 可越权落子                 | `step()` 校验回合：人回合拒绝 AI 指令、AI 回合拒绝人指令       |
| 悔棋无效       | `undo()` 只弹一手（弹的是 AI 那手），前端又立刻让 AI 重下 | `undo()` 连续弹子直到重新轮到人（通常 2 手）               |
| 换模型后每步 500 | `_load_weights` 无维度校验，误加载别的游戏模型       | 加载时校验 `input_dim`/`n_actions` 匹配           |

### 4.3 并发类

| 问题            | 根因                      | 修复                         |
| ------------- | ----------------------- | -------------------------- |
| 注册表并发写丢记录     | 读-改-写不在同一把锁内；跨进程无互斥     | 读-改-写收进同一锁 + 原子写 + 损坏容错    |
| 排行榜活引用泄漏      | 锁释放后序列化时后台线程正在改同一个 dict | 返回 `head_to_head` 深拷贝      |
| 循环赛/评测并发 500  | 无并发保护，`start()` 抛异常未捕获  | 加锁，并发返回 400；异常转 400        |
| 训练曲线读到半截 JSON | 直接覆盖写，与 Web 并发读冲突       | 全部改为"临时文件 + os.replace"原子写 |

### 4.4 安全类

| 问题                                           | 修复                            |
| -------------------------------------------- | ----------------------------- |
| `/api/models/load` 路径穿越（`../` 喂给 torch.load） | `resolve()` 校验路径必须落在项目根内      |
| 唯一会话头无限刷会话撑爆内存                               | `_MAX_SESSIONS=256` 上限，超限 429 |
| 会话永不被回收                                      | TTL 30 分钟 + 定期 GC             |

### 4.5 前端类

| 问题                    | 修复                                              |
| --------------------- | ----------------------------------------------- |
| 快速连按方向键，先按的被覆盖丢失      | 单格方向缓冲改为长度 2 的方向队列，每节拍消费一个                      |
| 复制标签页继承同一会话 ID，操作同一盘棋 | 用 Navigation Timing 判断：仅页面刷新沿用旧 ID，新建/复制标签页重新签发 |
| 训练结束后仍终生每 2 秒拉曲线      | 按训练状态自适应轮询（训练中 2s / 空闲 15s）                     |

***

## 5. 以后的注意事项（开发守则）

### 5.1 新增游戏时

1. 写环境 `env.py`（Gymnasium 风格 `reset/step/observation`）。
2. 写训练脚本 `train.py`，存模型时调 `registry.register()` 登记。
3. 写 `AgentService` 子类 `serve.py`，实现那 6 个方法，然后 `app = build_app(MyService(), static_dir)` 完成。
4. **必须**实现会话隔离（会话表 + TTL + 上限），否则多标签页会互相干扰。
5. **必须**在 `_load_weights` 做维度校验，防止跨游戏误加载模型。

### 5.2 修改共享资源（注册表/曲线/模型文件）时

* 写操作一律"临时文件 + `os.replace`"原子替换，读操作对损坏文件容错。

* 涉及读-改-写必须收进同一把锁。

* 记住 `threading.Lock` 只保护单进程；跨进程并发靠原子替换兜底。

### 5.3 修改协议层时

* 前端由元数据驱动，**不要在前端写死任何游戏细节**。

* 对局路由（state/reset/step/load）必须透传 `X-Session-Id` 给服务方法。

* 新扩展路由只加在游戏服务的 `serve.py`，不要动通用路由。

### 5.4 使用对局平台（Player / GameAdapter / Match）时

* 新游戏接入对战/竞技场：写一个 `GameAdapter`，不要自己写回合循环。

* 新参与方（模型/人/训练器）：实现 `Player.decide()`，不要在对局引擎里加 `if isinstance(agent, ...)` 分支。

* 训练热路径留在进程内（直接调 `play_one` / `Match.play()`），**不要**把每步决策走 HTTP——平台只编排，不当训练热路径。

* 人机对战逐步驱动用 `act()`（注入人类动作）+ `auto_step()`（模型应手）；人类座位不注册 Player，`auto_step()` 抛 KeyError 就是回合秩序防线，别捕获后静默跳过。

### 5.5 已知的设计取舍（有意为之，别当 bug 修）

| 取舍                         | 理由                              |
| -------------------------- | ------------------------------- |
| `GET /api/state` 有创建会话副作用  | 会话惰性创建是设计核心，前端首屏必须拿到状态          |
| CORS `allow_origins=["*"]` | 本地教学项目，无公网暴露                    |
| 对战/竞技场/评测是进程级单例            | 本就是全站共享的"表演"资源                  |
| registry 用原子替换而非 fcntl 文件锁 | 原子替换已把损坏概率压到极低，教学项目不值得引入平台相关复杂度 |
| 人机对战会话未迁到 Match            | 逐步交互要展示 Q 值/翻转数等驾驶舱数据，直接驱动 env 更直接 |

### 5.6 测试与验证

* 运行测试：`pytest tests/`（109 个用例全绿）。

* 前端 JS 语法检查：`for f in snake/static/js/*.js othello/static/js/*.js shared/static/frontend/*.js; do node --check "$f"; done`。

* 真实服务冒烟：`python snake/serve.py --port 8000`、`python othello/serve.py --port 8001`，用 curl 带/不带 `X-Session-Id` 验证会话隔离。

***

## 6. 其他重要信息

### 6.1 启动方式

```bash
# 本地直接跑
python snake/serve.py --port 8002      # 贪吃蛇驾驶舱
python othello/serve.py --port 8001    # 黑白棋驾驶舱

# 训练（黑白棋自对弈；用 python -m 以包方式运行，靠包内 __init__.py 定位依赖）
# 训练会定期存档 checkpoint；中断后重跑同一命令即自动续训（接着上次局数，探索率无缝衔接）
python -m othello.train --episodes 2000 --checkpoint models/othello.pt

# Docker 一键起全套（web-othello:8001 / web-snake:8002 / worker 训练）
docker compose up -d
docker compose logs -f worker
docker compose down        # 数据保留在命名卷 models/ 和 data/
```

### 6.2 会话协议速查

* 每个标签页带 `X-Session-Id` 头（前端自动生成，无需手动配置）。

* 不带头的请求（curl 等）用 `default` 会话。

* 会话 30 分钟无活动回收；最多 256 个。

### 6.3 关键技术选型

| 层    | 选择                   | 理由                   |
| ---- | -------------------- | -------------------- |
| 语言   | Python 3.10+         | 生态全，学习资料多            |
| 深度学习 | PyTorch 2.x          | 张量操作直观，便于手写 DQN      |
| 算法   | 手写 DQN + 自对弈         | 入门必修，贯穿所有现代 RL       |
| 后端   | FastAPI              | 轻量、类型友好              |
| 前端   | HTML5 Canvas + 原生 JS | 无构建负担，实时渲染           |
| 实验跟踪 | MLflow               | 记录超参数/指标/模型 artifact |

### 6.4 代码风格约定

* 代码注释面向新手，解释"这个模块/参数在干什么、调它会改变什么"，不写废话。

* 每个模块开头写"这个模块做什么"。

***

## 7. 文档索引

| 文档                           | 内容                               |
| ---------------------------- | -------------------------------- |
| [README.md](./README.md)（本文） | 项目总览：架构 / 机制 / 问题教训 / 注意事项       |
| [DESIGN.md](./DESIGN.md)     | 详细设计：RL 概念、环境/DQN/自对弈设计、协议层、调参速查 |
| [BUGFIX.md](./BUGFIX.md)     | 问题追踪：高/中/低优先级修复登记 + 设计取舍 + 修复日志  |

