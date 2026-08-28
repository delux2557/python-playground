# BUGFIX 问题追踪

代码审核发现的问题清单与修复登记。状态说明:
- ✅ 已修复(附修复方式)
- ⏳ 待修复
- 🚫 不修复(附理由,设计取舍)

审核日期:2026-08-28

---

## 高优先级

| # | 问题 | 位置 | 状态 | 修复说明 |
|---|------|------|------|----------|
| H1 | 换模型返回 default 会话快照:`/api/models/load` 不读 `X-Session-Id`,`load_model` 调 `snapshot()` 不带 session,跨会话串状态 | shared/protocol.py | ✅ | `load_model(name, session)` 透传 session;路由读取 `X-Session-Id`;顺带修正快照键覆盖顺序(`_load_weights` 返回的模型信息为权威) |
| H2 | 悔棋在人机模式下是空操作:`undo()` 只弹一手(弹的是 AI 那手),前端悔棋后立刻 `scheduleAi()`,AI 贪心重下同一手,棋盘恢复原状 | othello/serve.py | ✅ | `undo()` 连续弹子直到重新轮到人类(通常弹 2 手,pass 场景弹 1 手);纯 AI 对战弹 1 手;前端悔棋前先 `cancelScheduled()` |
| H3 | `--model` 命令行参数被完全忽略:服务在模块导入期用硬编码路径构造,`args.model` 从未使用 | 两个 serve.py | ✅ | `main()` 用 `resolve(args.model)` 解析(支持注册表 key),非默认模型时重建服务;保留导入期构建以兼容 `uvicorn xxx.serve:app` 和测试 |
| H4 | snake 模型加载失败直接崩溃:注释称"失败回退随机",但 `torch.load` 无 try/except 也无 exists 检查 | snake/serve.py | ✅ | `__init__` 中 try/except 包裹 `_load_weights`,失败打印警告并回退随机模型;othello 同样补上 |
| H5 | registry 读-改-写非原子 + 跨进程无锁:`register()` 的 load→改→save 不在同一把锁内;`threading.Lock` 只在单进程有效;`save_registry` 非原子写;`json.loads` 无容错 | shared/registry.py | ✅ | 读-改-写收进同一把锁;写改为"临时文件 + os.replace"原子替换;读损坏文件返回空表不崩溃。跨进程互斥靠原子替换兜底(注释已说明局限) |
| H6 | arena 排行榜 `head_to_head` 活引用泄漏到锁外:锁释放后序列化时后台线程可能正在改同一个 dict | shared/arena.py | ✅ | `_leaderboard()` 返回 `head_to_head` 的深拷贝 |
| H7 | eval.benchmark 用相对路径加载模型:直接拿注册表相对路径给 `torch.load`,按 CWD 解析,不从项目根启动就静默失败 | shared/eval.py | ✅ | 改用 `resolve(m["key"])` 得到项目根内的绝对路径 |

## 中优先级

| # | 问题 | 位置 | 状态 | 修复说明 |
|---|------|------|------|----------|
| M1 | 会话数无上限:客户端可用唯一 `X-Session-Id` 洪泛刷会话,内存线性增长 | 两个 serve.py | ✅ | 新增 `_MAX_SESSIONS=256` 上限,超限返回 429 |
| M2 | snake step 不检查 done(蛇死后可"复活")+ 人类动作不校验 0~3(非法动作 500) | snake/serve.py, snake/env.py | ✅ | env 增加 `done` 标志,终局后 step 抛 ValueError;服务层终局返回 400、动作范围校验 400、ValueError 统一转 400 |
| M3 | othello step 不校验回合归属:curl 可在人类回合发 `action=` 替人落子 | othello/serve.py | ✅ | step 校验回合归属:人的回合拒绝 `ai` 指令、AI 的回合拒绝 `action` 指令(纯 AI 对战不受限),均返回 400 |
| M4 | `snapshot()`/`config()` 嵌套调 `meta()` → 每个会话每次请求都触碰 default 会话(刷新 seen,default 永不被回收) | 两个 serve.py | ✅ | 抽出 `_model_info()` 供 snapshot/config 使用;`meta()` 不再读会话状态(othello 的 players 字段移除,执色信息由 `/api/state` 的 human_color 提供) |
| M5 | 前端单格方向缓冲吞输入:一个 tick 内快速按两个方向,先按的被覆盖丢失 | snake/static/app.js | ✅ | `queuedDir` 单缓冲改为长度 2 的 `dirQueue` 队列,每节拍消费一个,带队尾去重 |
| M6 | sessionStorage 会话 ID 在"复制标签页"时共享:两个标签页继承相同 SESSION_ID 操作同一盘棋 | 两个 app.js | ✅ | 用 Navigation Timing 判断:仅 `reload` 沿用旧 ID 续局,新开/复制标签页一律重新签发 |
| M7 | 并发启动循环赛返回 500:`api_arena_start` 无锁预检后 `start()` 抛 RuntimeError 未捕获 | othello/serve.py | ✅ | 捕获 RuntimeError 转 400(与 duel/start 一致) |
| M8 | benchmark 无并发保护 + 全局 numpy 种子竞态 + 排行榜非原子写 | shared/eval.py, othello/serve.py | ✅ | benchmark 加 `_benchmark_lock`(并发返回 400);排行榜原子写;评测器改用 `evaluate_win_rate(seed=2026)` 的独立随机源,删除全局 `np.random.seed` |
| M9 | othello 小棋盘越界:`step()` 用硬编码 `N_ACTIONS=64` 校验,size=6 时动作 36~63 通过后 IndexError | othello/env.py | ✅ | 改用 `self.size * self.size` 校验动作范围 |
| M10 | `evaluate_win_rate` 的 seed 参数是死代码:随机对手走全局 `np.random`,"固定种子统一基准"未实现 | othello/selfplay.py | ✅ | 随机对手改用 `np.random.default_rng(seed)`,seed 真正生效且不污染全局 |
| M11 | snake 训练文件非原子写:`curve.json`/`snake.pt` 直接覆盖写,与 Web 并发读冲突 | snake/train.py, shared/dqn.py | ✅ | curve.json 与 `DQNAgent.save()` 均改为"临时文件 + os.replace"原子写 |
| M12 | 前端 `setInterval(pollTrain)` 训练结束后仍终生每 2 秒拉曲线重绘 | othello/static/app.js | ✅ | 保存轮询句柄,按训练状态自适应调频:训练中 2s,空闲 15s |
| M13 | `resolve()` 路径穿越:`/api/models/load` 的 `name` 未净化就与项目根拼接 | shared/registry.py | ✅ | 解析后的路径必须落在项目根内(`resolve().relative_to(root)`),否则拒绝 |

## 低优先级

| # | 问题 | 位置 | 状态 | 修复说明 |
|---|------|------|------|----------|
| L1 | 前端 "选择 undefined" 显示 bug:init/switchModel 构造 `action: -1` 占位 | snake/static/app.js | ✅ | `renderDecision` 对 `action < 0` 显示"待决策" |
| L2 | `set_human` 非法输入返回 500:字典下标转换抛 KeyError | othello/serve.py | ✅ | 先校验 black/white/none,非法返回 400 |
| L3 | `_load_weights` 无维度校验:othello 服务误加载 snake.pt 后每步 step 必 500,服务瘫痪 | 两个 serve.py | ✅ | 加载时校验 `input_dim`/`n_actions` 与本游戏匹配,不匹配抛错(经协议层转 400) |
| L4 | `_load_weights` 换网后 optimizer 仍绑定旧参数(复用训练会出错) | 两个 serve.py | ✅ | 新增 `DQNAgent.replace_online()`:替换网络同时重建 optimizer 和目标网络,两个服务改用它 |
| L5 | resetGame/switchModel 不设 busy,可与在途 step 请求交错 | 两个 app.js | ✅ | snake:`resetGame` 置 busy、switchModel 递增 gameGen 并同步方向;othello:switchModel 先 `cancelScheduled()` |
| L6 | arena 单局异常终止整场循环赛(缺单局容错) | shared/arena.py | ✅ | 单局 try/except:异常局记为执黑方弃权负,赛程继续 |
| L7 | `load_benchmark` 无异常处理,读到半截 JSON 报 500 | shared/eval.py | ✅ | JSONDecodeError/OSError 时返回空结构 |
| L8 | `grid_size < 3` 时 snake reset 生成负坐标蛇身 | snake/env.py | ✅ | 构造函数校验 `grid_size >= 3`,否则抛 ValueError |
| L9 | `max_steps_since_food=0` 被 `or` 吞掉 | snake/env.py | ✅ | 改用 `is None` 判断 |
| L10 | experiment.py `start()` 中 start_run 成功后续异常 → MLflow run 永不关闭 | shared/experiment.py | ✅ | start_run 成功后立即置 `_active=True`,异常分支调 `end()` 兜底关闭 |
| L11 | othello/train.py `finish()` 硬写 `epsilon: 0.0`,训练面板展示失真 | othello/train.py | ✅ | `finish()` 接收真实终值(训练结束时的 ε 与对手池大小) |
| L12 | othello/train.py 最终评估/保存/登记任一异常 → progress 永远停在 running | othello/train.py | ✅ | try 范围扩大到"评估→保存→登记→finish",任一失败 `reporter.fail()` 置 error |
| L13 | `select_action` 全 False 掩码时探索分支抛难懂的 ValueError | shared/dqn.py | ✅ | 全非法掩码提前抛明确错误信息 |
| L14 | othello switchModel 未 cancelScheduled,可能短暂渲染错误棋盘 | othello/static/app.js | ✅ | 已随 L5 一并修复 |

## 不修复(设计取舍)

| # | 问题 | 理由 |
|---|------|------|
| N1 | GET `/api/state` 有创建会话副作用(不符合 GET 幂等) | 会话惰性创建是设计核心,前端首屏必须拿到状态 |
| N2 | CORS `allow_origins=["*"]` | 本地教学项目,无公网暴露 |
| N3 | obs 危险标记含尾格、不含掉头规则 | 观测建模误差,改动会使已训模型失效;记录在案供后续训练改进 |
| N4 | DuelSession/ArenaSession 进程级单例 | 设计取舍:对战/循环赛本就是全站共享的"表演"资源 |
| N5 | othello 终局后禁止悔棋 | 已通过 H2 修复一并放开(终局后可悔棋,前端按钮同步解禁) |
| N6 | registry 跨进程文件锁 | 单文件原子替换 + 整本重写已把损坏概率压到极低;引入 fcntl 会增加平台复杂度,教学项目不值得 |
| N7 | 对战/循环赛无停止机制 | 低优先级体验项,前端已可停止轮询;服务端线程为 daemon 不阻塞退出 |
| N8 | selfplay 每步把终局 outcome 当即时奖励(与自举叠加) | 算法设计问题,涉及训练效果,需单独实验验证后再改,不在本次修复范围 |

---

## 修复日志

### 2026-08-28 · 第一轮修复

**修复范围**:高优先级 7 项全部修复;中优先级 13 项全部修复;低优先级 14 项全部修复。

**改动文件**:
- `shared/protocol.py` — load_model 会话透传 + 路由读会话头
- `shared/registry.py` — 原子写、读-改-写同锁、损坏容错、路径穿越防护
- `shared/eval.py` — resolve 绝对路径、并发锁、原子写、读容错
- `shared/arena.py` — head_to_head 深拷贝、单局容错
- `shared/dqn.py` — 空掩码防御、replace_online、save 原子写
- `shared/experiment.py` — MLflow run 泄漏兜底
- `snake/env.py` — done 标志、grid_size 校验、max_steps is None
- `snake/serve.py` — 会话上限、终局拦截、动作校验、模型加载回退、维度校验、--model 生效、meta 去副作用
- `snake/train.py` — curve.json 原子写
- `snake/static/app.js` — 方向队列、会话 ID 复制标签页防护、busy 竞态、undefined 显示
- `othello/env.py` — 小棋盘动作校验
- `othello/selfplay.py` — seed 真正生效(独立 Generator)
- `othello/serve.py` — 悔棋连弹、回合秩序校验、会话上限、模型加载回退/维度校验、--model 生效、并发 400、meta 去副作用、set_human 校验
- `othello/train.py` — 异常兜底置 error、真实终值
- `othello/static/app.js` — 会话 ID 复制标签页防护、悔棋交互、轮询降频、switchModel 清理
- `tests/test_othello_serve_smoke.py`、`tests/test_othello_duel.py` — 适配回合校验与悔棋新契约

**验证结果**:
- `pytest tests/` 全部 93 个用例通过
- 真实服务冒烟验证(8901/8902 端口):
  - 非法动作 → 400 ✅
  - 蛇撞墙后再 step → 400(死后复活拦截)✅
  - 会话隔离 A/B 独立 ✅
  - 人回合发 ai 指令 → 400 ✅
  - 悔棋连弹回到人的回合(步数 0、human_turn=True)✅
  - set_human 非法值 → 400 ✅
- 前端 `node --check` 语法通过

### 2026-08-28 · 第二轮修复(部署反馈)

**问题**:容器内 `python othello/train.py` / `python snake/train.py` 直接运行
报 `ModuleNotFoundError: No module named 'othello'/'shared'`。

**根因**:直接运行 `python <子目录>/train.py` 时,Python 只把**脚本所在目录**
(`othello/` 或 `snake/`)放进 `sys.path[0]`,项目根不在;而 `othello/`、`snake/`
是无 `__init__.py` 的命名空间包,顶层包必须在项目根下才能被 import。
`serve.py` 早已有"项目根插入 sys.path"的处理,`train.py` 漏了。
(pytest 能过是因为它会把 rootdir 加入 sys.path,掩盖了此问题。)

**修复**:`othello/train.py`、`snake/train.py` 顶部加与 `serve.py` 一致的
`_ROOT` 定义 + `sys.path.insert(0, str(_ROOT))`,本地包导入移到其后。

**验证结果**:
- `python othello/train.py --help` / `python snake/train.py --help` 直接运行正常 ✅
- `pytest tests/` 107 个用例全部通过(无回归)✅
- Dockerfile 无需再加 `ENV PYTHONPATH=/app`(源码已根治);VM 上已加的
  PYTHONPATH 补丁冗余但无害,可保留也可随下次更新去掉。

### 2026-08-28 · 第三轮修复(前端工程化:共享层 + 模块拆分)

**问题**:
1. 两个游戏驾驶舱约 300 行通用逻辑(api 封装/会话 ID/曲线绘制/模型切换/
   训练轮询)在各自 app.js 里逐行复制——改一处通用逻辑要改两份。
2. othello/app.js 单文件 1302 行、snake/app.js 601 行,职责混杂,维护定位难。

**修复**(零构建、零新依赖,用浏览器原生 ES Modules):
- 新增 `shared/static/frontend/common.js` + `common.css`(设计 tokens + 公共组件);
  `shared/protocol.py` 的 `build_app` 把它们挂到 `/shared`(共享前端层)。
- 前端全部改为 `<script type="module">`:
  - othello 拆 `static/js/{main,board,features,state}.js`(入口/渲染/高级功能/状态)
  - snake 拆 `static/js/{main,board,state}.js`
  - 状态收敛到 `state.js` 避免循环依赖;流程编排留在 `main.js` 事件里;
    `features.js` 需要主流程的地方由 main 事件包装(scheduleAi 等)。
- CSS 拆 tokens + 公共布局:公共进 `common.css`,游戏特有留在各自 `style.css`。
- 附带修复一个重构后暴露的隐患:模块顶层在极少数环境早于 DOM 解析执行,
  `setupCanvas` 拿到 null 画布(间歇性 `Cannot set properties of null`)——
  两个 `board.js` 加顶层 `await` 的 DOM 就绪防御。

**验证结果**:
- `pytest tests/` 107 个用例全部通过(含 /shared 挂载验证)✅
- 真实服务浏览器冒烟(8901/8902):
  - 黑白棋页面:console 无任何错误,棋盘/对战/竞技场全渲染 ✅
  - 贪吃蛇页面:连续刷新 3 次零错误(竞态已修复),画面/决策/观察全渲染 ✅
- 两页面均未出现"无法连接后端服务"错误层 ✅

### 2026-08-28 · 第四轮修复(部署反馈:断点续训 + 镜像模型 + 包结构)

**问题**(部署 agent 反馈的 3 个 P1):
1. **训练无 checkpoint 续训**:2000 局跑几小时,中途崩了(断电/OOM/手动停)
   得从头再来,浪费数小时。
2. **`.dockerignore` 排除 `models/*` + 卷挂载覆盖**:预训练模型没进镜像,
   首次部署挂载命名卷 `models/` 时,Docker 从镜像复制的是空目录,
   预训练模型(蛇/黑白棋)直接丢失。
3. **包结构缺 `__init__.py`**:第二轮靠 `sys.path.insert` 补丁绕行,
   但 worker 用 `python -m` 方式运行才最稳;`othello/` 仍无 `__init__.py`。

**修复**:
- **断点续训**(新增 `shared/checkpoint.py`):
  - `save_with_meta()` 在每次评估点原子存"模型 + meta(已训练局数/探索率)",
    崩溃最多丢 `eval_freq` 局;`load_for_resume()` 启动时检测 checkpoint,
    加载权重并"接着上次局数"往下训,探索率从上次位置继续衰减。
  - `snake/train.py`:训练循环从 `start_ep+1` 开始,评估时存档;
    续训时读回旧曲线,新点往后接(Web 曲线连续)。
  - `othello/selfplay.py`:`SelfPlayTrainer.run()` 增加 `start_ep` 参数;
    `othello/train.py`:训练前检测 checkpoint 恢复 agent 与探索率,
    `ProgressReporter` 支持 `initial_curve` 保留历史点,评估回调里定期存档。
- **`.dockerignore`**:`models/*` 改为排除再放行
  `!models/*.pt` + `!models/registry.json`,预训练模型进镜像,
  首次挂载命名卷时自动复制进卷。
- **`.dockerignore` 追加 `mlruns/`**:MLflow 训练产物(单次训练可达 14MB)
  不应进镜像构建上下文,与 `mlflow.db` 一起排除。
- **包结构**:补 `othello/__init__.py`(与 snake/shared 一致),
  worker 与文档统一改为 `python -m othello.train`。

**验证结果**:
- 新增续训测试 `test_checkpoint_resume_continues_from_saved_episode`
  (snake + othello 各一个):第一段跑 8 局存档 → 第二段目标 12 局,
  断言输出含 `[续训] 从第 8 局续训`、曲线 `[4, 8, 12]` 连续、meta 终值 12 ✅
- `pytest tests/` 109 个用例全部通过(无回归)✅
- `python -m othello.train --help` / `python -m snake.train --help` 正常 ✅
