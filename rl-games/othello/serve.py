"""othello/serve.py —— 黑白棋人机对战服务(实现 shared/protocol.py 的协议)。

平台化示范(和 snake/serve.py 同一套骨架):
  · 游戏特有的部分(环境、模型、元数据、扩展路由)写在 OthelloService
  · 通用 HTTP 路由交给协议层 build_app() 生成,前端和路由零改动
  · 额外加了一条"扩展路由" POST /api/player 用于设置"人执黑/白/纯AI对战"
    ——协议允许游戏服务加自己的扩展,不影响通用部分

启动方式:
  python othello/serve.py --model models/othello.pt --port 8001
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import Header, HTTPException
from pydantic import BaseModel

# 让 "python othello/serve.py" 无论从哪启动都能找到包
_OTHELLO_DIR = Path(__file__).resolve().parent
_ROOT = _OTHELLO_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.arena import ArenaSession  # noqa: E402
from shared.dqn import DQNAgent, QNetwork  # noqa: E402
from shared.eval import (benchmark as run_benchmark,  # noqa: E402
                         load_benchmark, register_evaluator)
from shared.platform import ModelPlayer  # noqa: E402
from shared.protocol import AgentService, StepRequest, build_app  # noqa: E402
from shared.registry import list_models, resolve  # noqa: E402
from othello.dqn import OthelloDQNConfig, make_agent  # noqa: E402
from othello.duel import DuelSession, play_one, replay_boards  # noqa: E402
from othello.env import (BOARD_SIZE, BLACK, N_ACTIONS, OthelloEnv,  # noqa: E402
                         WHITE)
from othello.selfplay import evaluate_win_rate  # noqa: E402

# 颜色编号 → 中文名(前端展示)
COLOR_NAME = {BLACK: "黑", WHITE: "白"}


# ---------------- 统一评测基准:黑白棋评测器 ----------------
@register_evaluator("othello")
def _eval_othello(path: str, games: int = 30) -> dict:
    """在统一基准下给任意黑白棋模型打分:对"纯随机对手"的胜率。

    关键点:evaluate_win_rate 内部用固定种子的**独立**随机源产生
    对手落子,所有模型面对的是同一批随机棋(统一基准的本质),
    且不污染进程里的全局随机状态。
    """
    from shared.dqn import DQNAgent
    agent = DQNAgent.load(path)
    wr = evaluate_win_rate(agent, games=games, seed=2026)
    return {"score": round(wr, 4), "detail": f"对随机胜率 · {games} 局"}


class PlayerRequest(BaseModel):
    """扩展路由的请求:设置"人执什么色"。"""
    human: str            # "black" | "white" | "none"(纯 AI 对战)


class DuelRequest(BaseModel):
    """模型对战请求:黑/白各选一个模型(random=随机权重),打几局。"""
    black: str = "random"
    white: str = "random"
    games: int = 10


class OthelloService(AgentService):
    """黑白棋服务的游戏特有部分。

    与 SnakeService 相同的职责:维持"每个会话一局独立的棋"(带锁防并发)、
    实现协议 6 个方法、实现换模型入口。额外多了"人执哪色"的设置。
    模型/对战/竞技场是全局共享的,只有对局状态按会话隔离——
    多个浏览器标签页各下各的棋,互不干扰。
    """

    game_name = "othello"
    _SESSION_TTL = 30 * 60        # 会话 30 分钟无活动就回收
    _MAX_SESSIONS = 256           # 会话数上限:防止恶意刷会话头撑爆内存
    _DEFAULT_SESSION = "default"  # 不带会话头的请求(如 curl 测试)用这个

    def __init__(self, model_path: str | None = None):
        self.agent = make_agent(OthelloDQNConfig())
        self._lock = threading.RLock()     # 可重入锁:reset→snapshot 嵌套安全
        # 会话表:session_id -> {"env": 棋盘, "history": 落子序列,
        #                        "human": 人执色, "seen": 最后活跃时间}
        self._sessions: dict[str, dict] = {}
        self.model_path = None
        self.duel = DuelSession()          # 黑 vs 白 模型对战(全局共享)
        self.arena = ArenaSession()        # 循环赛打榜(全局共享)
        self._state_for(None)              # 先开局,否则拿不到第一帧
        if model_path and Path(model_path).exists():
            # 加载失败(文件损坏等)回退随机初始模型,服务照常启动
            try:
                self._load_weights(model_path)
            except Exception as e:
                print(f"[othello] 模型加载失败({e}),使用随机初始模型",
                      file=sys.stderr)

    # ---------------- 会话管理:每个标签页一盘独立的棋 ----------------
    def _state_for(self, session: str | None) -> dict:
        """取(或新建)某个会话的对局状态。会话隔离的核心。"""
        sid = session or self._DEFAULT_SESSION
        now = time.monotonic()
        with self._lock:
            self._gc_sessions(now)
            st = self._sessions.get(sid)
            if st is None:
                if len(self._sessions) >= self._MAX_SESSIONS:
                    # 上限保护:客户端可以用唯一会话头无限刷会话,
                    # 超限直接拒绝,防止内存被撑爆。
                    raise HTTPException(429, "会话数已达上限,请稍后再试")
                env = OthelloEnv()
                env.reset()
                st = {"env": env, "history": [], "human": BLACK, "seen": now}
                self._sessions[sid] = st
            st["seen"] = now
            return st

    def _gc_sessions(self, now: float):
        """回收超时会话,防止标签页越开越多撑爆内存(调用方需持锁)。"""
        stale = [sid for sid, st in self._sessions.items()
                 if now - st["seen"] > self._SESSION_TTL]
        for sid in stale:
            del self._sessions[sid]

    # ---------------- 元数据:告诉前端"这个游戏长什么样" ----------------
    def meta(self) -> dict:
        layers = [m for m in self.agent.online.net
                  if isinstance(m, torch.nn.Linear)]
        # 注意:meta 是全局接口,不再读取任何会话的状态(以前会取默认
        # 会话的"人执色",多标签页各设不同执色时会失真,还会顺带
        # 续命默认会话)。players 信息改由 /api/state 的 human_color 提供。
        return {
            "game": "othello",
            "board": {"type": "grid", "n": BOARD_SIZE},
            # 观察 = 3 通道(己方/对方/空) × 64 格。前端按"通道"渲染摘要
            "obs": {
                "dim": 3 * N_ACTIONS,
                "channels": [
                    {"key": "own", "label": "己方棋子"},
                    {"key": "opp", "label": "对方棋子"},
                    {"key": "empty", "label": "空格"},
                ],
            },
            # 动作 = 64 个格子(0~63,row*8+col)
            "actions": {"type": "discrete", "n": N_ACTIONS,
                        "names": [f"格{i}" for i in range(N_ACTIONS)]},
            "algorithm": "dqn-selfplay",
            "model": self._model_info(layers),
        }

    def _model_info(self, layers=None) -> dict:
        """当前模型信息(独立出来,避免 snapshot/config 嵌套调 meta())。"""
        if layers is None:
            layers = [m for m in self.agent.online.net
                      if isinstance(m, torch.nn.Linear)]
        return {
            "loaded": self.model_path is not None,
            "name": self.model_path or "随机初始模型",
            "input_dim": layers[0].in_features,
            "output_dim": layers[-1].out_features,
            "hidden_dims": [m.out_features for m in layers][:-1],
            "n_actions": N_ACTIONS,
        }

    # ---------------- 快照:当前局面 + 模型看到的一切 ----------------
    def snapshot(self, session: str | None = None) -> dict:
        with self._lock:
            st = self._state_for(session)
            env = st["env"]
            obs = env._get_obs().tolist()
            return {
                "state": self._serialize_state(st),
                "obs": {"dim": len(obs), "channels": self._obs_channels(obs)},
                "q": self._q_for(env, env.current),
                "epsilon": self.agent.epsilon,
                "model": self._model_info(),
            }

    def reset(self, session: str | None = None) -> dict:
        with self._lock:
            st = self._state_for(session)
            st["env"].reset()
            st["history"].clear()
            return self.snapshot(session)

    def step(self, req: StepRequest, session: str | None = None) -> dict:
        with self._lock:
            st = self._state_for(session)
            env = st["env"]
            if env.winner is not None:
                raise HTTPException(400, "本局已结束,请先重新开始")

            current = env.current
            legal = env.legal_moves(current)
            pre_obs = env._get_obs()
            pre_q = self._q_for(env, current)

            # 决定动作:模型决策(纯贪心) or 人类点格子
            if req.ai:
                # 回合秩序校验:人机模式下,人的回合不接受 ai 指令——
                # 防止任何客户端越权替人类落子(纯 AI 对战 human=None 不受限)
                if st["human"] is not None and current == st["human"]:
                    raise HTTPException(400, "当前是人的回合,不能由 AI 落子")
                mask = np.zeros(N_ACTIONS, dtype=bool)
                mask[legal] = True
                action = self.agent.select_action(pre_obs, greedy=True,
                                                  legal_mask=mask)
            elif req.action is not None:
                # 同理:AI 的回合不接受人类指令
                if st["human"] is not None and current != st["human"]:
                    raise HTTPException(400, "当前是 AI 的回合,请等 AI 落子")
                action = int(req.action)
                if action not in legal:
                    raise HTTPException(
                        400, f"落子 {action} 非法,当前合法落子: {legal}")
            else:
                raise HTTPException(400, "必须指定 ai=True 或 action=0~63")

            r, c = divmod(action, BOARD_SIZE)
            obs, reward, done, info = env.step(action)
            st["history"].append(action)     # 记录落子:悔棋/复盘用
            return {
                "pre_obs": pre_obs.tolist(),
                "q": pre_q,
                "action": action,
                "action_rc": [r, c],              # 落子位置(前端高亮)
                "reward": reward,                 # 恒 0(终局统一结算)
                "done": done,
                "flips": env.last_flips,     # 这次翻转了多少子
                "epsilon": self.agent.epsilon,
                "state": self._serialize_state(st),
            }

    def curve(self) -> dict:
        """训练曲线(协议统一用 episodes + scores;黑白棋 scores = 对随机胜率)。"""
        path = _ROOT / "data" / "othello_curve.json"
        if not path.exists():
            return {"episodes": [], "scores": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"episodes": data.get("episodes", []),
                "scores": data.get("win_rates", [])}

    def config(self) -> dict:
        cfg = OthelloDQNConfig()
        info = self._model_info()
        return {
            "algorithm": "dqn-selfplay",
            "lr": cfg.lr, "gamma": cfg.gamma, "batch_size": cfg.batch_size,
            "buffer_capacity": cfg.buffer_capacity,
            "target_update_freq": cfg.target_update_freq,
            "epsilon_start": cfg.epsilon_start, "epsilon_end": cfg.epsilon_end,
            "epsilon_decay": cfg.epsilon_decay,
            "random_opponent_prob": cfg.random_opponent_prob,
            "opponent_pool_size": cfg.opponent_pool_size,
            "hidden_dims": info["hidden_dims"],
            "n_actions": N_ACTIONS,
            "input_dim": info["input_dim"],
            "epsilon": self.agent.epsilon,
        }

    def train_status(self) -> dict:
        """训练进程实时状态(训练/Web 解耦)。

        训练在独立进程跑(python othello/train.py),每评估一次就把进度
        原子写到 data/othello_progress.json;这里只读不写,返回给前端。
        文件不存在 → 还没开始过训练,返回 idle。
        """
        path = _ROOT / "data" / "othello_progress.json"
        if not path.exists():
            return {"status": "idle", "running": False,
                    "message": "尚未开始训练 · 运行 python othello/train.py",
                    "episode": 0, "episodes": 0,
                    "win_rate": None, "epsilon": None,
                    "opponent_pool": 0, "updated_at": None}
        data = json.loads(path.read_text(encoding="utf-8"))
        running = data.get("status") in ("starting", "running")
        return {**data, "running": running}

    # ---------------- 悔棋 / 棋谱(复盘回放的数据来源) ----------------
    def undo(self, session: str | None = None) -> dict:
        """悔棋:撤销落子,重建到之前的局面。

        人机模式下只弹一手是"空操作":弹掉的往往是 AI 刚下的那手,
        轮到 AI 又会在相同局面贪心重下同一手。所以这里连续弹出,
        直到重新轮到人类落子(通常弹 2 手;pass 导致同色连续落子时
        弹 1 手即可)。纯 AI 对战(human=None)弹 1 手。
        终局后也允许悔棋——撤销最后一手重新争取翻盘。
        """
        with self._lock:
            st = self._state_for(session)
            env = st["env"]
            if not st["history"]:
                raise HTTPException(400, "没有可悔的棋")
            human = st["human"]
            # 弹子直到:历史空了,或"当前该落子的一方"是人类
            while st["history"]:
                st["history"].pop()
                self._replay_history(st)
                if human is None:
                    break                    # 纯 AI 对战:只悔一手
                if env.current == human:
                    break                    # 轮到人重新下了,悔棋完成
            return self.snapshot(session)

    def history(self, session: str | None = None) -> dict:
        """当前局完整棋谱:每步落子 + 每一步之后的棋盘(前端复盘回放)。"""
        with self._lock:
            st = self._state_for(session)
            env = st["env"]
            black, white = env.count()
            return {
                "moves": list(st["history"]),
                "boards": replay_boards(st["history"]),
                "result": env.winner,
                "counts": {"black": black, "white": white},
                "steps": env.steps,
            }

    # ---------------- 模型对战(黑 vs 白):选手加载 ----------------
    def _make_duel_agent(self, name: str):
        """加载一个对弈选手:注册表 key / 路径;'random' = 随机权重。"""
        if name == "random":
            return make_agent(OthelloDQNConfig())
        path = resolve(name)
        if not path:
            raise HTTPException(404, f"找不到模型: {name}")
        from shared.dqn import DQNAgent
        return DQNAgent.load(path)

    def _duel_info(self, name: str) -> dict:
        """选手的展示信息(名字 + 注册表里的评估分,便于对比强弱)。"""
        if name == "random":
            return {"name": "随机初始模型", "key": "random", "eval_score": None}
        for m in list_models(self.game_name):
            if m["key"] == name:
                return {"name": m["key"], "key": m["key"],
                        "eval_score": m.get("eval_score")}
        return {"name": name, "key": name, "eval_score": None}

    def _replay_history(self, st: dict):
        """按会话的落子历史重建 env(悔棋 / 测试共用)。"""
        st["env"].reset()
        for action in st["history"]:
            st["env"].step(action)

    # ---------------- 换模型:把 checkpoint 加载进自己的 agent ----------------
    def _load_weights(self, path: str) -> dict:
        with self._lock:
            ckpt = torch.load(path, map_location="cpu")
            # 维度校验:防止把别的游戏的模型(如贪吃蛇 11 维)加载进来——
            # 加载能成功,但之后每步推理都会因维度不匹配而 500。
            if (ckpt["input_dim"] != 3 * N_ACTIONS
                    or ckpt["n_actions"] != N_ACTIONS):
                raise ValueError(
                    f"模型与黑白棋不匹配:输入 {ckpt['input_dim']} 维/输出 "
                    f"{ckpt['n_actions']} 动作,需要 {3 * N_ACTIONS} 维/"
                    f"{N_ACTIONS} 动作")
            net = QNetwork(ckpt["input_dim"], ckpt["hidden_dims"],
                           ckpt["n_actions"])
            net.load_state_dict(ckpt["online"])
            net.eval()
            self.agent.replace_online(net)  # 同步重建 optimizer/目标网络
            self.agent.epsilon = self.agent.epsilon_end
            self.model_path = str(Path(path).relative_to(_ROOT)) \
                if Path(path).is_relative_to(_ROOT) else str(path)
            return self._model_info()

    # ---------------- 扩展:设置"人执什么色" ----------------
    _HUMAN_MAP = {"black": BLACK, "white": WHITE, "none": None}

    def set_human(self, human: str, session: str | None = None) -> dict:
        """human: 'black' / 'white' / 'none'(none = 纯 AI 对战)。"""
        if human not in self._HUMAN_MAP:
            raise HTTPException(400, "human 必须是 black/white/none")
        with self._lock:
            st = self._state_for(session)
            st["human"] = self._HUMAN_MAP[human]
            return self.snapshot(session)

    # ---------------- 内部工具 ----------------
    def _q_for(self, env, player) -> dict:
        """对 player 的所有合法落子跑一次推理,返回按 Q 值排序的候选。

        返回结构(驾驶舱"候选落子 Q 值分布"用):
          {"dim": 64,
           "top": [{"cell": 格号, "r":行, "c":列, "q": Q值}, ...],
           "cells": [64 个值,非法位置为 None]}
        """
        legal = env.legal_moves(player)
        with torch.no_grad():
            t = torch.as_tensor(env._get_obs(player),
                                dtype=torch.float32).unsqueeze(0)
            q = self.agent.online(t).cpu().numpy().flatten()
        cells = [None] * N_ACTIONS
        for i in legal:
            cells[i] = round(float(q[i]), 4)
        top = [{"cell": i, "r": i // BOARD_SIZE, "c": i % BOARD_SIZE, "q": cells[i]}
               for i in sorted(legal, key=lambda i: q[i], reverse=True)]
        return {"dim": N_ACTIONS, "top": top, "cells": cells}

    def _obs_channels(self, obs):
        """把 192 维观察拆成 3 个通道的统计(前端摘要展示)。"""
        n = N_ACTIONS
        return [
            {"key": "own", "label": "己方棋子", "count": int(sum(obs[0:n]))},
            {"key": "opp", "label": "对方棋子", "count": int(sum(obs[n:2 * n]))},
            {"key": "empty", "label": "空格", "count": int(sum(obs[2 * n:]))},
        ]

    def _serialize_state(self, st: dict) -> dict:
        env = st["env"]
        human_color = st["human"]
        info = env._info()
        current = info["current"]
        return {
            "board": info["board"].tolist(),          # 8×8: 1黑 -1白 0空
            "current": current,
            "current_name": COLOR_NAME[current],
            "legal_moves": info["legal_moves"],
            "game_over": info["game_over"],
            "winner": info["winner"],
            "winner_name": (None if info["winner"] is None
                            else ("平局" if info["winner"] == 0
                                  else COLOR_NAME[info["winner"]])),
            "counts": info["counts"],
            "steps": info["steps"],
            "human_color": human_color,
            # 方便前端判断:现在是不是该人 / 该 AI 落子
            "human_turn": (not info["game_over"]
                           and current == human_color),
            "ai_turn": (not info["game_over"]
                        and current != human_color),
        }


# 协议层包装成标准 FastAPI 应用(通用路由见 shared/protocol.py)。
# 注意:导入期就用默认模型路径构建,供 `uvicorn othello.serve:app` 和测试
# 直接使用;命令行 --model 指定其它模型时,main() 会用该模型重建服务。
_DEFAULT_MODEL = str(_ROOT / "models" / "othello.pt")
_service = OthelloService(_DEFAULT_MODEL)
app = build_app(_service, static_dir=_OTHELLO_DIR / "static")
app.state.othello_service = _service   # 挂到 app 上,扩展路由好取用


# 扩展路由:设置"人执黑/白/纯AI对战"(协议允许各游戏自加扩展)
# 对局类扩展路由同样读取 X-Session-Id,和通用路由保持一致的会话隔离
@app.post("/api/player")
def api_player(req: PlayerRequest, x_session_id: str | None = Header(default=None)):
    return app.state.othello_service.set_human(req.human, session=x_session_id)


# 扩展路由:悔棋(撤销最近一步,回到上一步的局面)
@app.post("/api/undo")
def api_undo(x_session_id: str | None = Header(default=None)):
    return app.state.othello_service.undo(session=x_session_id)


# 扩展路由:当前局棋谱(复盘回放数据源)
@app.get("/api/history")
def api_history(x_session_id: str | None = Header(default=None)):
    return app.state.othello_service.history(session=x_session_id)


# ---------------- 扩展路由:模型对战(黑 vs 白) ----------------
@app.post("/api/duel/start")
def api_duel_start(req: DuelRequest):
    """选两个已登记模型(或随机)对弈多局,直观对比谁强。

    参与方统一包成平台 ModelPlayer 再交给对局引擎——
    对战/竞技场/未来的训练器都走同一条"Player 接入"路径。
    """
    svc = app.state.othello_service
    if not 1 <= req.games <= 100:
        raise HTTPException(400, "games 需在 1~100 之间")
    try:
        black = ModelPlayer(svc._make_duel_agent(req.black), name=req.black)
        white = ModelPlayer(svc._make_duel_agent(req.white), name=req.white)
        svc.duel.start(black, white,
                       svc._duel_info(req.black), svc._duel_info(req.white),
                       req.games)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return svc.duel.status()


@app.get("/api/duel/status")
def api_duel_status():
    """对战进度 + 总比分(前端轮询)。"""
    return app.state.othello_service.duel.status()


@app.get("/api/duel/game/{index}")
def api_duel_game(index: int):
    """某一局的完整棋谱(点击局列表 → 复盘回放)。"""
    try:
        return app.state.othello_service.duel.game(index)
    except IndexError as e:
        raise HTTPException(404, str(e))


@app.get("/api/duel/regions")
def api_duel_regions():
    """按 角/边/中心 + 每格 统计落子胜率(热力图数据)。"""
    return app.state.othello_service.duel.regions()


# ---------------- 扩展路由:统一评测基准 + Arena 打榜 ----------------
class ArenaRequest(BaseModel):
    """循环赛请求:每对选手各执黑/白打几局。"""
    games_per_match: int = 10


def _arena_match(black_agent, white_agent):
    """一局竞技场的比赛:返回结果颜色码(黑胜1 / 白胜-1 / 平0)。

    走平台 Match:play_one 内部把 agent 包成 ModelPlayer 交给对局引擎。
    """
    return play_one(black_agent, white_agent)[1]


def _arena_players(svc) -> list[dict]:
    """参赛选手 = 注册表里所有黑白棋模型 + 随机初始模型(垫底基准)。

    全部预加载成 agent(和训练无关,只吃模型文件),比赛里直接调用。
    """
    players = []
    for m in list_models(svc.game_name):
        path = resolve(m["key"])
        if not path:
            continue
        players.append({"key": m["key"], "name": m["key"],
                        "eval_score": m.get("eval_score"),
                        "agent": DQNAgent.load(path)})
    players.append({"key": "random", "name": "随机初始模型",
                    "eval_score": None,
                    "agent": make_agent(OthelloDQNConfig())})
    return players


@app.post("/api/arena/start")
def api_arena_start(req: ArenaRequest):
    """全体模型循环赛:两两各执黑/白打 games_per_match 局,排出总榜。"""
    svc = app.state.othello_service
    if svc.arena.running:
        raise HTTPException(400, "已有一场循环赛在进行中")
    if not 1 <= req.games_per_match <= 50:
        raise HTTPException(400, "games_per_match 需在 1~50 之间")
    players = _arena_players(svc)
    if len(players) < 2:
        raise HTTPException(400, "至少需要 2 个参赛选手(先训练并登记模型)")
    try:
        svc.arena.start(_arena_match, players, req.games_per_match)
    except RuntimeError as e:
        # 并发启动:上面的预检通过后另一请求抢先开了赛 → 400 而非 500
        raise HTTPException(400, str(e))
    return svc.arena.status()


@app.get("/api/arena/status")
def api_arena_status():
    """循环赛进度 + 实时排行榜(前端轮询)。"""
    return app.state.othello_service.arena.status()


@app.get("/api/arena/leaderboard")
def api_arena_leaderboard():
    """循环赛最终(或进行中)的排行榜。"""
    return {"leaderboard": app.state.othello_service.arena.leaderboard()}


@app.post("/api/benchmark")
def api_benchmark_start(games: int = 30):
    """统一评测:把注册表所有模型当场重测(同一基准),生成排行榜。

    只重测 vs 随机对手,比循环赛轻;结果持久化到 data/othello_benchmark.json,
    同时刷新注册表 eval_score——之后所有模型的分数都可比。
    """
    svc = app.state.othello_service
    if svc.arena.running or svc.duel.running:
        raise HTTPException(400, "对战进行中,请稍后再评测")
    if not 1 <= games <= 200:
        raise HTTPException(400, "games 需在 1~200 之间")
    try:
        rows = run_benchmark(svc.game_name, games=games)
    except RuntimeError as e:
        # 并发评测:另一请求正在跑 → 400 而非 500
        raise HTTPException(400, str(e))
    return {"rows": rows}


@app.get("/api/benchmark")
def api_benchmark_get():
    """读取上次持久化的统一评测排行榜(重启后仍可见)。"""
    return load_benchmark("othello")


def main():
    parser = argparse.ArgumentParser(description="黑白棋人机对战服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="模型文件路径或注册表 key;不存在则用随机初始模型")
    args = parser.parse_args()

    global app
    # --model 真正生效:支持注册表 key 或路径;解析不到(不存在)时
    # 传 None,服务回退随机初始模型而不是崩溃。
    model_path = resolve(args.model)
    if model_path != _DEFAULT_MODEL:
        service = OthelloService(model_path)
        app = build_app(service, static_dir=_OTHELLO_DIR / "static")
        app.state.othello_service = service

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
