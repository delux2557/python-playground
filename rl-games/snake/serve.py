"""snake/serve.py —— 贪吃蛇的 Web 服务(实现 shared/protocol.py 的协议)。

这里是"平台化"的示范:游戏特有的部分(环境、模型、元数据)写在
SnakeService 里,通用的 HTTP 路由交给协议层的 build_app() 生成。
以后写黑白棋服务时,只要照着这个类再实现一份 OthelloService 即可,
前端和路由一行都不用改。

启动方式:
  python snake/serve.py --model models/snake.pt --port 8000
"""

import argparse
import json
import threading
import time
from pathlib import Path

import torch

# 让 "python snake/serve.py" 无论从哪启动都能找到包
_SNAKE_DIR = Path(__file__).resolve().parent
_ROOT = _SNAKE_DIR.parent
import sys  # noqa: E402
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import HTTPException  # noqa: E402

from shared.dqn import QNetwork  # noqa: E402
from shared.eval import register_evaluator  # noqa: E402
from shared.protocol import AgentService, StepRequest, build_app  # noqa: E402
from shared.registry import resolve  # noqa: E402
from snake.dqn import SnakeDQNConfig, make_agent  # noqa: E402
from snake.env import SnakeEnv  # noqa: E402

# 动作编号 → 中文名(前端驾驶舱展示用)
ACTION_NAMES = ["上", "下", "左", "右"]


# ---------------- 统一评测基准:贪吃蛇评测器 ----------------
@register_evaluator("snake")
def _eval_snake(path: str, games: int = 20) -> dict:
    """在统一基准下给任意贪吃蛇模型打分:固定种子环境里的平均得分。

    所有模型面对同一批环境(种子固定),得分只取决于模型强弱。
    """
    from shared.dqn import DQNAgent
    from snake.train import evaluate
    agent = DQNAgent.load(path)
    _, avg, _ = evaluate(agent, SnakeDQNConfig(), episodes=games)
    return {"score": round(avg, 3), "detail": f"{games} 局平均分"}


# 观察向量每个维度的含义和分组(与 snake/env.py 的 _get_obs 一一对应)
OBS_MEANING = ["危险·上", "危险·下", "危险·左", "危险·右",
               "食物·水平 dx", "食物·垂直 dy",
               "方向·上", "方向·下", "方向·左", "方向·右", "饥饿度"]
OBS_GROUP = ["danger", "danger", "danger", "danger",
             "food", "food", "dir", "dir", "dir", "dir", "hunger"]


class SnakeService(AgentService):
    """贪吃蛇服务的游戏特有部分。

    只负责三件事:
      1. 维持"每个会话一局独立的蛇"(带锁,防止并发踩坏状态)。
         模型(agent)是全局共享的,只有游戏局面按会话隔离——
         这样多个浏览器标签页各玩各的,互不干扰。
      2. 实现协议要求的 6 个方法(meta/snapshot/reset/step/curve/config)
      3. 实现"换模型"入口 _load_weights
    """

    game_name = "snake"
    _SESSION_TTL = 30 * 60      # 会话 30 分钟无活动就回收
    _MAX_SESSIONS = 256         # 会话数上限:防止恶意刷会话头撑爆内存
    _DEFAULT_SESSION = "default"  # 不带会话头的请求(如 curl 测试)用这个

    def __init__(self, model_path: str | None = None):
        self.agent = make_agent(SnakeDQNConfig())
        # 用"可重入锁"而非普通锁:reset() 内部还会再调 snapshot(),
        # 同一个线程会第二次获取这把锁。普通 Lock 不可重入会直接死锁,
        # RLock 允许同一个线程重复获取,保证 reset → snapshot 嵌套安全。
        self._lock = threading.RLock()
        # 会话表:session_id -> {"env": 蛇环境, "seen": 最后活跃时间}
        self._sessions: dict[str, dict] = {}
        self.model_path = None          # 当前加载的模型文件(相对路径)
        self._env_for(None)             # 先开局,否则拿不到第一帧
        if model_path:
            # 尝试加载模型;文件缺失/损坏时回退随机初始模型,服务照常启动
            try:
                self._load_weights(model_path)
            except Exception as e:
                print(f"[snake] 模型加载失败({e}),使用随机初始模型",
                      file=sys.stderr)

    # ---------------- 会话管理:每个标签页一条独立的蛇 ----------------
    def _env_for(self, session: str | None):
        """取(或新建)某个会话的蛇环境。会话隔离的核心。"""
        sid = session or self._DEFAULT_SESSION
        now = time.monotonic()
        with self._lock:
            self._gc_sessions(now)
            entry = self._sessions.get(sid)
            if entry is None:
                if len(self._sessions) >= self._MAX_SESSIONS:
                    # 上限保护:客户端可以用唯一会话头无限刷会话,
                    # 超限直接拒绝,防止内存被撑爆。
                    raise HTTPException(429, "会话数已达上限,请稍后再试")
                env = SnakeEnv()
                env.reset()
                entry = {"env": env, "seen": now}
                self._sessions[sid] = entry
            entry["seen"] = now
            return entry["env"]

    def _gc_sessions(self, now: float):
        """回收超时会话,防止标签页越开越多撑爆内存(调用方需持锁)。"""
        stale = [sid for sid, e in self._sessions.items()
                 if now - e["seen"] > self._SESSION_TTL]
        for sid in stale:
            del self._sessions[sid]

    # ---------------- 元数据:告诉前端"这个游戏长什么样" ----------------
    def meta(self) -> dict:
        layers = [m for m in self.agent.online.net
                  if isinstance(m, torch.nn.Linear)]
        # 注意:这里用 SnakeDQNConfig().grid_size 而不是 _env_for(None)——
        # meta 是全局接口,不该有"创建/续命默认会话"的副作用。
        return {
            "game": "snake",
            "board": {"type": "grid", "n": SnakeDQNConfig().grid_size},
            "obs": {"dim": len(OBS_MEANING), "meaning": OBS_MEANING,
                    "group": OBS_GROUP},
            "actions": {"type": "discrete", "n": 4, "names": ACTION_NAMES},
            "algorithm": "dqn",
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
            "n_actions": 4,
        }

    # ---------------- 快照:当前局面 + 模型看到的一切 ----------------
    def snapshot(self, session: str | None = None) -> dict:
        with self._lock:
            env = self._env_for(session)
            obs = env._get_obs().tolist()
            return {
                "state": self._serialize_state(env),
                "obs": obs,
                "q_values": self._q_values(obs),
                "epsilon": self.agent.epsilon,
                "model": self._model_info(),
            }

    def reset(self, session: str | None = None) -> dict:
        with self._lock:
            env = self._env_for(session)
            env.reset()
            return self.snapshot(session)

    def step(self, req: StepRequest, session: str | None = None) -> dict:
        with self._lock:
            env = self._env_for(session)
            if env.done:
                raise HTTPException(400, "本局已结束,请先重新开始")
            pre_obs = env._get_obs()
            pre_q = self._q_values(pre_obs)

            # 决定动作:模型决策(纯贪心) or 人类输入
            if req.ai:
                action = self.agent.select_action(pre_obs, greedy=True)
            elif req.action is not None:
                action = int(req.action)
                if not 0 <= action <= 3:
                    raise HTTPException(400, "action 必须是 0~3")
            else:
                raise HTTPException(400, "必须指定 ai=True 或 action=0~3")

            try:
                obs, reward, done, info = env.step(action)
            except ValueError as e:
                raise HTTPException(400, str(e))
            return {
                "pre_obs": pre_obs.tolist(),
                "q_values": pre_q,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "reward": reward,
                "done": done,
                "reason": info.get("reason"),
                "epsilon": self.agent.epsilon,
                "state": self._serialize_state(env),
            }

    def curve(self) -> dict:
        path = _ROOT / "data" / "curve.json"
        if not path.exists():
            return {"episodes": [], "scores": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def config(self) -> dict:
        cfg = SnakeDQNConfig()
        info = self._model_info()
        return {
            "grid_size": cfg.grid_size,
            "lr": cfg.lr, "gamma": cfg.gamma, "batch_size": cfg.batch_size,
            "buffer_capacity": cfg.buffer_capacity,
            "target_update_freq": cfg.target_update_freq,
            "epsilon_start": cfg.epsilon_start, "epsilon_end": cfg.epsilon_end,
            "epsilon_decay": cfg.epsilon_decay,
            "hidden_dims": info["hidden_dims"],
            "n_actions": 4, "input_dim": info["input_dim"],
            "epsilon": self.agent.epsilon,
        }

    # ---------------- 换模型:把 checkpoint 加载进自己的 agent ----------------
    def _load_weights(self, path: str) -> dict:
        with self._lock:
            ckpt = torch.load(path, map_location="cpu")
            # 维度校验:防止把别的游戏的模型(如黑白棋 192 维)加载进来——
            # 加载能成功,但之后每步推理都会因维度不匹配而 500。
            if ckpt["input_dim"] != 11 or ckpt["n_actions"] != 4:
                raise ValueError(
                    f"模型与贪吃蛇不匹配:输入 {ckpt['input_dim']} 维/输出 "
                    f"{ckpt['n_actions']} 动作,需要 11 维/4 动作")
            # 用 checkpoint 里的网络结构重建在线网络(输入/输出维度以文件为准)
            net = QNetwork(ckpt["input_dim"], ckpt["hidden_dims"],
                           ckpt["n_actions"])
            net.load_state_dict(ckpt["online"])
            net.eval()                    # 推理模式
            self.agent.replace_online(net)  # 同步重建 optimizer/目标网络
            self.agent.epsilon = self.agent.epsilon_end  # 加载后默认不探索
            self.model_path = str(Path(path).relative_to(_ROOT)) \
                if Path(path).is_relative_to(_ROOT) else str(path)
            return self._model_info()

    # ---------------- 内部工具 ----------------
    def _q_values(self, obs):
        """跑一次前向推理,得到 4 个方向的 Q 值。"""
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.agent.online(t).cpu().numpy().flatten().tolist()

    def _serialize_state(self, env) -> dict:
        return {
            "grid_size": env.grid_size,
            "score": env.score,
            "steps": env.steps,
            "snake": [list(c) for c in env.body],
            "food": list(env.food),
            "direction": env.direction,
        }


# 协议层把 SnakeService 包装成标准 FastAPI 应用(路由见 shared/protocol.py)。
# 注意:导入期就用默认模型路径构建,供 `uvicorn snake.serve:app` 和测试直接使用;
# 命令行 --model 指定其它模型时,main() 会用该模型重建服务(见下)。
_DEFAULT_MODEL = str(_ROOT / "models" / "snake.pt")
app = build_app(SnakeService(_DEFAULT_MODEL), static_dir=_SNAKE_DIR / "static")


def main():
    parser = argparse.ArgumentParser(description="贪吃蛇 RL 驾驶舱服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="模型文件路径或注册表 key;不存在则用随机初始模型")
    args = parser.parse_args()

    global app
    # --model 真正生效:支持注册表 key 或路径;解析不到(不存在)时
    # 传 None,服务回退随机初始模型而不是崩溃。
    model_path = resolve(args.model)
    if model_path != _DEFAULT_MODEL:
        app = build_app(SnakeService(model_path),
                        static_dir=_SNAKE_DIR / "static")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
