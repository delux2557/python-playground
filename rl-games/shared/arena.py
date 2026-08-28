"""shared/arena.py —— 循环赛打榜引擎(模型互相切磋,排出总榜)。

和 duel.py(一对一)的区别:
  duel    : 你亲手挑两个模型打一架,看单场比分;
  arena   : 把注册表里"所有"模型放进来,两两都打(黑方白方各打一遍,
            消除先后手偏差),最后按总积分排榜——谁强一目了然。

为什么"黑方白方各打一遍":
  黑白棋有先手优势,只让 A 一直执黑会不公平。每对选手
  各执黑/执白打 games_per_match 局,积分制(胜 1 / 平 0.5 / 负 0),
  最后按积分 + 胜率排榜。

本引擎是"通用的":它不知道棋盘长什么样,只依赖两样东西——
  match_fn(black_agent, white_agent) -> 结果颜色码(黑/白/0 平)
  players: [{key, name, eval_score, agent}]  谁参赛、用什么模型
所以 snake/othello 都能用同一套引擎,只需各自提供 match_fn。
"""

import json
import threading
from dataclasses import dataclass, field

# 结果颜色码(与游戏解耦:由 match_fn 返回,本模块只当整数用)
BLACK_WIN = 1      # 黑方胜
WHITE_WIN = -1     # 白方胜
DRAW = 0           # 平局


@dataclass
class ArenaPlayer:
    """一个参赛选手 + 累计战绩。"""
    key: str
    name: str
    eval_score: float | None
    agent: object = None                     # 已加载的模型(可空,惰性加载)
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: int = 0
    points: float = 0.0
    score: float = 0.0                       # 积分 / 总对局数(排序用)
    # 头对头:opponent_key -> {"win","draw","loss","games"}
    head_to_head: dict = field(default_factory=dict)

    def record(self, opp_key: str, outcome: str):
        """记一场结果。outcome: "win" / "draw" / "loss"(本选手视角)。"""
        self.games += 1
        if outcome == "win":
            self.wins += 1
            self.points += 1.0
        elif outcome == "draw":
            self.draws += 1
            self.points += 0.5
        else:
            self.losses += 1
        h = self.head_to_head.setdefault(opp_key,
                                         {"win": 0, "draw": 0, "loss": 0,
                                          "games": 0})
        h[outcome] += 1
        h["games"] += 1
        self.score = self.points / self.games if self.games else 0.0


class ArenaSession:
    """一场循环赛。用法:
        session.start(match_fn, players, games_per_match)
        后台线程打完全部组合;轮询 status() 看进度,结束后 leaderboard()。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._thread = None
        self.match_fn = None
        self.players: list[ArenaPlayer] = []
        self.games_per_match = 0
        self._total_games = 0
        self._played = 0
        self.running = False
        self.error = None
        self.schedule: list[tuple] = []      # [(i, j, color)] 待打组合

    # ---------------- 启动 ----------------
    def start(self, match_fn, players, games_per_match=10):
        """启动循环赛。
        match_fn(black_agent, white_agent) -> 结果颜色码(黑/白/0)。
        players: [{key, name, eval_score, agent}]。
        """
        with self._lock:
            if self.running:
                raise RuntimeError("已有一场循环赛在进行中")
            self.match_fn = match_fn
            self.players = [ArenaPlayer(**p) for p in players]
            self.games_per_match = int(games_per_match)
            self._played = 0
            self.error = None
            self.running = True
            # 排赛程:每对选手各执黑/执白打 games_per_match 局
            n = len(self.players)
            self.schedule = []
            for i in range(n):
                for j in range(i + 1, n):
                    for _ in range(self.games_per_match):
                        self.schedule.append((i, j, BLACK_WIN))   # i 执黑
                        self.schedule.append((j, i, WHITE_WIN))   # j 执黑
            self._total_games = len(self.schedule)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    # ---------------- 状态 / 排行榜 ----------------
    def status(self) -> dict:
        """进度 + 实时排行榜(前端轮询用)。"""
        with self._lock:
            return {
                "running": self.running,
                "played": self._played,
                "total": self._total_games,
                "progress": round(self._played / self._total_games, 4)
                            if self._total_games else 1.0,
                "games_per_match": self.games_per_match,
                "error": self.error,
                "leaderboard": self._leaderboard(),
            }

    def leaderboard(self) -> list[dict]:
        """打完(或进行中)的排行榜,按积分/胜率从高到低。"""
        with self._lock:
            return self._leaderboard()

    def _leaderboard(self) -> list[dict]:
        rows = []
        for p in self.players:
            rows.append({
                "key": p.key, "name": p.name, "eval_score": p.eval_score,
                "wins": p.wins, "draws": p.draws, "losses": p.losses,
                "games": p.games, "points": round(p.points, 1),
                "score": round(p.score, 4),
                "win_rate": round((p.wins + 0.5 * p.draws) / p.games, 4)
                            if p.games else None,
                # 深拷贝:锁释放后 FastAPI 才做 JSON 序列化,若返回活引用,
                # 后台对局线程此刻正在 record() 里改它 → 序列化损坏/报错。
                "head_to_head": json.loads(json.dumps(p.head_to_head)),
            })
        # 排序:积分多 → 胜率高 → 负场少,给出明确的名次
        rows.sort(key=lambda r: (r["points"], r["score"], -r["losses"]),
                  reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return rows

    # ---------------- 内部 ----------------
    def _run(self):
        try:
            for i, j, _color in self.schedule:
                # 赛程元组的第一个元素就是执黑者((i,j,BLACK) i 执黑,(j,i,WHITE) j 执黑),
                # color 只是"谁执黑"的标记,不再参与换位,避免两局都同一个人执黑。
                black, white = self.players[i], self.players[j]
                # 单局容错:某一局崩了(坏模型/数值异常)不该终止整场
                # 循环赛——记为该选手弃权负,继续打剩下的赛程。
                try:
                    # 比赛函数返回"黑方视角"的结果码,换算成双方各自的战绩
                    result = self.match_fn(black.agent, white.agent)
                except Exception as e:
                    with self._lock:
                        self.error = f"第 {self._played + 1} 局异常(已跳过): {e}"
                        white.record(black.key, "win")
                        black.record(white.key, "loss")
                        self._played += 1
                    continue
                with self._lock:
                    if result == BLACK_WIN:
                        black.record(white.key, "win")
                        white.record(black.key, "loss")
                    elif result == WHITE_WIN:
                        white.record(black.key, "win")
                        black.record(white.key, "loss")
                    else:
                        black.record(white.key, "draw")
                        white.record(black.key, "draw")
                    self._played += 1
        except Exception as e:
            with self._lock:
                self.error = str(e)
        finally:
            with self._lock:
                self.running = False
