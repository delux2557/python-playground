"""othello/duel.py —— 模型对战(黑 vs 白)与分区胜率统计。

平台化的一个自然延伸:注册表里已经登记了各个模型和它们的 eval_score,
"谁更强"不能只看单次评估——把两个模型放进同一张棋盘,一个执黑一个执白,
打多局,用比分说话,是最直观的验证方式(类似 AlphaGo 的"内部对战选拔")。

对局引擎已上提到平台层:本模块不再手写回合循环,而是把两个参与方
(模型/随机/训练器,都是 Player)放进 shared/platform.py 的 Match,
由 Match 负责回合仲裁与棋谱记录。本模块只保留黑白棋特有的部分:
  1. replay_boards : 给定一局落子序列,重放出"每步之后的完整棋盘"(复盘用)
  2. region_of     : 把 64 个格子分成 角/边/中心 三个战略区(黑白棋经典分区)
  3. DuelSession   : 管理一场"模型对战"——后台线程逐局对弈、记录每局
                     结果与落子、实时统计进度、事后聚合分区胜率
"""

import threading
from dataclasses import dataclass, field

import numpy as np

from othello.adapter import OthelloAdapter
from othello.env import BLACK, WHITE
from shared.platform import Match, ModelPlayer, Player

# 分区定义(黑白棋布子战略里最经典的三个概念):
#   角   : 四个角落,拿住就极难被翻,通常是最优落子;
#   边   : 紧贴边界但非角落,容易被"贴边"的战术卡住,双刃剑;
#   中心 : 棋盘中部,争夺激烈但早晚会被翻转,不是终局保险。
CORNER = "corner"
EDGE = "edge"
CENTER = "center"
REGION_NAMES = {CORNER: "角", EDGE: "边", CENTER: "中心"}

_RANK = {CORNER: 0, EDGE: 1, CENTER: 2}   # 排序用,不是重要性排名


def region_of(cell: int, size: int = 8) -> str:
    """返回格子 cell(0~63,row*8+col)所属的战略分区。"""
    r, c = divmod(cell, size)
    if r in (0, size - 1) and c in (0, size - 1):
        return CORNER
    if r in (0, size - 1) or c in (0, size - 1):
        return EDGE
    return CENTER


def replay_boards(moves, size: int = 8):
    """把一局落子序列重放成"每一步之后的棋盘"序列。

    返回 list,第 0 项是初始局面,第 k 项是第 k 步落完后的局面
    (每个局面是 size×size 的整数数组:1黑 -1白 0空)。
    复盘回放时前端逐格渲染任意一帧即可,不需要前端懂翻转规则。
    """
    from othello.env import OthelloEnv
    env = OthelloEnv(size=size)
    boards = [env.state_grid().tolist()]
    for action in moves:
        env.step(int(action))
        boards.append(env.state_grid().tolist())
    return boards


def _as_player(x, seat: int) -> Player:
    """把"agent 或 Player"统一成 Player。

    对战/竞技场历史上传的是 DQNAgent;平台层统一用 Player。这里做兼容:
    已是 Player 直接用,否则包一层 ModelPlayer(纯贪心,不探索)。
    """
    if isinstance(x, Player):
        return x
    return ModelPlayer(x, name=f"seat{seat}", greedy=True)


def play_one(black, white) -> tuple:
    """用两个参与方打一局,返回 (moves, winner, counts)。

    moves 是 [(落子方, action), ...] 的列表。为什么必须记落子方:
    黑白棋有"自动让子",一方无棋可下时会被跳过,所以实际落子方并不
    严格黑白交替——只记 action 会推断错(见 DuelSession.regions)。

    内部走平台 Match:回合仲裁、合法性、终局判定全由 Match/Adapter 负责。
    """
    match = Match(OthelloAdapter(), {
        BLACK: _as_player(black, BLACK),
        WHITE: _as_player(white, WHITE),
    })
    rec = match.play()
    return rec["moves"], rec["winner"], rec["counts"]


# 兼容旧名:测试与 serve 里仍以 _play_one 引用
_play_one = play_one


@dataclass
class DuelGame:
    """一场对局的结果 + 完整落子记录。"""
    index: int
    result: int                       # BLACK / WHITE / 0(平)
    counts: dict                      # {"black": n, "white": n}
    moves: list = field(default_factory=list)   # [(落子方, action), ...]


class DuelSession:
    """一场"黑 vs 白"模型对战。

    用法:
      session.start(black, white, black_info, white_info, games)
      后台线程逐局对弈;前端轮询 status() 看进度,结束后再调
      game(i) 复盘某一局、regions() 看分区胜率。

    black/white 可以是 DQNAgent(旧用法)或 Player(平台用法)。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._thread = None
        self.black = None            # {"name","key","eval_score"}
        self.white = None
        self.games = 0
        self.results: list[DuelGame] = []
        self.running = False
        self.error = None

    # ---------------- 启动 ----------------
    def start(self, black, white, black_info, white_info, games):
        """启动一场对战(参与方已就绪,不在此处加载)。"""
        with self._lock:
            if self.running:
                raise RuntimeError("已有一场对战在进行中")
            self.black = black_info
            self.white = white_info
            self.games = int(games)
            self.results = []
            self.error = None
            self.running = True
            self._thread = threading.Thread(
                target=self._run, args=(black, white, self.games),
                daemon=True)
            self._thread.start()

    # ---------------- 状态 / 数据 ----------------
    def status(self) -> dict:
        """对局进度 + 总比分(前端轮询用)。"""
        with self._lock:
            wins = {"black": 0, "white": 0, "draws": 0}
            for g in self.results:
                if g.result == BLACK:
                    wins["black"] += 1
                elif g.result == WHITE:
                    wins["white"] += 1
                else:
                    wins["draws"] += 1
            played = len(self.results)
            return {
                "running": self.running,
                "black": self.black,
                "white": self.white,
                "games": self.games,
                "played": played,
                "black_wins": wins["black"],
                "white_wins": wins["white"],
                "draws": wins["draws"],
                "black_win_rate": round(wins["black"] / played, 4) if played else None,
                "error": self.error,
                "results": [
                    {"index": g.index, "result": g.result,
                     "counts": g.counts, "moves": len(g.moves)}
                    for g in self.results
                ],
            }

    def game(self, index: int) -> dict:
        """某一局的完整棋谱:落子序列 + 每一步后的棋盘(复盘回放用)。"""
        with self._lock:
            if not (0 <= index < len(self.results)):
                raise IndexError(f"没有第 {index} 局(已打 {len(self.results)} 局)")
            g = self.results[index]
            return {
                "index": g.index,
                "result": g.result,
                "counts": g.counts,
                "moves": [a for _, a in g.moves],
                "players": [p for p, _ in g.moves],
                "boards": replay_boards([a for _, a in g.moves]),
            }

    def regions(self) -> dict:
        """按 角/边/中心 + 每个格子 统计"落子后最终获胜"的比例。

        直觉:把几百个落子按"落在哪一格"分桶,统计这些落子对应的
        对局里落子方最终赢的比例。样本越多越能看出"哪些位置是好棋"——
        经典结论:角胜率高、中心低,和职业棋手的直觉一致。
        """
        with self._lock:
            size = 8
            win = np.zeros((size, size), dtype=int)
            total = np.zeros((size, size), dtype=int)
            for g in self.results:
                for player, action in g.moves:
                    r, c = divmod(action, size)
                    total[r, c] += 1
                    if g.result == player:
                        win[r, c] += 1

            # 每个格子的胜率(None = 没有样本);无样本统一记 None,前端显示 "—"
            grid = [
                [round(win[r, c] / total[r, c], 4) if total[r, c] else None
                 for c in range(size)]
                for r in range(size)
            ]
            return {
                "samples": int(total.sum()),
                "grid": grid,
                "regions": self._region_stats(win, total),
            }

    # ---------------- 内部 ----------------
    def _run(self, black, white, games):
        """后台线程:逐局对弈。每局结束把结果追加进列表(加锁保护)。"""
        try:
            for i in range(games):
                moves, result, counts = play_one(black, white)
                with self._lock:
                    self.results.append(DuelGame(
                        index=i, result=result, counts=counts, moves=moves))
        except Exception as e:                       # 防止线程静默崩掉
            with self._lock:
                self.error = str(e)
        finally:
            with self._lock:
                self.running = False

    def _region_stats(self, win, total):
        """聚合三个分区的总胜率(所有落子合并统计)。"""
        stats = {}
        for region in (CORNER, EDGE, CENTER):
            w = t = 0
            for r in range(8):
                for c in range(8):
                    if region_of(r * 8 + c) != region:
                        continue
                    w += int(win[r, c])
                    t += int(total[r, c])
            stats[region] = {
                "name": REGION_NAMES[region],
                "win": int(w), "total": int(t),
                "win_rate": round(w / t, 4) if t else None,
            }
        return stats
