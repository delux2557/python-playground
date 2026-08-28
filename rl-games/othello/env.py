"""othello/env.py —— 黑白棋(Reversi / Othello)游戏环境。

这是一个"Gymnasium 风格"的环境,和贪吃蛇共用同一套交互方式:
    obs, info  = env.reset()              # 开局
    obs, reward, done, info = env.step(a) # 落一子
好处:训练脚本 / 自对弈 / Web 服务都能复用同一份"游戏规则"。

和贪吃蛇的 4 个关键不同(对应 DESIGN.md 第 5.2 节):
  1. 双人零和:棋盘上有 黑(BLACK=1) 和 白(WHITE=-1) 两种棋子,
     你多一个子就是我少一个子,所以赢家通吃、奖励只在终局给。
  2. 动作空间:64 个格子(0~63,row*8+col),而不是 4 个方向。
  3. 合法动作掩码(legal mask):只有"能夹住对方棋子"的位置才能落子,
     模型必须通过掩码选动作,否则会下出非法棋。
  4. 自动让子(pass):某方无棋可下时自动跳过,轮到对方;双方都无棋
     可下(或棋盘下满)才终局,子多者胜。

奖励设计(稀疏但干净):
  每步 reward = 0;胜负在终局由训练方结算(+1 胜 / -1 负 / 0 平)。
  为什么不每步给"当前子数"当奖励?因为那会诱导模型只顾眼前多吃子,
  学会"局中领先、终局翻车"的短视策略(见 DESIGN.md 第 7 章调参速查)。
"""

import numpy as np

BOARD_SIZE = 8                # 标准 8×8 棋盘
N_ACTIONS = BOARD_SIZE * BOARD_SIZE   # 64 个落子位置
OBS_DIM = 3 * N_ACTIONS       # 观察维度:3 通道(己方/对方/空)× 64 = 192

# 棋子取值:1 = 黑, -1 = 白, 0 = 空。
# 用 ±1 而非 1/2 的好处:取反 -player 就得到对手,判断/翻转代码更简洁。
BLACK = 1
WHITE = -1
EMPTY = 0

# 8 个方向(行偏移, 列偏移):上下左右 + 四条对角线
_DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class OthelloEnv:
    """黑白棋环境。

    主要属性(测试和调试可以直接读写):
      board   : 8×8 整数数组,取值 BLACK / WHITE / EMPTY
      current : 当前轮到谁落子(BLACK 或 WHITE)
      steps   : 本局已落子数(不含自动 pass)
      winner  : 终局后的赢家(BLACK / WHITE / 0=平局),未终局为 None
    """

    def __init__(self, size=BOARD_SIZE, seed=None):
        """
        参数说明:
          size: 棋盘边长,默认 8。调成 6/4 能大幅加快训练(实验用),
                但界面/自对弈默认都用标准 8×8。
          seed: 随机种子。本环境其实不用随机(开局固定),留着和其他
                环境接口保持一致,方便将来扩展。
        """
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---------------- 与算法交互的接口 ----------------
    def reset(self):
        """开局:棋盘中央摆好 4 子,黑先手。返回 (obs, info)。"""
        c = self.size // 2
        # 标准开局: 白  黑
        #            黑  白  (d4=白 e4=黑 d5=黑 e5=白)
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.board[c - 1, c - 1] = WHITE
        self.board[c - 1, c] = BLACK
        self.board[c, c - 1] = BLACK
        self.board[c, c] = WHITE

        self.current = BLACK          # 黑方先手
        self.steps = 0
        self.last_flips = 0           # 最近一次落子翻转了几颗子(驾驶舱展示用)
        self.winner = None            # None=未终局;0=平局;±1=对应赢家
        return self._get_obs(), self._info()

    def step(self, action):
        """在格子 action(0~63,row*8+col)落一子。

        返回 (obs, reward, done, info):
          obs   : 新局面的观察(以"下一个落子方"的视角编码)
          reward: 本环境恒为 0.0(奖励由训练方在终局统一结算)
          done  : 是否终局
          info  : 当前局面详情(棋盘/轮到谁/合法落子/计数/胜负等)

        落子非法(越界或不能夹子)会抛 ValueError——算法应该永远只选
        合法动作(掩码保证),抛错是为了及早暴露 bug。
        """
        # 用"实际棋盘大小"校验,而不是写死 N_ACTIONS=64——
        # 否则小棋盘(如 6×6)时动作 36~63 能过检,随后索引越界。
        n = self.size * self.size
        if not (0 <= action < n):
            raise ValueError(f"非法动作 {action}:必须在 0~{n - 1}")

        r, c = divmod(action, self.size)
        if not self.legal_mask(self.current)[r, c]:
            raise ValueError(
                f"非法落子 {action}(第{r}行第{c}列):这里夹不住对方的子")

        # 落子并翻转被夹住的对方棋子(规则核心);记录翻转数供驾驶舱展示
        self.last_flips = self._flip(r, c, self.current)
        self.steps += 1

        # 换手:轮到对方
        self.current = -self.current

        # 自动让子(pass):对方无棋可下 → 自动跳过,轮回到自己
        if not self.legal_mask(self.current).any():
            self.current = -self.current
            # 自己也无棋可下 → 双方都无法落子,终局
            if not self.legal_mask(self.current).any():
                self._finish()

        return self._get_obs(), 0.0, self.winner is not None, self._info()

    # ---------------- 黑白棋规则核心 ----------------
    def legal_mask(self, player=None):
        """返回 8×8 布尔矩阵,True = 该位置对 player 合法。

        规则:落子后必须沿某条直线"夹住"至少一颗对方棋子——
        即从该格出发,某方向上连续是对方棋子,再往后有一颗己方棋子。
        """
        player = self.current if player is None else player
        opp = -player
        mask = np.zeros((self.size, self.size), dtype=bool)
        b = self.board

        for r in range(self.size):
            for c in range(self.size):
                if b[r, c] != EMPTY:          # 已有棋子的格子不能落子
                    continue
                for dr, dc in _DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    # 紧挨着必须先是对方的棋子,否则这个方向不夹子
                    if not (0 <= nr < self.size and 0 <= nc < self.size):
                        continue
                    if b[nr, nc] != opp:
                        continue
                    # 沿方向继续走,直到碰到己方棋子 → 合法;越界/空格 → 不合法
                    while True:
                        nr += dr
                        nc += dc
                        if not (0 <= nr < self.size and 0 <= nc < self.size):
                            break
                        if b[nr, nc] == player:
                            mask[r, c] = True
                            break
                        if b[nr, nc] == EMPTY:
                            break
        return mask

    def legal_moves(self, player=None) -> list[int]:
        """当前(或指定)玩家的所有合法落子位置(平铺成一维动作编号)。"""
        return [int(i) for i in np.flatnonzero(
            self.legal_mask(player).flatten())]

    def count(self):
        """返回 (黑子数, 白子数)。"""
        black = int((self.board == BLACK).sum())
        white = int((self.board == WHITE).sum())
        return black, white

    # ---------------- 观察编码 ----------------
    def _get_obs(self, player=None):
        """把棋盘编码成 192 维向量,作为模型的输入。

        用指定玩家(默认当前落子方)的视角编码 3 个通道:
          [0:64]   该玩家棋子在哪(1.0 / 0.0)
          [64:128] 对方棋子在哪
          [128:192]空格在哪
        好处:同一个网络既会下黑棋也会下白棋(黑白对称),
        不用为每个颜色各训一个模型。

        参数 player 供自对弈使用:某一步结束后要"站在刚才落子的玩家
        视角"重新编码棋盘(此时 env.current 已换到对方),必须显式传入。
        """
        player = self.current if player is None else player
        own = (self.board == player).astype(np.float32)
        opp = (self.board == -player).astype(np.float32)
        empty = (self.board == EMPTY).astype(np.float32)
        return np.concatenate([own.flatten(), opp.flatten(), empty.flatten()])

    def state_grid(self):
        """返回棋盘整型数组(1=黑 -1=白 0=空),供可视化/测试使用。"""
        return self.board.copy()

    # ---------------- 内部工具 ----------------
    def _flip(self, r, c, player):
        """在 (r, c) 落 player 的子,翻转所有被夹住的对方棋子。

        前提:该位置对 player 合法(调用方已用 legal_mask 校验过)。
        返回被翻转的棋子数量(驾驶舱展示用)。
        """
        opp = -player
        flipped = 0
        for dr, dc in _DIRECTIONS:
            nr, nc = r + dr, c + dc
            line = []                       # 这条线上"连续对方棋子"的位置
            while (0 <= nr < self.size and 0 <= nc < self.size
                   and self.board[nr, nc] == opp):
                line.append((nr, nc))
                nr += dr
                nc += dc
            # 只有线尽头是己方棋子才算"被夹住",才翻转
            if (line and 0 <= nr < self.size and 0 <= nc < self.size
                    and self.board[nr, nc] == player):
                for rr, cc in line:
                    self.board[rr, cc] = player
                    flipped += 1
        self.board[r, c] = player
        return flipped

    def _finish(self):
        """双方都无棋可下 → 终局,按子数定胜负。"""
        black, white = self.count()
        if black > white:
            self.winner = BLACK
        elif white > black:
            self.winner = WHITE
        else:
            self.winner = 0                  # 平局

    def _info(self) -> dict:
        """当前局面的完整信息(服务端 / 前端驾驶舱都用它)。"""
        mask = self.legal_mask(self.current)
        return {
            "board": self.board.copy(),
            "current": self.current,
            "legal_moves": [int(i) for i in np.flatnonzero(mask.flatten())],
            "game_over": self.winner is not None,
            "winner": self.winner,
            "counts": {"black": self.count()[0], "white": self.count()[1]},
            "steps": self.steps,
        }
