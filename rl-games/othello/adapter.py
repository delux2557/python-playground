"""othello/adapter.py —— 把黑白棋规则包装成平台的"游戏插件"。

shared/platform.py 定义了平台与游戏的交互契约(GameAdapter),
这里实现黑白棋这一侧:平台靠这组方法开局、问轮到谁、要观察、
查合法落子、落子、判终局——平台完全不懂黑白棋规则。

参与方编号约定:BLACK=1 执黑,WAIT/WHITE=-1 执白(沿用 env 的取值,
取反 -p 即对手,与现有代码一致)。
"""

import numpy as np

from othello.env import BLACK, N_ACTIONS, OthelloEnv, WHITE
from shared.platform import GameAdapter


class OthelloAdapter(GameAdapter):
    """黑白棋规则插件。无状态:每次 new_game() 返回独立环境,
    平台可以同时托管任意多场对局(对战/竞技场/人机各用各的)。"""

    name = "othello"

    def __init__(self, size: int = 8):
        self.size = size

    def new_game(self) -> OthelloEnv:
        return OthelloEnv(size=self.size)

    def current_player(self, game: OthelloEnv) -> int:
        return game.current

    def observe(self, game: OthelloEnv, player: int) -> np.ndarray:
        return game._get_obs(player)

    def legal_mask(self, game: OthelloEnv, player: int):
        return game.legal_mask(player).flatten()

    def apply(self, game: OthelloEnv, action: int):
        game.step(int(action))

    def done(self, game: OthelloEnv) -> bool:
        return game.winner is not None

    def result(self, game: OthelloEnv) -> dict:
        black, white = game.count()
        return {
            "winner": game.winner,          # BLACK / WHITE / 0(平) / None(未终局)
            "counts": {"black": black, "white": white},
        }


# 参与方编号 → 中文名(展示用)
PLAYER_NAME = {BLACK: "黑", WHITE: "白"}
