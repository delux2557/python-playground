"""黑白棋环境(othello/env.py)的单元测试。

测试策略:直接构造"确定性的棋盘局面",验证每一条黑白棋规则:
  开局摆位 / 合法动作掩码 / 翻转夹子 / 非法落子 / 自动让子 / 终局计分。
"""

import numpy as np
import pytest

from othello.env import (BOARD_SIZE, BLACK, EMPTY, N_ACTIONS, OBS_DIM,
                         OthelloEnv, WHITE)


def make_env(**kwargs):
    kwargs.setdefault("seed", 0)
    return OthelloEnv(**kwargs)


def flat(r, c):
    """把 (row, col) 转成一维动作编号。"""
    return r * BOARD_SIZE + c


# ---------------- 开局 ----------------
def test_reset_initial_setup():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.current == BLACK                 # 黑先手
    assert env.winner is None                   # 未终局
    c = BOARD_SIZE // 2
    assert env.board[c - 1, c - 1] == WHITE
    assert env.board[c - 1, c] == BLACK
    assert env.board[c, c - 1] == BLACK
    assert env.board[c, c] == WHITE
    assert info["current"] == BLACK


def test_initial_legal_moves_are_four_center_cells():
    """标准开局黑方的合法落子恰好是中央 4 格(能夹住白子的位置)。"""
    env = make_env()
    env.reset()
    moves = env.legal_moves(BLACK)
    assert sorted(moves) == sorted([flat(2, 3), flat(3, 2), flat(4, 5), flat(5, 4)])


# ---------------- 观察编码 ----------------
def test_obs_three_channels_match_board():
    env = make_env()
    env.reset()
    own = obs_own, obs_opp, obs_empty = np.split(env._get_obs(BLACK), 3)
    assert obs_own.sum() == 2                     # 开局黑子 2 颗
    assert obs_opp.sum() == 2                     # 白子 2 颗
    assert obs_empty.sum() == 60                  # 空 60 格
    assert obs_own.shape == (64,) and obs_opp.shape == (64,) and obs_empty.shape == (64,)


def test_obs_symmetric_between_colors():
    """黑白视角互为镜像:黑视角的"己方通道"应等于白视角的"对方通道"。"""
    env = make_env()
    env.reset()
    black_own = np.split(env._get_obs(BLACK), 3)[0]
    white_opp = np.split(env._get_obs(WHITE), 3)[1]
    assert np.array_equal(black_own, white_opp)


# ---------------- 落子与翻转 ----------------
def test_move_flips_flanked_discs():
    """黑在 (2,3) 落子,应把 (3,3) 的白子翻成黑子。"""
    env = make_env()
    env.reset()
    flipped = env._flip(2, 3, BLACK)
    assert flipped == 1
    assert env.board[3, 3] == BLACK
    black, white = env.count()
    assert (black, white) == (4, 1)


def test_step_flips_and_switches_turn():
    env = make_env()
    env.reset()
    obs, reward, done, info = env.step(flat(2, 3))
    assert reward == 0.0            # 每步无即时奖励(终局才结算)
    assert done is False
    assert env.board[3, 3] == BLACK
    assert env.current == WHITE     # 黑落完轮到白
    assert info["current"] == WHITE


def test_illegal_move_raises():
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(0)                 # 角落 (0,0) 开局不合法(夹不住子)
    with pytest.raises(ValueError):
        env.step(flat(3, 3))        # 已有棋子的格子不能落子
    with pytest.raises(ValueError):
        env.step(-1)                # 越界


def test_edge_move_flips_whole_line():
    """白在边界 (3,0) 落子,应翻转一条线上的黑子。"""
    env = make_env()
    env.reset()
    # 手动摆一个局面:白从左侧夹住一行黑子
    env.board[:] = EMPTY
    env.board[3, 0] = EMPTY          # 落子点
    env.board[3, 1] = BLACK
    env.board[3, 2] = BLACK
    env.board[3, 3] = WHITE
    env.board[3, 4] = WHITE
    env.current = WHITE
    env._flip(3, 0, WHITE)
    assert env.board[3, 1] == WHITE and env.board[3, 2] == WHITE


# ---------------- 自动让子与终局 ----------------
def test_board_full_ends_game_with_winner():
    """白填掉最后一格 → 黑无棋可下 → 自动让子 → 双方都无棋 → 终局计分。"""
    env = make_env()
    env.reset()
    env.board[:] = BLACK                    # 先全部填黑
    env.board[3, 3] = EMPTY                 # 留一个空
    env.board[3, 2] = BLACK
    env.board[3, 1] = BLACK
    env.board[3, 0] = WHITE                 # 白落 (3,3) 能夹回 2 颗黑
    env.current = WHITE

    obs, reward, done, info = env.step(flat(3, 3))
    assert done is True
    # 翻转生效:被夹住的 2 颗黑变白
    assert env.board[3, 1] == WHITE and env.board[3, 2] == WHITE
    # 终局按子数计分:白吃到 3 个格子(3,0/3,1/3,2 + 落子 3,3 = 4 子),
    # 其余 60 格还是黑 → 黑胜。这里验证的是"子多者胜"这条规则。
    assert env.count() == (60, 4)
    assert env.winner == BLACK
    assert info["game_over"] is True


def test_random_games_always_terminate():
    """随机乱下也必须能终局(≤64 子 + 自动让子),且结果合法。"""
    env = make_env()
    for _ in range(10):
        env.reset()
        guard = 0
        while env.winner is None:
            legal = env.legal_moves(env.current)
            # env 保证当前方至少有一步(否则已自动让子/终局)
            assert legal, "当前方应该有合法落子"
            env.step(int(np.random.choice(legal)))
            guard += 1
            assert guard <= 64, "一局最多 60 子,不可能超过 64 步"
        assert env.winner in (BLACK, WHITE, 0)


# ---------------- 辅助契约 ----------------
def test_state_grid_contract():
    env = make_env()
    env.reset()
    grid = env.state_grid()
    assert grid.shape == (BOARD_SIZE, BOARD_SIZE)
    assert set(np.unique(grid)) <= {BLACK, WHITE, EMPTY}
    assert (grid == BLACK).sum() == 2 and (grid == WHITE).sum() == 2


def test_legal_moves_and_mask_consistent():
    env = make_env()
    env.reset()
    moves = env.legal_moves(BLACK)
    mask = env.legal_mask(BLACK).flatten()
    assert sorted(moves) == sorted(np.flatnonzero(mask).tolist())
