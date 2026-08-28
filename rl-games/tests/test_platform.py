"""对局平台(shared/platform.py)的接口测试。

验证"训练模型 / 人类 / 随机基线都以同一种方式接入平台"的核心抽象:
  1. Player 家族:RandomPlayer / ModelPlayer / HumanPlayer 的 decide 契约
  2. GameAdapter:OthelloAdapter 无状态、多局独立
  3. Match:同步 play() 打完整局、逐步 act()/auto_step() 驱动、
     人类座位无 Player 时 auto_step 报错(回合秩序的天然防线)
"""

import numpy as np
import pytest

from othello.adapter import OthelloAdapter
from othello.dqn import OthelloDQNConfig, make_agent
from othello.duel import play_one
from othello.env import BLACK, WHITE
from shared.platform import (HumanPlayer, Match, ModelPlayer, RandomPlayer)


# ---------------- Player 家族 ----------------
def test_random_player_picks_legal_action():
    adapter = OthelloAdapter()
    game = adapter.new_game()
    p = RandomPlayer(seed=0)
    mask = adapter.legal_mask(game, BLACK)
    a = p.decide(adapter.observe(game, BLACK), mask)
    assert mask[a] == 1


def test_random_player_requires_mask():
    with pytest.raises(ValueError):
        RandomPlayer().decide(np.zeros(192))


def test_model_player_wraps_agent():
    agent = make_agent(OthelloDQNConfig())
    adapter = OthelloAdapter()
    game = adapter.new_game()
    p = ModelPlayer(agent, name="m", greedy=True)
    mask = adapter.legal_mask(game, BLACK)
    a = p.decide(adapter.observe(game, BLACK), mask)
    assert mask[a] == 1


def test_human_player_submit_and_decide():
    p = HumanPlayer(timeout=1)
    p.submit(20)
    assert p.decide(np.zeros(192)) == 20


def test_human_player_rejects_illegal_move():
    adapter = OthelloAdapter()
    game = adapter.new_game()
    p = HumanPlayer(timeout=1)
    p.submit(0)                       # 开局 a1 不合法
    mask = adapter.legal_mask(game, BLACK)
    with pytest.raises(ValueError):
        p.decide(adapter.observe(game, BLACK), mask)


def test_human_player_timeout():
    p = HumanPlayer(timeout=0.05)
    with pytest.raises(TimeoutError):
        p.decide(np.zeros(192))


# ---------------- GameAdapter ----------------
def test_adapter_games_are_independent():
    adapter = OthelloAdapter()
    g1, g2 = adapter.new_game(), adapter.new_game()
    adapter.apply(g1, 19)             # 只在 g1 落子(开局合法点)
    assert not np.array_equal(g1.board, g2.board)
    assert adapter.done(g1) is False and adapter.done(g2) is False


def test_adapter_result_contract():
    adapter = OthelloAdapter()
    game = adapter.new_game()
    r = adapter.result(game)
    assert r["winner"] is None        # 未终局
    assert r["counts"] == {"black": 2, "white": 2}


# ---------------- Match:同步打完整局 ----------------
def test_match_play_full_game_random_vs_random():
    match = Match(OthelloAdapter(), {
        BLACK: RandomPlayer(seed=1),
        WHITE: RandomPlayer(seed=2),
    })
    rec = match.play()
    assert match.done()
    assert rec["winner"] in (BLACK, WHITE, 0)
    assert rec["counts"]["black"] + rec["counts"]["white"] == 64
    assert len(rec["moves"]) >= 1
    for p, a in rec["moves"]:
        assert p in (BLACK, WHITE)
        assert 0 <= a < 64


def test_match_play_model_vs_random():
    agent = make_agent(OthelloDQNConfig())
    match = Match(OthelloAdapter(), {
        BLACK: ModelPlayer(agent),
        WHITE: RandomPlayer(seed=3),
    })
    rec = match.play()
    assert rec["winner"] in (BLACK, WHITE, 0)


# ---------------- Match:逐步驱动(人机对战形态) ----------------
def test_match_step_by_step_human_seat():
    """人类座位不放 Player:act() 注入人类落子,auto_step() 驱动 AI。"""
    agent = make_agent(OthelloDQNConfig())
    match = Match(OthelloAdapter(), {WHITE: ModelPlayer(agent)})  # 人执黑

    assert match.current() == BLACK
    # 轮到人类:auto_step 必须报错(没有注册 Player)——回合秩序防线
    with pytest.raises(KeyError):
        match.auto_step()

    match.act(19)                     # 人类落子(开局合法点 d3)
    assert match.current() == WHITE
    match.auto_step()                 # AI 应手
    assert len(match.moves) == 2
    assert not match.done()


def test_match_act_after_done_raises():
    match = Match(OthelloAdapter(), {
        BLACK: RandomPlayer(seed=4),
        WHITE: RandomPlayer(seed=5),
    })
    match.play()
    with pytest.raises(ValueError):
        match.act(0)


def test_match_reset_clears_history():
    match = Match(OthelloAdapter(), {
        BLACK: RandomPlayer(seed=6),
        WHITE: RandomPlayer(seed=7),
    })
    match.play()
    n = len(match.moves)
    assert n > 0
    match.reset()
    assert match.moves == [] and not match.done()
    assert match.current() == BLACK


# ---------------- 兼容:play_one 接受裸 agent ----------------
def test_play_one_accepts_raw_agents():
    agent = make_agent(OthelloDQNConfig())
    moves, winner, counts = play_one(agent, agent)
    assert winner in (BLACK, WHITE, 0)
    assert counts["black"] + counts["white"] == 64
    assert all(p in (BLACK, WHITE) for p, _ in moves)
