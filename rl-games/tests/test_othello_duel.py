"""黑白棋"模型对战 + 悔棋 + 复盘"的接口测试。

覆盖(与前端新功能一一对应):
  1. DuelSession:随机模型对局、状态/比分、单局棋谱、分区胜率统计
  2. /api/duel/* 路由:发起、进度、单局复盘、分区热力图
  3. /api/undo 悔棋、/api/history 当前局棋谱
"""

import time

import pytest
from fastapi.testclient import TestClient

from othello.duel import DuelSession, region_of, replay_boards
from othello.env import BLACK, WHITE
from othello.serve import app

client = TestClient(app)


def _wait_duel_done(timeout=30):
    """等待后台对局跑完(TestClient 也是真线程,需要轮询)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get("/api/duel/status").json()
        if not st["running"]:
            return st
        time.sleep(0.1)
    raise AssertionError("对局超时未结束")


# ---------------- 单元:重放与分区 ----------------
def test_replay_boards_length_and_shape():
    # 用真实环境走几步,确保落子序列合法(含自动让子也可能出现)
    from othello.dqn import OthelloDQNConfig, make_agent
    from othello.env import N_ACTIONS, OthelloEnv
    import numpy as np
    env = OthelloEnv()
    agent = make_agent(OthelloDQNConfig())
    moves = []
    for _ in range(3):
        legal = env.legal_moves(env.current)
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[legal] = True
        a = agent.select_action(env._get_obs(), greedy=True, legal_mask=mask)
        env.step(a)
        moves.append(int(a))

    boards = replay_boards(moves)
    assert len(boards) == len(moves) + 1               # 初始 + 每步后
    for b in boards:
        assert len(b) == 8 and all(len(r) == 8 for r in b)
    # 第 0 帧是标准开局 4 子;之后每帧子数递增 1
    assert sum(sum(1 for v in r if v) for r in boards[0]) == 4
    for i in range(1, len(boards)):
        nxt = sum(sum(1 for v in r if v) for r in boards[i])
        assert nxt == 4 + i


def test_region_of_classification():
    assert region_of(0) == "corner"      # a1
    assert region_of(7) == "corner"      # h1
    assert region_of(56) == "corner"     # a8
    assert region_of(63) == "corner"     # h8
    assert region_of(1) == "edge"        # b1 上边
    assert region_of(8) == "edge"        # a2 左边
    assert region_of(3) == "edge"        # d1 上边
    assert region_of(27) == "center"     # d4 中央
    assert region_of(35) == "center"     # e4 中央


def test_play_one_returns_full_records():
    from othello.dqn import OthelloDQNConfig, make_agent
    from othello.duel import _play_one
    agent = make_agent(OthelloDQNConfig())
    moves, result, counts = _play_one(agent, agent)
    assert result in (BLACK, WHITE, 0)
    assert counts["black"] + counts["white"] == 64
    for player, action in moves:
        assert player in (BLACK, WHITE)
        assert 0 <= action < 64


# ---------------- DuelSession 端到端 ----------------
def test_duel_session_random_vs_random():
    from othello.dqn import OthelloDQNConfig, make_agent
    sess = DuelSession()
    a = make_agent(OthelloDQNConfig())
    info = {"name": "随机A", "key": "random", "eval_score": None}
    sess.start(a, a, info, info, games=2)
    deadline = time.time() + 30
    while sess.running and time.time() < deadline:
        time.sleep(0.1)
    assert sess.running is False
    st = sess.status()
    assert st["played"] == 2 and st["games"] == 2
    assert st["black_wins"] + st["white_wins"] + st["draws"] == 2
    assert 0.0 <= (st["black_win_rate"] or 0) <= 1.0
    assert st["black"]["name"] == "随机A"

    g = sess.game(0)
    assert len(g["moves"]) == len(g["boards"]) - 1
    assert g["result"] in (BLACK, WHITE, 0)

    reg = sess.regions()
    assert reg["samples"] == sum(len(g.moves) for g in sess.results)
    assert len(reg["grid"]) == 8 and len(reg["grid"][0]) == 8
    assert set(reg["regions"]) == {"corner", "edge", "center"}


# ---------------- API 路由 ----------------
def test_duel_api_flow():
    d = client.post("/api/duel/start",
                    json={"black": "random", "white": "random", "games": 2}).json()
    assert d["running"] is True and d["games"] == 2
    st = _wait_duel_done()
    assert st["played"] == 2
    assert st["black_wins"] + st["white_wins"] + st["draws"] == 2

    g = client.get("/api/duel/game/0").json()
    assert "moves" in g and "boards" in g and "players" in g
    assert len(g["moves"]) == len(g["boards"]) - 1

    reg = client.get("/api/duel/regions").json()
    assert len(reg["grid"]) == 8
    assert reg["samples"] > 0


def test_duel_games_range_validation():
    r = client.post("/api/duel/start",
                    json={"black": "random", "white": "random", "games": 0})
    assert r.status_code == 400
    r = client.post("/api/duel/start",
                    json={"black": "random", "white": "random", "games": 101})
    assert r.status_code == 400


def test_duel_unknown_model_404():
    r = client.post("/api/duel/start",
                    json={"black": "no-such-key", "white": "random", "games": 1})
    assert r.status_code == 404


def test_duel_game_out_of_range_404():
    r = client.get("/api/duel/game/999")
    assert r.status_code == 404


# ---------------- 悔棋 & 棋谱 ----------------
def test_undo_returns_previous_position():
    # 纯 AI 对战:悔棋只弹一手(人机模式才连续弹到"轮到人类")
    client.post("/api/player", json={"human": "none"})
    client.post("/api/reset")
    client.post("/api/step", json={"ai": True})
    d2 = client.post("/api/step", json={"ai": True}).json()
    assert d2["state"]["steps"] == 2

    u = client.post("/api/undo").json()
    assert u["state"]["steps"] == 1
    # 悔棋后当前局面 = 走完第 1 步的局面
    d1 = client.post("/api/step", json={"ai": True}).json()  # 重走第 2 步
    assert d1["state"]["steps"] == 2
    client.post("/api/player", json={"human": "black"})      # 还原


def test_undo_empty_returns_400():
    client.post("/api/reset")
    r = client.post("/api/undo")
    assert r.status_code == 400


def test_undo_after_game_over_allowed():
    """终局后允许悔棋:撤销最后一手重新争取翻盘。"""
    client.post("/api/player", json={"human": "none"})
    client.post("/api/reset")
    for _ in range(64):
        d = client.post("/api/step", json={"ai": True}).json()
        if d["state"]["game_over"]:
            break
    assert d["state"]["game_over"] is True
    u = client.post("/api/undo")
    assert u.status_code == 200
    assert u.json()["state"]["game_over"] is False   # 悔回终局前
    client.post("/api/player", json={"human": "black"})      # 还原


def test_history_returns_full_game_record():
    client.post("/api/player", json={"human": "none"})
    client.post("/api/reset")
    client.post("/api/step", json={"ai": True})
    client.post("/api/step", json={"ai": True})
    h = client.get("/api/history").json()
    assert len(h["moves"]) == 2
    assert len(h["boards"]) == 3
    assert h["counts"]["black"] + h["counts"]["white"] == 6
    assert h["result"] is None          # 还没终局
    client.post("/api/player", json={"human": "black"})      # 还原
