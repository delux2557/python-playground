"""黑白棋"统一评测基准 + 循环赛打榜"的测试。

覆盖(P2 演进的新功能):
  1. ArenaSession:循环赛引擎——排赛程、后台对局、积分/胜率排行榜
  2. /api/arena/*:发起、轮询进度、实时排行榜
  3. /api/benchmark:统一评测——全体模型同一基准重打分,生成排行榜

所有测试都用临时目录隔离注册表与评测结果,不污染真实 models/registry.json。
"""

import time

import pytest
from fastapi.testclient import TestClient

from othello.dqn import OthelloDQNConfig, make_agent
from othello.serve import app
from shared.arena import ArenaSession

client = TestClient(app)


def _fresh_registry(monkeypatch, tmp_path):
    """把注册表文件指到临时目录,隔离测试数据。"""
    import shared.registry as registry
    reg_file = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "_REGISTRY_PATH", reg_file)
    return reg_file


def _register_saved_models(monkeypatch, tmp_path, n=2):
    """登记 n 个真实可加载的随机初始模型,返回它们的 key。"""
    _fresh_registry(monkeypatch, tmp_path)
    import shared.registry as registry
    keys = []
    for i in range(n):
        ckpt = tmp_path / "models" / f"m{i}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        make_agent(OthelloDQNConfig()).save(str(ckpt))
        keys.append(registry.register("othello", str(ckpt), eval_score=None))
    return keys


def _wait_arena_done(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get("/api/arena/status").json()
        if not st["running"]:
            return st
        time.sleep(0.1)
    raise AssertionError("循环赛超时未结束")


# ---------------- ArenaSession 单元:循环赛引擎 ----------------
def test_arena_session_round_robin_schedule():
    sess = ArenaSession()
    agents = [make_agent(OthelloDQNConfig()) for _ in range(3)]
    players = [
        {"key": f"p{i}", "name": f"选手{i}", "eval_score": None, "agent": a}
        for i, a in enumerate(agents)
    ]
    sess.start(lambda b, w: 0, players, games_per_match=2)

    # 3 人赛程 = C(3,2) 对 × 每对黑白各 2 局 = 12 场;每人打 8 场
    assert sess.status()["total"] == 12
    deadline = time.time() + 30
    while sess.running and time.time() < deadline:
        time.sleep(0.1)
    assert sess.running is False
    st = sess.status()
    assert st["played"] == 12
    lb = st["leaderboard"]
    assert len(lb) == 3
    # 全平局:每人 8 平 → 8 分,胜率 50%
    for r in lb:
        assert r["wins"] == 0 and r["losses"] == 0
        assert r["draws"] == 8
        assert r["points"] == 4.0
        assert r["win_rate"] == 0.5


def test_arena_session_records_wins_and_losses():
    sess = ArenaSession()
    a, b = make_agent(OthelloDQNConfig()), make_agent(OthelloDQNConfig())
    # 固定比赛结果:黑执甲必赢,黑执乙必输 → 各赢各输一局
    sess.start(lambda blk, wht: 1, [
        {"key": "甲", "name": "甲", "eval_score": None, "agent": a},
        {"key": "乙", "name": "乙", "eval_score": None, "agent": b},
    ], games_per_match=1)
    deadline = time.time() + 30
    while sess.running and time.time() < deadline:
        time.sleep(0.1)
    lb = {r["key"]: r for r in sess.status()["leaderboard"]}
    # 甲:执黑赢 1 局得 1 分,执白输 1 局 → wins=1, losses=1, points=1
    assert lb["甲"]["wins"] == 1 and lb["甲"]["losses"] == 1
    assert lb["甲"]["points"] == 1.0
    assert lb["乙"]["wins"] == 1 and lb["乙"]["losses"] == 1
    # 同名次并列时按 key 排序(稳定性),胜率 50%
    assert lb["甲"]["win_rate"] == 0.5


# ---------------- /api/arena/* 集成 ----------------
def test_arena_api_requires_two_players(monkeypatch, tmp_path):
    _fresh_registry(monkeypatch, tmp_path)       # 空注册表 → 只有随机基准 1 人
    r = client.post("/api/arena/start", json={"games_per_match": 1})
    assert r.status_code == 400


def test_arena_api_full_round_robin(monkeypatch, tmp_path):
    keys = _register_saved_models(monkeypatch, tmp_path, n=2)
    r = client.post("/api/arena/start", json={"games_per_match": 1})
    assert r.status_code == 200
    st = r.json()
    assert st["running"] is True
    # 2 模型 + 1 随机基准 = 3 人 → 3 对 × 黑/白 = 6 场
    assert st["total"] == 6

    final = _wait_arena_done()
    assert final["running"] is False
    lb = final["leaderboard"]
    assert len(lb) == 3
    names = {row["key"] for row in lb}
    assert set(keys) <= names and "random" in names
    # 积分守恒:每场产生 1 分(胜负)或 2×0.5(平),总积分 = 6
    assert sum(row["points"] for row in lb) == 6.0

    # 实时排行榜路由也应一致
    r = client.get("/api/arena/leaderboard").json()
    assert len(r["leaderboard"]) == 3


def test_arena_rejects_second_concurrent(monkeypatch, tmp_path):
    keys = _register_saved_models(monkeypatch, tmp_path, n=2)
    client.post("/api/arena/start", json={"games_per_match": 1})
    try:
        r = client.post("/api/arena/start", json={"games_per_match": 1})
        assert r.status_code == 400
    finally:
        _wait_arena_done()


# ---------------- /api/benchmark 统一评测 ----------------
def test_benchmark_scores_all_models(monkeypatch, tmp_path):
    import shared.eval as eval_mod
    monkeypatch.setattr(eval_mod, "_ROOT", tmp_path)   # 评测结果写临时目录
    _register_saved_models(monkeypatch, tmp_path, n=2)

    r = client.post("/api/benchmark?games=2")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 2                            # 2 个已登记模型
    for row in rows:
        assert 0.0 <= row["score"] <= 1.0            # 对随机胜率,合法区间
        assert "detail" in row
    # 排行榜按分数从高到低
    scores = [row["score"] for row in rows]
    assert scores == sorted(scores, reverse=True)

    # 统一分同步刷新进注册表
    import shared.registry as registry
    data = registry.load_registry()
    for row in rows:
        assert data[row["key"]]["eval_score"] == row["score"]

    # 持久化排行榜 → GET 能读回
    r = client.get("/api/benchmark").json()
    assert len(r["rows"]) == 2 and r["saved_at"] is not None


def test_benchmark_empty_registry(monkeypatch, tmp_path):
    import shared.eval as eval_mod
    monkeypatch.setattr(eval_mod, "_ROOT", tmp_path)
    _fresh_registry(monkeypatch, tmp_path)
    r = client.post("/api/benchmark?games=2").json()
    assert r["rows"] == []
    # 持久化文件为空结构,GET 友好返回
    assert client.get("/api/benchmark").json()["rows"] == []
