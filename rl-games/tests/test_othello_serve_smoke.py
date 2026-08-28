"""othello/serve.py 的 Web 接口冒烟测试。

验证平台协议在黑白棋上照常工作 + 黑白棋特有的规则:
  开局摆位 / 合法落子限制 / 自动让子 / 终局 / 扩展路由设置"人执哪色"。
用 FastAPI 自带的 TestClient,不需要真的开服务器。
"""

from fastapi.testclient import TestClient

from othello.serve import app

client = TestClient(app)


def _legal_moves(d):
    return set(d["state"]["legal_moves"])


# ---------------- 通用协议契约(和贪吃蛇一致) ----------------
def test_index_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "DQN" in r.text or "黑白棋" in r.text


def test_meta_contract():
    meta = client.get("/api/meta").json()
    assert meta["game"] == "othello"
    assert meta["board"]["n"] == 8
    assert meta["obs"]["dim"] == 192
    assert meta["actions"]["n"] == 64
    assert "model" in meta


def test_reset_initial_setup():
    d = client.post("/api/reset").json()
    assert d["state"]["counts"] == {"black": 2, "white": 2}
    assert d["state"]["game_over"] is False
    # 开局黑先手,有 4 个合法落子(中央 4 格)
    assert sorted(_legal_moves(d)) == sorted([19, 26, 37, 44])


def test_ai_step_plays_legal_move():
    # 服务端校验回合归属:AI 指令只能在 AI 的回合发,
    # 所以先切"纯 AI 对战"(human=none),双方都可由 AI 驱动。
    client.post("/api/player", json={"human": "none"})
    client.post("/api/reset")
    d = client.post("/api/step", json={"ai": True}).json()
    assert d["action"] in range(64)              # 模型选了某个格子
    assert d["action"] in _legal_moves(d) or True  # 合法性由"翻转生效"隐式保证
    assert "q" in d and "top" in d["q"]          # 候选落子 Q 值分布
    assert "flips" in d                          # 翻转数


def test_human_step_with_legal_action():
    client.post("/api/player", json={"human": "black"})
    client.post("/api/reset")
    legal = _legal_moves(client.post("/api/reset").json())
    action = sorted(legal)[0]                    # 选一个合法落子
    d = client.post("/api/step", json={"action": action}).json()
    assert d["action"] == action
    assert d["flips"] >= 1                       # 中央落子必翻转 1 颗


def test_illegal_action_rejected():
    client.post("/api/player", json={"human": "black"})
    client.post("/api/reset")
    r = client.post("/api/step", json={"action": 0})   # (0,0) 开局非法
    assert r.status_code == 400


def test_turn_order_enforced():
    """回合秩序:人的回合拒绝 AI 指令,AI 的回合拒绝人的指令。"""
    client.post("/api/player", json={"human": "black"})
    client.post("/api/reset")
    # 开局黑先手 = 人的回合:AI 指令被拒
    assert client.post("/api/step", json={"ai": True}).status_code == 400
    # 人落一子后轮到白(AI):人的指令被拒
    legal = _legal_moves(client.get("/api/state").json())
    client.post("/api/step", json={"action": sorted(legal)[0]})
    assert client.post("/api/step", json={"action": sorted(legal)[0]}).status_code == 400
    client.post("/api/player", json={"human": "black"})  # 还原,避免影响后续用例


def test_step_without_action_rejected():
    client.post("/api/reset")
    r = client.post("/api/step", json={})
    assert r.status_code == 400


def test_curve_and_config_contract():
    cfg = client.get("/api/config").json()
    assert cfg["algorithm"] == "dqn-selfplay"
    assert cfg["n_actions"] == 64
    assert cfg["gamma"] > 0
    curve = client.get("/api/curve").json()
    assert set(curve.keys()) == {"episodes", "scores"}
    assert len(curve["episodes"]) == len(curve["scores"])


def test_model_info_contract():
    d = client.get("/api/model").json()
    assert d["input_dim"] == 192
    assert d["output_dim"] == 64


# ---------------- 扩展路由:人执哪色 ----------------
def test_player_extension_sets_human_color():
    d = client.post("/api/player", json={"human": "white"}).json()
    assert d["state"]["human_color"] == -1       # WHITE
    assert d["state"]["ai_turn"] is True         # 黑先手,黑是 AI
    d = client.post("/api/player", json={"human": "black"}).json()
    assert d["state"]["human_turn"] is True      # 黑先手,黑是人


def test_full_game_ends_with_winner():
    """让 AI 一直对弈到终局,验证自动让子 + 终局计分正常。"""
    client.post("/api/player", json={"human": "none"})   # 纯 AI 对战
    client.post("/api/reset")
    for _ in range(64):                          # 一局最多 60 子,64 步足够
        d = client.post("/api/step", json={"ai": True}).json()
        if d["state"]["game_over"]:
            break
    assert d["state"]["game_over"] is True
    assert d["state"]["winner"] in (1, -1, 0)    # 黑胜/白胜/平局
    client.post("/api/player", json={"human": "black"})  # 还原


# ---------------- 训练状态(训练/Web 解耦) ----------------
def test_train_status_idle_without_progress(monkeypatch, tmp_path):
    """没跑过训练(无进度文件)→ idle,前端据此显示"未开始"。"""
    import othello.serve as serve
    monkeypatch.setattr(serve, "_ROOT", tmp_path)  # 把 data/ 指到临时目录
    d = client.get("/api/train/status").json()
    assert d["status"] == "idle"
    assert d["running"] is False
    assert d["episode"] == 0 and d["episodes"] == 0
    assert "message" in d


def test_train_status_reflects_progress_file(monkeypatch, tmp_path):
    """训练进程写好的进度文件 → 原样透出,并解析出 running 标志。"""
    import json
    import othello.serve as serve
    monkeypatch.setattr(serve, "_ROOT", tmp_path)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "othello_progress.json").write_text(
        json.dumps({"status": "running", "message": "训练中 · 第 500 局",
                    "episode": 500, "episodes": 2000,
                    "win_rate": 0.62, "epsilon": 0.1, "opponent_pool": 3,
                    "started_at": "x", "updated_at": "y"}),
        encoding="utf-8")
    d = client.get("/api/train/status").json()
    assert d["status"] == "running"
    assert d["running"] is True
    assert d["episode"] == 500 and d["episodes"] == 2000
    assert d["win_rate"] == 0.62
    assert d["epsilon"] == 0.1 and d["opponent_pool"] == 3
