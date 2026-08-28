"""统一 Agent 服务协议(shared/protocol.py)的接口测试。

验证平台化的三项核心能力,对所有 AgentService 实现都成立:
  1. /api/meta  自报家门:棋盘 / 动作 / 观察含义 + 已登记模型
  2. /api/models 列出注册表里的模型
  3. /api/models/load 切换模型(平台化的"换模型"落地)
"""

from fastapi.testclient import TestClient

from snake.serve import app

client = TestClient(app)


def test_meta_self_reports_game():
    """协议层要求:每个游戏都要"自报家门",前端全靠它渲染。"""
    d = client.get("/api/meta").json()
    assert d["game"] == "snake"
    assert d["actions"]["type"] == "discrete"
    assert d["actions"]["n"] == 4
    assert len(d["actions"]["names"]) == 4
    assert d["obs"]["dim"] == 11
    assert len(d["obs"]["meaning"]) == 11
    assert "models" in d  # 协议层自动把已登记模型附在元数据里


def test_models_list_contract():
    d = client.get("/api/models").json()
    assert isinstance(d, list)
    for m in d:
        assert set(m) >= {"key", "game", "path", "algorithm", "created_at"}


def test_model_load_switches_model():
    """通过注册表 key 切换模型,应返回最新快照供前端刷新。"""
    d = client.get("/api/models").json()
    assert d, "需要已登记的模型(先跑训练或播种注册表)才能测切换"
    key = d[0]["key"]

    r = client.post("/api/models/load", json={"name": key})
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    assert "model" in body
    assert "state" in body    # 切换后带出当前局面
    assert len(body["q_values"]) == 4


def test_model_load_unknown_returns_404():
    r = client.post("/api/models/load", json={"name": "no-such-key"})
    assert r.status_code == 404


def test_model_info_contract():
    d = client.get("/api/model").json()
    assert "input_dim" in d and "output_dim" in d and "loaded" in d


def test_train_status_contract():
    """协议层要求:每个游戏都提供 /api/train/status。

    默认实现(未接入进度上报)返回 idle;游戏可覆盖读自己的进度文件。
    前端训练面板靠它轮询显示(训练/Web 解耦)。
    """
    d = client.get("/api/train/status").json()
    assert d["status"] == "idle"
    assert d["running"] is False
    assert "episodes" in d and "win_rate" in d and "message" in d
