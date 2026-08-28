"""snake/serve.py 的 Web 接口冒烟测试。

验证驾驶舱后端每个接口都按"契约"返回数据:
状态/开局/模型决策/人类输入/曲线/参数/模型信息。
用 FastAPI 自带的 TestClient,不需要真的开服务器。
"""

from fastapi.testclient import TestClient

from snake.serve import app

# TestClient 会在内存里跑一个"假的服务器",直接调用我们的路由
client = TestClient(app)


def test_index_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "DQN" in r.text          # 返回的是我们的驾驶舱页面


def test_state_contract():
    d = client.get("/api/state").json()
    assert len(d["obs"]) == 11           # 观察向量 11 维
    assert len(d["q_values"]) == 4       # 4 个方向各一个 Q 值
    assert d["state"]["grid_size"] == 11
    assert d["state"]["score"] == 0      # 开局 0 分


def test_reset_starts_clean():
    d = client.post("/api/reset").json()
    assert d["state"]["score"] == 0
    assert len(d["state"]["snake"]) == 3  # 蛇初始长度 3


def test_ai_step_returns_decision():
    client.post("/api/reset")
    d = client.post("/api/step", json={"ai": True}).json()
    assert d["action"] in (0, 1, 2, 3)   # 模型选了一个合法方向
    assert len(d["pre_obs"]) == 11       # 决策前看到的输入
    assert len(d["q_values"]) == 4
    assert "reward" in d                 # 这一步拿到的奖励
    assert "reason" in d                 # 结果原因(move/ate/...)


def test_human_step_uses_given_action():
    client.post("/api/reset")
    d = client.post("/api/step", json={"action": 3}).json()
    assert d["action"] == 3              # 人类指定向右,就向右


def test_step_without_action_rejected():
    client.post("/api/reset")
    r = client.post("/api/step", json={})
    assert r.status_code == 400          # 必须给 ai 或 action


def test_curve_and_config_contract():
    cfg = client.get("/api/config").json()
    assert cfg["gamma"] > 0 and cfg["lr"] > 0
    assert cfg["n_actions"] == 4
    curve = client.get("/api/curve").json()
    assert set(curve.keys()) == {"episodes", "scores"}
    assert len(curve["episodes"]) == len(curve["scores"])


def test_model_info_contract():
    d = client.get("/api/model").json()
    assert d["input_dim"] == 11
    assert d["output_dim"] == 4
    assert "loaded" in d
