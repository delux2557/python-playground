"""shared/registry.py 模型注册表的单元测试。

不碰真实的 models/registry.json——用 pytest 的临时文件,测完自动清理。
"""

import shared.registry as registry


def _fresh_registry(monkeypatch, tmp_path):
    """把注册表文件位置指到临时目录,隔离测试数据。"""
    reg_file = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "_REGISTRY_PATH", reg_file)
    return reg_file


def test_register_creates_entry(monkeypatch, tmp_path):
    _fresh_registry(monkeypatch, tmp_path)
    key = registry.register("snake", "models/snake.pt", algorithm="dqn",
                            hidden_dims=[64, 64], eval_score=12.5, episodes=100)
    assert key.startswith("snake-")
    data = registry.load_registry()
    assert data[key]["path"] == "models/snake.pt"
    assert data[key]["eval_score"] == 12.5
    assert data[key]["episodes"] == 100


def test_list_filters_by_game_and_sorts_newest_first(monkeypatch, tmp_path):
    _fresh_registry(monkeypatch, tmp_path)
    registry.register("snake", "models/snake.pt", eval_score=10)
    registry.register("snake", "models/snake_v2.pt", eval_score=20)
    registry.register("othello", "models/othello.pt", eval_score=5)

    snakes = registry.list_models("snake")
    assert len(snakes) == 2
    assert all(m["game"] == "snake" for m in snakes)
    assert snakes[0]["path"] == "models/snake_v2.pt"  # 最新排最前

    assert len(registry.list_models()) == 3           # 不筛游戏 = 全部


def test_resolve_by_key_and_path(monkeypatch, tmp_path):
    _fresh_registry(monkeypatch, tmp_path)
    # 造一个真实存在的模型文件:注册表契约是"路径对应的文件得在,才能切换"
    model_file = tmp_path / "models" / "snake.pt"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"fake-ckpt")

    # 按 key 解析 → 返回真实存在文件的路径
    key = registry.register("snake", str(model_file))
    assert registry.resolve(key) == str(model_file)
    # 未知 key → None
    assert registry.resolve("no-such-key") is None
    # 已登记但文件被删 → None(不能加载不存在的文件)
    key2 = registry.register("othello", str(tmp_path / "ghost.pt"))
    assert registry.resolve(key2) is None
    # 直接传路径但文件不存在 → None
    assert registry.resolve("models/ghost.pt") is None
