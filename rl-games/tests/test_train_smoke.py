"""训练链路冒烟测试 + 学习效果验证。

冒烟测试:用很小的局数跑一遍完整训练(环境 → DQN → 存模型 → 存曲线),
确认整条链路不报错、产物齐全。
学习效果测试:用小棋盘、固定随机种子训练一小段时间,
验证模型确实"学会了"(平均得分明显高于随机乱走),这是 RL 项目最重要的验证。
"""

import json

import numpy as np
import pytest

import shared.registry as registry
from snake.dqn import SnakeDQNConfig
from snake.env import SnakeEnv
from snake.train import evaluate, run_training


def _fresh_registry(monkeypatch, tmp_path):
    """把注册表指到临时目录:训练脚本存模型时会自动登记,
    不能让它写进真实 models/registry.json 污染用户数据。"""
    reg_file = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "_REGISTRY_PATH", reg_file)
    return reg_file


def small_cfg(**overrides):
    """一个"又小又快"的训练配置,专供测试用。"""
    defaults = dict(episodes=30, eval_freq=10, eval_episodes=5, grid_size=7,
                    seed=0, buffer_capacity=2000, batch_size=32,
                    hidden_dims=(32, 32), target_update_freq=50, lr=3e-4)
    defaults.update(overrides)
    return SnakeDQNConfig(**defaults)


def random_baseline(cfg, episodes=50, seed=0):
    """随机策略的平均得分:作为"模型有没有学会"的对照基线。"""
    scores = []
    for i in range(episodes):
        env = SnakeEnv(grid_size=cfg.grid_size, seed=seed + i)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, done, _ = env.step(int(np.random.randint(4)))
        scores.append(env.score)
    return float(np.mean(scores))


def test_training_smoke_runs_and_saves(tmp_path, monkeypatch):
    _fresh_registry(monkeypatch, tmp_path)
    cfg = small_cfg()
    ckpt = tmp_path / "snake.pt"
    curve = tmp_path / "curve.json"
    result = run_training(cfg, checkpoint_path=str(ckpt), curve_path=str(curve),
                          verbose=False)

    assert ckpt.exists()                      # 模型保存成功
    data = json.loads(curve.read_text("utf-8"))  # 曲线数据保存成功
    assert len(data["episodes"]) == cfg.episodes // cfg.eval_freq
    assert len(data["scores"]) == len(data["episodes"])
    assert np.isfinite(result["final_avg_score"])


def test_trained_model_beats_random_policy():
    """核心学习验证:训练过的模型应显著强于随机乱走。"""
    cfg = small_cfg(episodes=250)
    result = run_training(cfg, verbose=False)
    baseline = random_baseline(cfg)

    assert result["final_avg_score"] > baseline + 1.0, (
        f"模型平均分 {result['final_avg_score']:.2f} "
        f"应明显高于随机基线 {baseline:.2f}"
    )
    # 曲线整体应呈上升趋势(后期评估分 > 前期)
    assert result["eval_scores"][-1] > result["eval_scores"][0]


def test_evaluate_returns_finite_stats():
    from snake.dqn import make_agent
    cfg = small_cfg()
    agent = make_agent(cfg)
    scores, avg, avg_steps = evaluate(agent, cfg, episodes=5)
    assert len(scores) == 5
    assert np.isfinite(avg) and np.isfinite(avg_steps)
    assert avg >= 0.0


def test_checkpoint_resume_continues_from_saved_episode(tmp_path, monkeypatch, capsys):
    """断点续训:第一次跑存下 checkpoint,第二次应检测到并"接着上次局数"往下训。"""
    _fresh_registry(monkeypatch, tmp_path)
    ckpt = tmp_path / "snake.pt"
    curve = tmp_path / "curve.json"

    # 第一段:跑 8 局,评估间隔 4 → checkpoint 记录 episode=8,曲线 [4, 8]
    run_training(small_cfg(episodes=8, eval_freq=4),
                 checkpoint_path=str(ckpt), curve_path=str(curve), verbose=False)

    # 第二段:目标 12 局 → 应从第 9 局续训到 12,曲线追加 [12]
    run_training(small_cfg(episodes=12, eval_freq=4),
                 checkpoint_path=str(ckpt), curve_path=str(curve), verbose=True)
    out = capsys.readouterr().out
    assert "[续训]" in out
    assert "从第 8 局续训到 12" in out

    # 曲线保留第一段旧点,新点往后接(Web 端曲线连续)
    data = json.loads(curve.read_text("utf-8"))
    assert data["episodes"] == [4, 8, 12]
    # 续训元数据:最终 episode 记为 12
    from shared.checkpoint import read_meta
    assert read_meta(ckpt)["episode"] == 12
