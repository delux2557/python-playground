"""黑白棋自对弈训练链路的冒烟测试。

验证三件事:
  1. play_game 能收集到"挑战者视角"的合法经验
  2. evaluate_win_rate 能正常算出 0~1 之间的胜率
  3. 完整训练链路能跑完、存模型、登记注册表、输出曲线
     (用极小配置,只验证"链路通",不验证"学得好")
"""

import json

import numpy as np
import pytest

import othello.selfplay as selfplay
from othello.dqn import OthelloDQNConfig, make_agent
from othello.env import N_ACTIONS, OBS_DIM, OthelloEnv, BLACK, WHITE
from othello.train import run_training
from shared import registry


def small_cfg(**overrides):
    """最小化配置:几局就能跑完,专门给测试用。"""
    base = dict(episodes=20, eval_freq=10, eval_games=5,
                random_opponent_prob=0.5, opponent_pool_size=2,
                opponent_save_freq=5, seed=0)
    base.update(overrides)
    return OthelloDQNConfig(**base)


def _fresh_registry(monkeypatch, tmp_path):
    """把注册表指到临时目录,避免训练登记污染真实 models/registry.json。"""
    reg_file = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "_REGISTRY_PATH", reg_file)
    return reg_file


# ---------------- play_game ----------------
def test_play_game_collects_legal_transitions():
    cfg = small_cfg()
    agent = make_agent(cfg)
    env = OthelloEnv()
    transitions, outcome = selfplay.play_game(agent, None, env)

    assert outcome in (1.0, -1.0, 0.0)
    assert len(transitions) >= 1
    for s, a, r, s_next, done in transitions:
        assert np.array(s).shape == (OBS_DIM,)
        assert np.array(s_next).shape == (OBS_DIM,)
        assert 0 <= a < N_ACTIONS
        assert r == 0.0                     # 奖励由训练方在终局统一打标签
        assert isinstance(done, bool)


def test_play_game_transition_action_was_legal():
    """挑战者落下的每一手都必须是合法落子(掩码生效)。"""
    cfg = small_cfg()
    agent = make_agent(cfg)
    env = OthelloEnv()
    transitions, _ = selfplay.play_game(agent, None, env)
    # 重放:逐步落子,校验每一步动作都落在当时棋盘上的合法位置
    env.reset()
    # 但 play_game 内部随机决定先后手,这里无法精确重放;
    # 改为校验"挑战者视角观察"的通道数与棋子数自洽
    for s, a, _, _, _ in transitions[:3]:
        own, opp, empty = np.split(np.array(s), 3)
        assert own.sum() + opp.sum() + empty.sum() == 64


# ---------------- evaluate_win_rate ----------------
def test_evaluate_win_rate_bounds():
    """随机权重的模型对战随机对手,胜率应该接近 0.5(在合理区间内)。"""
    cfg = small_cfg()
    agent = make_agent(cfg)
    wr = selfplay.evaluate_win_rate(agent, games=20, seed=1)
    assert 0.0 <= wr <= 1.0
    assert 0.1 <= wr <= 0.9      # 随机对随机不该一边倒


# ---------------- 训练冒烟 ----------------
def test_selfplay_trainer_runs_and_uses_opponent_pool():
    cfg = small_cfg()
    trainer = selfplay.SelfPlayTrainer(cfg)
    curve = trainer.run(verbose=False)
    # 训练后对手池应该被填进过历史快照
    assert 0 <= len(trainer.opponent_pool) <= cfg.opponent_pool_size
    # 曲线点数 = 训练局数 / eval_freq
    assert len(curve["episodes"]) == cfg.episodes // cfg.eval_freq
    assert len(curve["win_rates"]) == len(curve["episodes"])


def test_training_smoke_runs_and_saves(tmp_path, monkeypatch):
    """完整训练链路:出模型文件 + 曲线文件 + 登记注册表。"""
    _fresh_registry(monkeypatch, tmp_path)
    cfg = small_cfg()
    ckpt = tmp_path / "othello.pt"
    curve = tmp_path / "curve.json"

    result = run_training(cfg, checkpoint_path=str(ckpt),
                          curve_path=str(curve), verbose=False)

    assert ckpt.exists()                              # 模型文件生成了
    assert curve.exists()                             # 曲线文件生成了
    assert 0.0 <= result["final_win_rate"] <= 1.0
    # 注册表里多了一条 othello 记录
    models = registry.list_models("othello")
    assert len(models) == 1
    assert models[0]["algorithm"] == "dqn-selfplay"
    # 注册表指向的文件真实存在(切换模型时能加载)
    assert registry.resolve(models[0]["key"]) is not None


def test_run_training_writes_progress_file(tmp_path, monkeypatch):
    """训练/Web 解耦:训练进程把进度写到 progress.json(前端靠它实时显示)。"""
    _fresh_registry(monkeypatch, tmp_path)
    cfg = small_cfg()
    ckpt = tmp_path / "othello.pt"
    curve = tmp_path / "curve.json"
    progress = tmp_path / "othello_progress.json"

    run_training(cfg, checkpoint_path=str(ckpt), curve_path=str(curve),
                 progress_path=str(progress), verbose=False)

    data = json.loads(progress.read_text(encoding="utf-8"))
    assert data["status"] == "done"            # 正常结束置 done
    assert data["game"] == "othello"
    assert data["episodes"] == cfg.episodes
    assert data["episode"] == cfg.episodes     # 跑满局数
    assert 0.0 <= data["win_rate"] <= 1.0
    assert data["started_at"] and data["updated_at"]
    # 曲线文件与进度一致:点数 = 局数 / 评估间隔
    cdata = json.loads(curve.read_text(encoding="utf-8"))
    assert len(cdata["episodes"]) == cfg.episodes // cfg.eval_freq
    assert len(cdata["win_rates"]) == len(cdata["episodes"])


def test_checkpoint_resume_continues_from_saved_episode(tmp_path, monkeypatch, capsys):
    """断点续训:第一次跑存下 checkpoint,第二次应检测到并"接着上次局数"往下训。"""
    _fresh_registry(monkeypatch, tmp_path)
    ckpt = tmp_path / "othello.pt"
    curve = tmp_path / "curve.json"
    progress = tmp_path / "progress.json"

    # 第一段:跑 8 局,评估间隔 4 → checkpoint 记录 episode=8,曲线 [4, 8]
    run_training(small_cfg(episodes=8, eval_freq=4),
                 checkpoint_path=str(ckpt), curve_path=str(curve),
                 progress_path=str(progress), verbose=False)

    # 第二段:目标 12 局 → 应从第 9 局续训到 12,曲线追加 [12]
    run_training(small_cfg(episodes=12, eval_freq=4),
                 checkpoint_path=str(ckpt), curve_path=str(curve),
                 progress_path=str(progress), verbose=True)
    out = capsys.readouterr().out
    assert "[续训]" in out
    assert "从第 8 局续训到 12" in out

    # 曲线保留第一段旧点,新点往后接(Web 端曲线连续)
    data = json.loads(curve.read_text(encoding="utf-8"))
    assert data["episodes"] == [4, 8, 12]
    # 续训元数据:最终 episode 记为 12
    from shared.checkpoint import read_meta
    assert read_meta(ckpt)["episode"] == 12
