"""snake/train.py —— 贪吃蛇 DQN 训练脚本(同时是被测试复用的核心逻辑)。

用法:
    python snake/train.py --episodes 3000 --checkpoint models/snake.pt --curve data/curve.json

训练流程(对应 DESIGN.md 第 4.3 节):
    每局: 环境开局 → 模型按 ε-greedy 选动作 → 存经验 → 从回放池抽样更新网络
    定期: 用"纯贪心"(ε=0)跑评估局,记录平均得分,画收敛曲线
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# 项目根目录(用于把模型路径存成相对路径,注册表里统一格式)
# 并插入 sys.path:直接运行 "python snake/train.py" 时,Python 只把
# 脚本所在目录(snake/)放进 sys.path[0],项目根不在——没有这一句,
# 下面的 `from snake.dqn import ...` 会报 ModuleNotFoundError
# (与 othello/train.py、serve.py 的处理保持一致)。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.checkpoint import load_for_resume, save_with_meta
from shared.experiment import track_training  # noqa: E402
from shared.registry import register  # noqa: E402
from snake.dqn import SnakeDQNConfig, make_agent  # noqa: E402
from snake.env import SnakeEnv  # noqa: E402


def set_seed(seed):
    """固定所有随机源,保证训练可复现(最佳实践)。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent, cfg, episodes=20, seed=0):
    """纯贪心评估:关闭探索,让模型"认真发挥",返回 (得分列表, 平均分, 平均步数)。

    评估必须和训练分开做(ε=0),否则曲线会高估模型真实水平。
    """
    scores, steps = [], []
    for i in range(episodes):
        env = SnakeEnv(grid_size=cfg.grid_size, seed=seed + i)
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, _, done, _ = env.step(action)
        scores.append(env.score)
        steps.append(env.steps)
    return scores, float(np.mean(scores)), float(np.mean(steps))


def run_training(cfg, checkpoint_path=None, curve_path=None, verbose=True):
    """核心训练函数。

    参数:
      cfg            : SnakeDQNConfig(超参数都在这)
      checkpoint_path: 保存模型的路径(结尾 .pt)
      curve_path     : 保存收敛曲线的 JSON 路径
    返回:
      {"final_avg_score": 最终平均分, "eval_indices": 评估局序号,
       "eval_scores": 各次评估平均分}
    """
    set_seed(cfg.seed)
    agent = make_agent(cfg)
    env = SnakeEnv(grid_size=cfg.grid_size, seed=cfg.seed)

    eval_indices, eval_scores = [], []
    start_ep = 0

    # ---- 断点续训:检测已有 checkpoint,接着上次的局数往下训 ----
    if checkpoint_path:
        resume_agent, start_ep, resume_eps = load_for_resume(checkpoint_path)
        if resume_agent is not None:
            agent = resume_agent
            if resume_eps is not None:
                agent.epsilon = resume_eps   # 恢复探索率,从上次位置继续衰减
            if verbose:
                print(f"[续训] 检测到 checkpoint,从第 {start_ep} 局续训到 "
                      f"{cfg.episodes} 局(ε={agent.epsilon:.3f})")
            # 保留历史曲线:旧评估点还在,新评估点往后接,Web 曲线连续
            if curve_path and Path(curve_path).exists():
                try:
                    old = json.loads(Path(curve_path).read_text(encoding="utf-8"))
                    eval_indices = old.get("episodes", [])
                    eval_scores = old.get("scores", [])
                except (json.JSONDecodeError, OSError):
                    pass

    for ep in range(start_ep + 1, cfg.episodes + 1):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, greedy=False)  # 训练:带探索
            s_next, reward, done, _ = env.step(action)
            agent.store(obs, action, reward, s_next, done)   # 存经验
            obs = s_next
            agent.update()                                   # 抽 batch 更新网络

        agent.decay_epsilon()  # 每局结束,探索率降一点

        # 定期评估一次,记录收敛曲线;并定期存档(崩溃最多丢 eval_freq 局)
        if ep % cfg.eval_freq == 0:
            _, avg_score, avg_steps = evaluate(agent, cfg,
                                               episodes=cfg.eval_episodes,
                                               seed=cfg.seed + ep)
            eval_indices.append(ep)
            eval_scores.append(avg_score)
            if checkpoint_path:
                save_with_meta(agent, checkpoint_path,
                               {"episode": ep, "epsilon": agent.epsilon})
            if verbose:
                print(f"[局 {ep:>5}/{cfg.episodes}] ε={agent.epsilon:.3f} "
                      f"评估平均分={avg_score:.2f} 平均步数={avg_steps:.0f}")

    # 最终评估(和训练随机种子区分开,测的是模型真实水平)
    _, final_score, _ = evaluate(agent, cfg, episodes=cfg.eval_episodes,
                                 seed=cfg.seed + 999_999)

    # 保存模型和曲线
    run_id = None
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        save_with_meta(agent, checkpoint_path,
                       {"episode": cfg.episodes, "epsilon": agent.epsilon})
        # 实验跟踪:把本次训练的超参/得分曲线/模型档案记进 MLflow
        run_id = track_training(
            game="snake",
            experiment="snake-dqn",
            params=dict(cfg.__dict__),
            curves={"avg_score": (eval_indices, eval_scores)},
            checkpoint_path=checkpoint_path,
            final_metrics={"final_avg_score": round(final_score, 3)},
        )
        # 登记到模型注册表:前端 /api/models 就能列出并切换这个模型。
        # 这里保存的是相对项目根的路径,注册表里格式统一。
        model_rel = Path(checkpoint_path)
        if model_rel.is_absolute():
            try:
                model_rel = model_rel.relative_to(_ROOT)
            except ValueError:
                pass
        register(
            game="snake",
            path=str(model_rel),
            algorithm="dqn",
            hidden_dims=list(cfg.hidden_dims),
            eval_score=round(final_score, 2),
            episodes=cfg.episodes,
            run_id=run_id,
        )
    if curve_path:
        os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)
        # 原子写:Web 端可能正在轮询 /api/curve 读这个文件,直接覆盖
        # 会读到半截 JSON。先写临时文件再 os.replace(同目录原子替换)。
        tmp = curve_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"episodes": eval_indices, "scores": eval_scores}, f)
        os.replace(tmp, curve_path)

    if verbose:
        print(f"完成!最终平均分 = {final_score:.2f} → 模型已保存到 {checkpoint_path}")

    return {"final_avg_score": final_score,
            "eval_indices": eval_indices,
            "eval_scores": eval_scores}


def main():
    parser = argparse.ArgumentParser(description="贪吃蛇 DQN 训练")
    parser.add_argument("--episodes", type=int, default=SnakeDQNConfig.episodes,
                        help="总训练局数")
    parser.add_argument("--grid-size", type=int, default=SnakeDQNConfig.grid_size)
    parser.add_argument("--checkpoint", default="models/snake.pt", help="模型保存路径")
    parser.add_argument("--curve", default="data/curve.json", help="曲线数据保存路径")
    parser.add_argument("--seed", type=int, default=SnakeDQNConfig.seed)
    parser.add_argument("--eval-freq", type=int, default=SnakeDQNConfig.eval_freq)
    parser.add_argument("--eval-episodes", type=int, default=SnakeDQNConfig.eval_episodes)
    args = parser.parse_args()

    cfg = SnakeDQNConfig(episodes=args.episodes, grid_size=args.grid_size,
                         seed=args.seed, eval_freq=args.eval_freq,
                         eval_episodes=args.eval_episodes)
    run_training(cfg, checkpoint_path=args.checkpoint, curve_path=args.curve)


if __name__ == "__main__":
    main()
