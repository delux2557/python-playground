"""othello/dqn.py —— 黑白棋专用的 DQN 配置。

真正的 DQN 组件在 shared/dqn.py(通用、可复用),和贪吃蛇完全一样。
这里只定义黑白棋特有的超参数,并提供一个"开箱即用"的建智能体函数。

关键差异(对比 snake/dqn.py):
  · 输入维度 192(3 通道 × 64 格),而不是贪吃蛇的 11
  · 动作维度 64(每个格子),而不是 4 个方向
  · gamma 取 0.99:黑白棋只在终局给奖励,中间几十步都没有信号,
    模型必须"看得更远"才能把终局的输赢回传到前面的每一步
  · 网络更宽(128, 128):要同时评估 64 个候选落子的好坏
"""

from dataclasses import dataclass

from shared.dqn import DQNAgent
from othello.env import N_ACTIONS, OBS_DIM


@dataclass
class OthelloDQNConfig:
    """黑白棋自对弈 DQN 的超参数(含义详见 shared/dqn.py 与 DESIGN.md)。"""

    # 网络
    hidden_dims: tuple = (128, 128)     # 比贪吃蛇宽:要评估 64 个落子

    # 学习
    lr: float = 1e-4                    # 学习率:太大发散,太小学得慢
    gamma: float = 0.99                 # 折扣因子:终局才给奖励 → 取大,有远见
    batch_size: int = 64                # 每次更新的样本数
    buffer_capacity: int = 100_000      # 经验池容量(棋类经验更多,池子大一点)
    target_update_freq: int = 1000      # 目标网络同步间隔(步数)

    # 探索(ε-greedy)
    epsilon_start: float = 1.0          # 初始探索率
    epsilon_end: float = 0.05           # 最终探索率(棋类比贪吃蛇略高,保留变化)
    epsilon_decay: float = 0.998        # 每局衰减倍率

    # 自对弈(详见 selfplay.py)
    episodes: int = 2000                # 总训练局数
    games_per_update: int = 1           # 每打几局做一次梯度更新
    random_opponent_prob: float = 0.2   # 打"纯随机"对手的概率(保留基线)
    opponent_pool_size: int = 5         # 对手池最多存几个历史模型
    opponent_save_freq: int = 100       # 每隔几局把当前模型存档进对手池

    # 评估
    eval_freq: int = 100                # 每隔几局评估一次
    eval_games: int = 30                # 每次评估打几局(打纯随机,量胜率)
    seed: int = 42                      # 随机种子,固定后可复现


def make_agent(cfg: OthelloDQNConfig, seed: int = 0) -> DQNAgent:
    """按配置创建一个黑白棋 DQN 智能体(64 个动作)。"""
    return DQNAgent(
        input_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dims=cfg.hidden_dims,
        lr=cfg.lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        buffer_capacity=cfg.buffer_capacity,
        batch_size=cfg.batch_size,
        target_update_freq=cfg.target_update_freq,
    )
