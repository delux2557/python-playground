"""snake/dqn.py —— 贪吃蛇专用的 DQN 配置。

真正的 DQN 组件在 shared/dqn.py(通用、可复用)。
这里只定义贪吃蛇特有的常量,并提供一个"开箱即用"的建智能体函数,
训练脚本和 Web 服务都用它,保证参数只写一份。
"""

from dataclasses import dataclass, field

from shared.dqn import DQNAgent
from snake.env import STATE_DIM


@dataclass
class SnakeDQNConfig:
    """贪吃蛇 DQN 的所有超参数(含义详见 shared/dqn.py 和 DESIGN.md 4.4 节)。"""

    # 游戏
    grid_size: int = 11

    # 网络
    hidden_dims: tuple = (64, 64)

    # 学习
    lr: float = 1e-4          # 学习率:太大发散,太小学得慢
    gamma: float = 0.95       # 折扣因子:越接近 1 越有远见
    batch_size: int = 64      # 每次更新的样本数
    buffer_capacity: int = 50_000  # 经验池容量
    target_update_freq: int = 1000 # 目标网络同步间隔(步数)

    # 探索(ε-greedy)
    epsilon_start: float = 1.0    # 初始探索率:前期多乱走
    epsilon_end: float = 0.01     # 最终探索率:后期几乎纯贪心
    epsilon_decay: float = 0.995  # 每局探索率衰减倍率:0.995 衰减比较温和

    # 训练
    episodes: int = 3000      # 总训练局数
    eval_freq: int = 100      # 每隔多少局做一次"纯贪心评估"
    eval_episodes: int = 20   # 每次评估跑几局
    seed: int = 42            # 随机种子,固定后可复现


def make_agent(cfg: SnakeDQNConfig, seed: int = 0) -> DQNAgent:
    """按配置创建一个贪吃蛇 DQN 智能体(4 个动作)。"""
    return DQNAgent(
        input_dim=STATE_DIM,
        n_actions=4,
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
