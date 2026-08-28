"""shared/dqn.py —— 整个项目共用的 DQN 组件(贪吃蛇和黑白棋都用它)。

本模块只负责"深度 Q 学习"的通用部分,与具体游戏无关:

  1. QNetwork    : 一个多层感知机(MLP),输入"状态",输出"每个动作的 Q 值"
  2. ReplayBuffer: 经验回放池,存放 (s, a, r, s', done),训练时随机抽样
  3. DQNAgent    : 把网络、回放池、ε-greedy 探索、目标网络打包成一个"智能体"

游戏本身的规则(撞墙怎么判、奖励怎么给)不在这里,请去看各游戏的 env.py。
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------
# 1. Q 网络:输入"状态" → 输出"每个动作的 Q 值"
# ---------------------------------------------------------------
class QNetwork(nn.Module):
    """一个简单的多层感知机(MLP)。

    关键直觉:网络输出 4 个数(贪吃蛇)或 64 个数(黑白棋),
    第 i 个数就代表"在这个状态下,选动作 i 的预期长期回报"。
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        参数说明(每个参数影响什么):
          input_dim  : 状态向量的长度(贪吃蛇 11,黑白棋 8×8×3=192)
          hidden_dims: 中间隐藏层神经元数列表,如 [64, 64]。
                       神经元越多表达力越强,但训练更慢、更易在小游戏上过拟合。
          output_dim : 动作个数(贪吃蛇 4,黑白棋 64)
        """
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())  # 激活函数:给网络非线性表达能力
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 最后一层不加激活函数:Q 值本身可以是任意实数(可正可负)
        return self.net(x)


# ---------------------------------------------------------------
# 2. 经验回放池:存样本,训练时随机抽样,打破样本间相关性
# ---------------------------------------------------------------
class ReplayBuffer:
    """经验回放池(DQN 三大技巧之一)。

    为什么需要它:相邻几步的状态高度相关,直接连续喂给网络会让梯度
    剧烈抖动。随机抽样能"打散"相关性,让训练更稳。
    """

    def __init__(self, capacity):
        """
        capacity: 池子最多存多少条经验。
          越大样本越多样、训练越稳;但会混入过旧经验、也更占内存。
          贪吃蛇常用 50_000 左右。
        """
        self.buffer = deque(maxlen=capacity)  # 满了自动丢最旧的

    def push(self, s, a, r, s_next, done):
        """存一条经验:(当前状态, 动作, 奖励, 下一状态, 是否结束)。"""
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        """随机抽 batch_size 条,打包成 torch 张量,方便一次性喂给网络。"""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(np.array(a), dtype=torch.int64),
            torch.as_tensor(np.array(r), dtype=torch.float32),
            torch.as_tensor(np.array(s_next), dtype=torch.float32),
            torch.as_tensor(np.array(done), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------
# 3. DQN 智能体:把网络、回放池、目标网络、ε-greedy 打包在一起
# ---------------------------------------------------------------
class DQNAgent:
    """一个"会玩"的智能体。

    在线网络(online)   : 每步都更新,负责做决定;
    目标网络(target)   : 每隔 target_update_freq 步才同步一次,
                         用"旧一点的自己"来算目标 Q 值,防止训练发散
                         (DQN 三大技巧之二);
    ε-greedy            : 以概率 ε 随机走、否则按 Q 值贪心(DQN 三大技巧之三)。
    """

    def __init__(self, input_dim, n_actions, hidden_dims=(64, 64),
                 lr=1e-4, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_capacity=50_000, batch_size=64,
                 target_update_freq=1000, device=None):
        """
        每个参数影响什么(对应 DESIGN.md 第 4.4 节):
          lr                : 学习率。太大易震荡发散,太小学得慢。
          gamma             : 折扣因子。越接近 1 越有远见,但稀疏环境更难学。
          epsilon_start/end : 探索率上下限。开始多探索,后期多用已有经验。
          epsilon_decay     : 每局探索率的衰减倍率(每次 × 这个数)。
          buffer_capacity   : 经验池容量。
          batch_size        : 每次更新用的样本数。
          target_update_freq: 目标网络同步间隔(步数)。
          device            : 'cpu' 或 'cuda',默认自动选择。
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 在线网络与目标网络:结构完全一样,参数各自独立
        self.online = QNetwork(input_dim, list(hidden_dims), n_actions).to(self.device)
        self.target = QNetwork(input_dim, list(hidden_dims), n_actions).to(self.device)
        self.hard_update_target()

        # 目标网络只用来"算目标",不参与梯度更新
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.train_steps = 0  # 累计做过多少次梯度更新

    # ---------- 决策 ----------
    def select_action(self, state, greedy=False, legal_mask=None):
        """按 ε-greedy 选动作。

        参数:
          state     : 当前状态(一维数组)。
          greedy    : True = 纯贪心(评估/对战用,不探索);False = 按 ε 探索。
          legal_mask: 黑白棋用。长度为 n_actions 的 0/1 数组,
                      1 表示该动作合法,非法动作的 Q 值会被压成 -∞。
        """
        if legal_mask is not None:
            legal_mask = np.asarray(legal_mask, dtype=bool)
            legal_idx = np.flatnonzero(legal_mask)
            if len(legal_idx) == 0:
                # 防御:全非法掩码说明上游局面管理出了错(比如终局后还
                # 在要动作)。给出明确错误,而不是让后面的 choice/argmax
                # 报出难懂的底层异常。
                raise ValueError("legal_mask 全为 0:当前局面没有任何合法动作")

        # 探索:在合法动作里随机挑一个
        if not greedy and random.random() < self.epsilon:
            if legal_mask is not None:
                return int(np.random.choice(legal_idx))
            return int(np.random.randint(self.n_actions))

        # 利用:取 Q 值最大的动作
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            q = self.online(state_t)
            if legal_mask is not None:
                q = q.clone()
                q[0, ~legal_mask] = -float("inf")  # 非法动作强制压成 -∞
            return int(q.argmax(dim=1).item())

    # ---------- 学习 ----------
    def store(self, s, a, r, s_next, done):
        """把一条经验丢进回放池。"""
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        """从回放池抽一个 batch,做一次梯度更新;返回 loss(或 None)。

        若经验还不够一个 batch,返回 None(此时什么都还没学)。
        """
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        s, a, r, s_next, done = (s.to(self.device), a.to(self.device),
                                 r.to(self.device), s_next.to(self.device),
                                 done.to(self.device))

        # 当前 Q 值:Q(s, a)—— 只取"实际执行的那个动作"对应的 Q 值
        q_now = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # 目标值:r + γ * max Q'(s', a') —— 用"目标网络"算,这是关键
        with torch.no_grad():
            q_next = self.target(s_next).max(dim=1).values
            target = r + self.gamma * q_next * (1.0 - done)  # 结束局没有未来奖励

        loss = nn.functional.mse_loss(q_now, target)  # 让 Q 值逼近目标值

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        # 每隔 target_update_freq 步,把在线网络"抄"给目标网络
        if self.train_steps % self.target_update_freq == 0:
            self.hard_update_target()

        return loss.item()

    # ---------- 辅助 ----------
    def hard_update_target(self):
        """把在线网络的参数完整复制给目标网络(同步)。"""
        self.target.load_state_dict(self.online.state_dict())

    def replace_online(self, net):
        """整体替换在线网络(换模型的唯一正确入口)。

        直接给 self.online 赋新值会留下隐患:optimizer 仍持有旧网络
        的参数引用,一旦有人拿这个 agent 继续训练,梯度会更新到已被
        丢弃的旧网络上。这里替换后同步重建 optimizer 和目标网络。
        """
        self.online = net
        self.optimizer = optim.Adam(self.online.parameters(),
                                    lr=self.optimizer.param_groups[0]["lr"])
        self.hard_update_target()

    def decay_epsilon(self):
        """每局结束调用一次:让探索率逐步下降,模型越来越"认真"。"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """保存模型(只存网络参数 + 超参数,文件小、加载快)。

        原子写:先写临时文件再 os.replace——训练进程保存模型时,
        Web 端可能正通过 /api/models/load 加载同一个文件,直接覆盖
        会让加载方读到半截文件而崩溃。
        """
        tmp = str(path) + ".tmp"
        torch.save({
            "online": self.online.state_dict(),
            "n_actions": self.n_actions,
            "input_dim": self.online.net[0].in_features,
            "hidden_dims": [m.out_features for m in self.online.net if isinstance(m, nn.Linear)][:-1],
        }, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, device=None):
        """从 checkpoint 加载模型(用于界面/对战)。"""
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(input_dim=ckpt["input_dim"],
                    n_actions=ckpt["n_actions"],
                    hidden_dims=tuple(ckpt["hidden_dims"]),
                    device=device)
        agent.online.load_state_dict(ckpt["online"])
        agent.hard_update_target()
        agent.epsilon = agent.epsilon_end  # 加载后默认不探索
        return agent
