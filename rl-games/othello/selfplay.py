"""othello/selfplay.py —— 黑白棋自对弈训练(当前模型 vs 对手池)。

为什么自对弈(对应 DESIGN.md 第 5.3 节):
  黑白棋是"双人零和"游戏,你的赢就是对手的输。和贪吃蛇"自己 vs 环境"
  不同,这里需要一个对手。让模型一直和自己最新版本打,会陷入
  "两个模型互相 exploit 同一个漏洞、原地打转";所以让模型去和
  "自己以前的历史版本"(对手池)打,策略才能稳定提升——
  这是 AlphaGo 时代就验证过的经典做法(采样旧版本当对手)。

模块职责:
  play_game()          : 打一局,收集"挑战者视角"的每一步经验
  evaluate_win_rate()  : 让模型用纯贪心打随机对手,量"胜率"评估水平
  SelfPlayTrainer      : 整个训练循环:挑对手 → 打局 → 存经验 → 更新 → 存档
"""

import random

import numpy as np

from othello.dqn import OthelloDQNConfig, make_agent
from othello.env import N_ACTIONS, BLACK, OthelloEnv, WHITE


def _legal_mask(env, player) -> np.ndarray:
    """把"合法落子集合"变成 64 维 0/1 掩码(DQN 选动作要用)。"""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[env.legal_moves(player)] = True
    return mask


def play_game(challenger, opponent, env: OthelloEnv):
    """打一局完整对局,返回 (transitions, outcome)。

    参数:
      challenger: 正在训练的模型(挑战者,带 ε 探索)
      opponent  : 对手。None = 纯随机(保留基线);否则是一个模型(纯贪心)
      env       : 黑白棋环境(每局复用同一个,内部会 reset)
    返回:
      transitions: 挑战者视角的每一步 (s, a, r=0, s', done),r 先占位
      outcome    : 挑战者视角的终局结果:+1 胜 / -1 负 / 0 平

    两个关键设计(新手注意):
      1. 谁执黑随机决定——消除"先手优势"偏差,让模型黑白都会下;
      2. 存经验时用 challenger 自己的视角编码棋盘(显式传 player),
         否则棋盘会编码成"对手视角",模型就学反了。
    """
    env.reset()
    challenger_color = BLACK if random.random() < 0.5 else WHITE

    transitions = []
    done = False
    obs = env._get_obs()                 # 当前落子方视角的观察
    while not done:
        current = env.current
        legal = env.legal_moves(current)
        if not legal:
            # 防御:理论上不会到这(env 内部已自动让子),
            # 留着避免"空掩码选动作"报出难懂的错。
            obs = env._get_obs()
            continue

        if current == challenger_color:
            # 挑战者出招:带 ε 探索,从合法落子里选
            action = challenger.select_action(obs, greedy=False,
                                              legal_mask=_legal_mask(env, current))
            s_before = env._get_obs(challenger_color)
            _, _, done, _ = env.step(action)
            s_after = env._get_obs(challenger_color)   # 落完子,仍是挑战者视角
            transitions.append((s_before, action, 0.0, s_after, done))
        else:
            # 对手出招:纯随机(opponent=None)或纯贪心模型
            if opponent is None:
                action = int(np.random.choice(legal))
            else:
                action = opponent.select_action(obs, greedy=True,
                                                legal_mask=_legal_mask(env, current))
            _, _, done, _ = env.step(action)
        obs = env._get_obs()

    # 终局结算:挑战者视角的结果
    w = env.winner
    if w == 0:
        outcome = 0.0
    else:
        outcome = 1.0 if w == challenger_color else -1.0
    return transitions, outcome


def evaluate_win_rate(agent, games=30, seed=0):
    """让模型用纯贪心对战"纯随机对手",返回胜率(平局算半分)。

    为什么打随机而不是打自己:评估必须"独立于训练过程",
    用固定、简单的参照物(随机)才能稳定反映模型水平的变化。
    评估时 agent 必须 ε=0(纯贪心),否则胜率被探索"拉低",测不准。

    seed 真正生效:随机对手用独立的 Generator,不碰全局 np.random——
    这样"固定种子统一基准"才成立,也不会污染进程里其它随机调用。
    """
    rng = np.random.default_rng(seed)
    env = OthelloEnv()
    wins = draws = 0.0
    for i in range(games):
        env.reset()
        agent_color = BLACK if i % 2 == 0 else WHITE  # 黑白各一半,消除先手偏差
        done = False
        obs = env._get_obs()
        while not done:
            current = env.current
            legal = env.legal_moves(current)
            if not legal:
                obs = env._get_obs()
                continue
            if current == agent_color:
                action = agent.select_action(obs, greedy=True,
                                             legal_mask=_legal_mask(env, current))
            else:
                action = int(rng.choice(legal))
            _, _, done, _ = env.step(action)
            obs = env._get_obs()
        w = env.winner
        if w == 0:
            draws += 1
        elif w == agent_color:
            wins += 1
    return (wins + 0.5 * draws) / games


class SelfPlayTrainer:
    """自对弈训练器:管好"挑战者模型"和"对手池",逐局训练。

    训练循环(对应 DESIGN.md 第 5.3 节):
      1. 从对手池挑一个对手(偶尔打纯随机,保留基线)
      2. 打一局,收集挑战者每一步的经验
      3. 把"终局胜负"回传给整局的每一步,存入经验池(蒙特卡洛回传)
      4. 抽批更新网络(和贪吃蛇同一套 DQN)
      5. 定期把当前模型"冻结存档"进对手池,淘汰最旧的
    """

    def __init__(self, cfg: OthelloDQNConfig):
        self.cfg = cfg
        self.agent = make_agent(cfg)     # 正在训练的挑战者
        self.opponent_pool = []          # 历史模型快照(冻结,不再更新)
        self.env = OthelloEnv()
        self.eval_indices = []           # 评估时的局号(曲线横轴)
        self.eval_win_rates = []         # 评估胜率(曲线纵轴)

    # ---------------- 对手管理 ----------------
    def _pick_opponent(self):
        """挑对手:按概率打纯随机(None),否则从对手池均匀抽一个。"""
        if (not self.opponent_pool
                or random.random() < self.cfg.random_opponent_prob):
            return None
        return random.choice(self.opponent_pool)

    def _save_opponent(self):
        """把当前模型的"冻结副本"存进对手池,满了淘汰最旧的。

        冻结 = 复制网络参数但不更新。对手池里是"过去的自己",
        模型每次面对的都是"打不过也打不过的历史版本",策略才稳。
        """
        clone = make_agent(self.cfg)
        clone.online.load_state_dict(self.agent.online.state_dict())
        clone.hard_update_target()
        clone.epsilon = clone.epsilon_end
        self.opponent_pool.append(clone)
        if len(self.opponent_pool) > self.cfg.opponent_pool_size:
            self.opponent_pool.pop(0)

    # ---------------- 训练主循环 ----------------
    def run(self, verbose=True, progress_cb=None, start_ep=0):
        """跑完整训练,返回评估曲线 {"episodes": [...], "win_rates": [...]}。

        参数:
          verbose    : 是否在终端打印进度
          progress_cb: 每次评估后回调(训练/Web 解耦的钩子)。回调收到
                       {"episode", "win_rate", "epsilon", "opponent_pool"}
                       这样训练逻辑不用关心进度写到哪,由上层(如 train.py
                       的 ProgressReporter)决定是打日志、写文件还是推送。
          start_ep   : 断点续训的起点(已训练局数)。>0 时从 start_ep+1 局
                       继续往下训,self.agent 需由调用方预先加载 checkpoint。
        """
        for ep in range(start_ep + 1, self.cfg.episodes + 1):
            opponent = self._pick_opponent()
            transitions, outcome = play_game(self.agent, opponent, self.env)

            # 把整局结果回传给挑战者的每一步(蒙特卡洛式打标签)
            for s, a, _, s_next, done in transitions:
                self.agent.store(s, a, outcome, s_next, done)

            # 用这局收集的经验做梯度更新(每步经验抽一次批)
            for _ in transitions:
                self.agent.update()

            self.agent.decay_epsilon()   # 每局降一点探索率

            # 定期把当前模型存档进对手池
            if ep % self.cfg.opponent_save_freq == 0:
                self._save_opponent()

            # 定期评估:对战纯随机,量胜率
            if ep % self.cfg.eval_freq == 0:
                wr = evaluate_win_rate(self.agent, games=self.cfg.eval_games,
                                       seed=self.cfg.seed + ep)
                self.eval_indices.append(ep)
                self.eval_win_rates.append(wr)
                if verbose:
                    print(f"[局 {ep:>5}/{self.cfg.episodes}] "
                          f"ε={self.agent.epsilon:.3f} "
                          f"对手池={len(self.opponent_pool)} "
                          f"对随机胜率={wr:.2f}")
                if progress_cb:
                    progress_cb({"episode": ep, "win_rate": wr,
                                 "epsilon": self.agent.epsilon,
                                 "opponent_pool": len(self.opponent_pool)})

        return {"episodes": self.eval_indices, "win_rates": self.eval_win_rates}
