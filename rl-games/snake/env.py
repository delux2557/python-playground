"""snake/env.py —— 贪吃蛇游戏环境(只负责"游戏规则",不管算法)。

这是一个"Gymnasium 风格"的环境:任何强化学习算法都通过
    obs = env.reset()                 # 开局
    obs, reward, done, info = env.step(action)   # 走一步
来和它交互。好处是游戏逻辑和算法完全解耦,换个游戏环境即可。

设计要点(对应 DESIGN.md 第 4.1 节):
  1. 动作:0=上 1=下 2=左 3=右;禁止 180° 直接掉头(否则模型会原地打转)
  2. 奖励塑形:吃到食物 +10、撞墙/撞自己 -10、每步 -0.1(防绕圈)
  3. 状态(observation):11 维特征向量,告诉模型"怎么看局面"
  4. 超时保护:超过 max_steps_since_food 步没吃到就强制结束,防止死循环
"""

import numpy as np

# 动作编号 → 移动方向(dr, dc)。行向下为正,列向右为正
# 0=上(-1,0)  1=下(+1,0)  2=左(0,-1)  3=右(0,+1)
_ACTION_DELTA = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
# 与某个方向相反的方向(用于禁止 180° 掉头)
_REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}

# 观察向量的维度(设计文档约定为 11)
STATE_DIM = 11


class SnakeEnv:
    """贪吃蛇环境。

    主要属性(测试和调试时可以直接读写它们):
      body     : 蛇身坐标列表,body[0] 是蛇头,元素为 (row, col)
      direction: 当前移动方向(动作编号 0~3)
      food     : 食物坐标 (row, col)
      score    : 当前得分(吃到的食物数)
      steps    : 本局总步数
      steps_since_food: 距上次吃到食物已经走了多少步
    """

    def __init__(self, grid_size=11, reward_eat=10.0, reward_death=-10.0,
                 reward_step=-0.1, max_steps_since_food=None, seed=None):
        """
        参数说明(每个参数影响什么,对应 DESIGN.md 第 4.1 节奖励塑形):
          grid_size        : 棋盘边长(默认 11×11)。越小训练越快,越大越难。
          reward_eat       : 吃到食物的奖励。这是模型"主动去吃"的主要动力。
          reward_death     : 撞墙/撞自己的惩罚。越大模型越惜命。
          reward_step      : 每走一步的小惩罚。防止模型原地绕圈刷步数。
          max_steps_since_food: 多少步没吃到就强制结束(防死循环)。
                             默认 = 4 倍格子数,大约是一局合理步数的上限。
          seed             : 随机种子。固定后结果可复现(测试会用到)。
        """
        self.grid_size = grid_size
        if grid_size < 3:
            # 初始蛇身长度 3,棋盘再小就摆不下(会生成负坐标)
            raise ValueError(f"grid_size 至少为 3,当前 {grid_size}")
        self.reward_eat = reward_eat
        self.reward_death = reward_death
        self.reward_step = reward_step
        # 用 is None 判断:显式传 0 也是合法意图(虽然没意义),不能被 or 吞掉
        self.max_steps_since_food = (4 * grid_size * grid_size
                                     if max_steps_since_food is None
                                     else max_steps_since_food)
        self.rng = np.random.default_rng(seed)

        self.body = []
        self.direction = 3  # 初始向右
        self.food = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False   # 本局是否已结束(撞墙/撞身/超时)

    # ---------------- 与算法交互的接口 ----------------
    def reset(self):
        """开始新一局。蛇初始在棋盘中央、长度 3、向右移动。"""
        c = self.grid_size // 2
        self.body = [(c, c), (c, c - 1), (c, c - 2)]  # body[0] 是蛇头
        self.direction = 3  # 向右
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self._place_food()
        return self._get_obs(), {}

    def step(self, action):
        """执行一个动作,返回 (obs, reward, done, info)。

        action 非法(不是 0~3)会抛 ValueError——算法不该传非法动作。
        本局已结束(done=True)后再 step 同样抛 ValueError——必须先
        reset(),否则蛇会"死后复活"从死亡前的状态继续走。
        """
        if self.done:
            raise ValueError("本局已结束,请先 reset() 开新局")
        if action not in _ACTION_DELTA:
            raise ValueError(f"非法动作 {action},动作必须是 0~3")

        # 关键规则:如果动作是"往当前方向的反方向走"(180° 掉头),
        # 就强制改成继续当前方向,避免蛇原地对折、也避免模型学会自杀式打转。
        if action == _REVERSE[self.direction]:
            action = self.direction
        self.direction = action
        dr, dc = _ACTION_DELTA[action]

        self.steps += 1
        self.steps_since_food += 1

        # 新蛇头位置
        head_r, head_c = self.body[0]
        new_head = (head_r + dr, head_c + dc)

        # 1) 撞墙?
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.done = True
            return self._get_obs(), self.reward_death, True, {"reason": "crash_wall"}

        # 2) 撞到自己?(注意:尾巴那一格马上要被移走,所以不算撞)
        if new_head in self.body[:-1]:
            self.done = True
            return self._get_obs(), self.reward_death, True, {"reason": "crash_body"}

        # 3) 正常移动:蛇头先走一步
        self.body.insert(0, new_head)

        # 4) 吃到食物?
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            return self._get_obs(), self.reward_eat, False, {"reason": "ate"}

        # 5) 没吃到:尾巴移除(蛇身长度不变)
        self.body.pop()

        # 6) 超时保护:太久没吃到,强制结束,防止死循环白耗时间
        if self.steps_since_food >= self.max_steps_since_food:
            self.done = True
            return self._get_obs(), self.reward_death, True, {"reason": "timeout"}

        return self._get_obs(), self.reward_step, False, {"reason": "move"}

    # ---------------- 观察编码 ----------------
    def _get_obs(self):
        """把"当前局面"编码成 11 维向量(这就是模型看到的输入)。

        维度含义:
          [0:4]   四个方向的"危险"标记:该方向下一步会不会撞墙或撞自己
          [4:6]   食物相对蛇头的方向(dx, dy,已归一化到 [-1,1])
          [6:10]  当前移动方向 one-hot(上/下/左/右)
          [10]    距上次吃食物已走步数(归一化),帮模型学会"别绕圈"
        """
        head_r, head_c = self.body[0]
        g = self.grid_size
        body = set(self.body)

        # 4 个危险标记:撞墙或撞到自己 → 1
        danger = []
        for dr, dc in [_ACTION_DELTA[i] for i in range(4)]:
            nr, nc = head_r + dr, head_c + dc
            out = not (0 <= nr < g and 0 <= nc < g)
            danger.append(1.0 if (out or (nr, nc) in body) else 0.0)

        # 食物方向(归一化到 [-1, 1])
        dx = (self.food[1] - head_c) / g
        dy = (self.food[0] - head_r) / g

        # 当前方向 one-hot
        direction_onehot = [0.0, 0.0, 0.0, 0.0]
        direction_onehot[self.direction] = 1.0

        # 距上次吃食物的步数(归一化)
        starve = self.steps_since_food / self.max_steps_since_food

        return np.array(danger + [dx, dy] + direction_onehot + [starve],
                        dtype=np.float32)

    def state_grid(self):
        """返回整张棋盘的二维数组,供可视化/测试使用。

        取值:0=空地 1=蛇身 2=蛇头 3=食物
        """
        g = self.grid_size
        grid = np.zeros((g, g), dtype=int)
        for r, c in self.body[1:]:
            grid[r, c] = 1
        hr, hc = self.body[0]
        grid[hr, hc] = 2
        grid[self.food[0], self.food[1]] = 3
        return grid

    # ---------------- 内部工具 ----------------
    def _place_food(self):
        """在"没有蛇身"的格子里随机放一个食物。"""
        g = self.grid_size
        occupied = set(self.body)
        free = [(r, c) for r in range(g) for c in range(g)
                if (r, c) not in occupied]
        if not free:
            # 棋盘被蛇占满 → 其实已经赢了,但为简单起见直接放回蛇头位置(不会触发)
            self.food = self.body[0]
            return
        self.food = free[int(self.rng.integers(len(free)))]
