"""shared/platform.py —— 对局平台的核心抽象(把"游戏服务"升级成"对战平台")。

这一层回答一个问题:如何让"训练模型、人类玩家、随机基线"都以同一种方式
接入同一套对局引擎?答案是把三件事拆开——

  1. Player       : 参与方。人 / 模型 / 训练器 / 随机基线都是 Player,
                    平台只认 decide(obs, legal_mask) -> action 这一个动作。
  2. GameAdapter  : 游戏插件。纯规则,无状态——怎么开局、谁轮到、
                    怎么观察、哪些动作合法、如何落子、怎么判终局。
  3. Match        : 一场有状态的对局。把 N 个 Player 放到一个 GameAdapter
                    上,负责回合仲裁、记录棋谱、驱动对局走完。

这样分工后(对应业内"环境/智能体分离"的最佳实践):
  · 新增游戏     = 写一个 GameAdapter,平台自动获得对战/评测/打榜能力
  · 新增参与方   = 写一个 Player(模型、人、训练器都一样)
  · 平台只做仲裁与编排,不懂任何具体游戏规则

重要约定(训练热路径):
  训练时的高频自对弈应在进程内直接调用,不要把每步决策都走网络——
  平台负责"编排"(开对局、派任务、托管),而不是充当训练的热路径。
"""

import queue
from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------
# 1. Player:统一参与方接口
# ---------------------------------------------------------------
class Player(ABC):
    """一个"会做决定"的参与方。

    人类、已训模型、训练器、随机基线——全都实现这个接口。
    平台(Match)只调用 decide(),不关心决定是怎么做出来的。
    """

    name: str = "player"

    @abstractmethod
    def decide(self, obs, legal_mask=None) -> int:
        """看局面,返回一个动作编号。

        参数:
          obs        : 以"当前该落子的一方"视角编码的观察向量
          legal_mask : 长度 = 动作数的 0/1 数组(1 = 合法);None = 全部合法
        """

    def on_match_end(self, result: dict):
        """对局结束回调(可选)。训练器用它记经验/更新模型,普通参与方忽略。"""


class ModelPlayer(Player):
    """把一个 DQNAgent 包装成参与方:decide = 调用智能体选动作。

    持有的是 agent 对象的引用——换模型若用 replace_online 原地替换网络,
    这个 Player 会自动跟上(仍指向同一个 agent,读到的已是新网络)。
    """

    def __init__(self, agent, name: str = "model", greedy: bool = True):
        self.agent = agent
        self.name = name
        self.greedy = greedy      # 对战/评测默认纯贪心(不探索)

    def decide(self, obs, legal_mask=None) -> int:
        return int(self.agent.select_action(
            obs, greedy=self.greedy, legal_mask=legal_mask))


class RandomPlayer(Player):
    """随机基线:在合法动作里随机挑一个(评测的"地板"、冒烟测试用)。"""

    def __init__(self, name: str = "random", seed=None):
        self.name = name
        self.rng = np.random.default_rng(seed)

    def decide(self, obs, legal_mask=None) -> int:
        if legal_mask is None:
            raise ValueError("RandomPlayer 需要 legal_mask 才知道哪些动作合法")
        legal = np.flatnonzero(np.asarray(legal_mask, dtype=bool))
        if len(legal) == 0:
            raise ValueError("当前局面没有任何合法动作")
        return int(self.rng.choice(legal))


class HumanPlayer(Player):
    """人类参与方:动作由外部提交,decide() 从队列里取。

    Web 人机对战通常不直接用它——服务拿到 HTTP 请求里的动作后,直接调
    Match.act() 落子。这个类更多用于"异步房间"或测试:预先提交一整局
    的落子,就能驱动 Match.play() 走完。
    """

    def __init__(self, name: str = "human", timeout=None):
        self.name = name
        self.timeout = timeout
        self._q: queue.Queue = queue.Queue()

    def submit(self, action):
        """外部(键盘 / 网络)把人类落子推进来。"""
        self._q.put(int(action))

    def decide(self, obs, legal_mask=None) -> int:
        try:
            action = self._q.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("HumanPlayer 还没收到人类输入")
        if legal_mask is not None:
            legal = np.asarray(legal_mask, dtype=bool)
            if not (0 <= action < len(legal)) or not legal[action]:
                raise ValueError(f"人类输入 {action} 不是合法落子")
        return int(action)


# ---------------------------------------------------------------
# 2. GameAdapter:游戏插件(纯规则,无状态)
# ---------------------------------------------------------------
class GameAdapter(ABC):
    """一个游戏的"规则插件"。

    平台靠这组方法与任何游戏交互;游戏不依赖平台,平台也不懂游戏规则。
    每个方法都以一个 game 状态对象为参数——adapter 自己不持有任何对局,
    因此是"无状态"的,可以同时服务多场对局。
    """

    name: str = "game"

    @abstractmethod
    def new_game(self):
        """创建并返回一个全新的对局状态对象(互不共享)。"""

    @abstractmethod
    def current_player(self, game) -> int:
        """当前轮到谁(参与方编号)。"""

    @abstractmethod
    def observe(self, game, player) -> np.ndarray:
        """以 player 视角编码的观察向量(喂给模型)。"""

    def legal_mask(self, game, player):
        """长度 = 动作数的 0/1 数组;默认返回 None 表示"全部合法"。"""
        return None

    @abstractmethod
    def apply(self, game, action):
        """落一步子(调用方保证合法;非法时底层环境会抛错)。"""

    @abstractmethod
    def done(self, game) -> bool:
        """对局是否结束。"""

    @abstractmethod
    def result(self, game) -> dict:
        """终局结算。至少含 {"winner": ...},可附带 counts / score 等。"""


# ---------------------------------------------------------------
# 3. Match:一场有状态的对局(仲裁 + 棋谱 + 驱动)
# ---------------------------------------------------------------
class Match:
    """一场对局:把若干 Player 放到一个 GameAdapter 上。

    两种驱动方式:
      · play()      : 同步打完整局——所有参与方都是"自动"的(模型/随机/训练器),
                      用于对战、竞技场、自对弈。
      · act()/auto_step(): 一步一步走——人类落子用 act() 注入动作,
                      模型落子用 auto_step(),用于 Web 人机对战逐回合驱动。
    """

    def __init__(self, adapter: GameAdapter, players: dict):
        """
        参数:
          adapter : 游戏规则插件
          players : {参与方编号: Player}。人机对战里,人类的座位可以不放
                    Player(由 act() 直接注入动作),只给 AI 座位放 ModelPlayer。
        """
        self.adapter = adapter
        self.players = dict(players)
        self.game = adapter.new_game()
        self.moves: list[tuple] = []        # [(参与方编号, action)]

    # ---------------- 查询 ----------------
    def current(self) -> int:
        """当前轮到谁。"""
        return self.adapter.current_player(self.game)

    def done(self) -> bool:
        return self.adapter.done(self.game)

    def observe(self, player=None):
        """某方(默认当前方)视角的观察。"""
        return self.adapter.observe(
            self.game, self.current() if player is None else player)

    def legal_mask(self, player=None):
        """某方(默认当前方)的合法动作掩码。"""
        return self.adapter.legal_mask(
            self.game, self.current() if player is None else player)

    # ---------------- 驱动 ----------------
    def act(self, action) -> tuple:
        """落一步子并记录,返回 (落子方, action)。

        由调用方注入动作(人类落子走这里)。合法性交给底层环境校验,
        非法会抛错。
        """
        if self.done():
            raise ValueError("对局已结束")
        p = self.current()
        self.adapter.apply(self.game, int(action))
        self.moves.append((p, int(action)))
        return p, int(action)

    def auto_step(self) -> tuple:
        """让当前参与方自己决策并落子,返回 (落子方, action)。

        模型/随机/训练器走这里。若当前座位没注册 Player(比如轮到了
        人类却想 auto_step),抛 KeyError——这正是"回合秩序"的天然防线。
        """
        p = self.current()
        player = self.players.get(p)
        if player is None:
            raise KeyError(f"参与方 {p} 没有注册 Player,不能自动落子")
        action = player.decide(self.observe(p), self.legal_mask(p))
        return self.act(action)

    def play(self) -> dict:
        """同步打完整局(要求所有参与方都是自动的),返回终局记录。"""
        while not self.done():
            self.auto_step()
        return self.record()

    def reset(self):
        """重开一局:换新对局状态、清空棋谱。"""
        self.game = self.adapter.new_game()
        self.moves.clear()

    # ---------------- 记录 ----------------
    def record(self) -> dict:
        """终局(或进行中)记录:棋谱 + 结算结果。"""
        return {"moves": list(self.moves), **self.adapter.result(self.game)}
