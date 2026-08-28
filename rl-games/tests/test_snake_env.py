"""贪吃蛇环境(snake/env.py)的单元测试。

测试策略:直接读写 env 的内部状态(body/food/direction 等),
构造"确定性的局面",验证每一条游戏规则是否按设计工作。
"""

import numpy as np
import pytest

from snake.env import SnakeEnv, STATE_DIM


def make_env(**kwargs):
    kwargs.setdefault("seed", 0)
    return SnakeEnv(grid_size=11, **kwargs)


# ---------------- 开局与观察 ----------------
def test_reset_returns_valid_state():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape == (STATE_DIM,)
    assert obs.dtype == np.float32
    assert env.score == 0
    assert len(env.body) == 3          # 蛇初始长度 3
    assert env.body[0] not in env.body[1:]  # 蛇头在蛇身外(废话,但保险)
    assert env.food not in env.body     # 食物不能放在蛇身上
    assert info == {}


def test_observation_dimension_bounds():
    env = make_env()
    obs, _ = env.reset()
    # 危险标记(0~3)必须是 0 或 1
    assert set(obs[0:4]) <= {0.0, 1.0}
    # 方向 one-hot(6~9)必须恰好有一个 1
    assert obs[6:10].sum() == pytest.approx(1.0)
    # 食物方向(4~5)归一化到 [-1,1]
    assert np.all((obs[4:6] >= -1.0) & (obs[4:6] <= 1.0))
    # 饥饿度(10)归一化到 [0,1]
    assert 0.0 <= obs[10] <= 1.0


def test_action_contract_rejects_invalid():
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)   # 只有 0~3 合法
    with pytest.raises(ValueError):
        env.step(-1)


# ---------------- 游戏规则 ----------------
def test_reverse_action_is_blocked():
    """向右走时按"左"应被强制改成"右"(禁止 180° 掉头)。"""
    env = make_env()
    env.reset()
    env.direction = 3  # 向右
    head_r, head_c = env.body[0]
    env.step(2)        # 按左(与当前方向相反)
    assert env.body[0] == (head_r, head_c + 1)  # 实际向右移动了
    assert env.direction == 3


def test_eat_food_grows_and_scores():
    env = make_env()
    env.reset()
    # 把食物放到蛇头正前方,方向向右
    head_r, head_c = env.body[0]
    env.direction = 3
    env.food = (head_r, head_c + 1)
    before_len = len(env.body)

    obs, reward, done, info = env.step(3)
    assert reward == pytest.approx(env.reward_eat)   # +10
    assert done is False
    assert env.score == 1
    assert len(env.body) == before_len + 1           # 吃到食物 → 变长
    assert info["reason"] == "ate"


def test_crash_wall_ends_episode():
    env = make_env()
    env.reset()
    # 蛇头放到最上一行,方向向上 → 下一步必撞墙
    env.body = [(0, 5), (0, 4), (0, 3)]
    env.direction = 0
    env.food = (9, 9)  # 食物放远,排除干扰

    obs, reward, done, info = env.step(0)
    assert done is True
    assert reward == pytest.approx(env.reward_death)  # -10
    assert info["reason"] == "crash_wall"


def test_crash_body_ends_episode():
    env = make_env()
    env.reset()
    # 构造"前方就是自己身体"的局面:头向右,身体挡在右侧
    env.body = [(5, 5), (5, 6), (5, 7), (6, 7)]
    env.direction = 3
    env.food = (0, 0)

    obs, reward, done, info = env.step(3)
    assert done is True
    assert reward == pytest.approx(env.reward_death)
    assert info["reason"] == "crash_body"


def test_timeout_ends_episode():
    env = make_env()
    env.reset()
    env.direction = 3
    env.food = (0, 0)  # 食物放远,保证这步吃不到
    env.steps_since_food = env.max_steps_since_food - 1

    obs, reward, done, info = env.step(3)
    assert done is True
    assert info["reason"] == "timeout"
    assert reward == pytest.approx(env.reward_death)


def test_observation_danger_marks_wall():
    env = make_env()
    env.reset()
    env.body = [(0, 5), (0, 4), (0, 3)]
    env.direction = 0
    obs = env._get_obs()
    assert obs[0] == 1.0   # 上方是墙 → 危险


def test_state_grid_contract():
    env = make_env()
    env.reset()
    grid = env.state_grid()
    assert grid.shape == (env.grid_size, env.grid_size)
    assert set(np.unique(grid)) <= {0, 1, 2, 3}
    assert 2 in grid  # 蛇头
    assert 3 in grid  # 食物


# ---------------- 可复现性与食物放置 ----------------
def test_seed_reproducible():
    env1 = make_env(seed=7)
    env2 = make_env(seed=7)
    env1.reset()
    env2.reset()
    assert env1.food == env2.food
    # 同一动作序列应产生完全相同的局面
    for action in [3, 3, 0, 2, 3, 1]:
        o1, r1, d1, _ = env1.step(action)
        o2, r2, d2, _ = env2.step(action)
        assert np.array_equal(o1, o2)
        assert r1 == r2 and d1 == d2


def test_food_never_on_snake_during_random_play():
    env = make_env(seed=123)
    env.reset()
    for _ in range(300):
        if np.random.random() < 0.5:
            action = int(np.random.randint(4))
        else:
            # 允许"往食物方向走"的随机策略,增加吃到食物的概率
            hr, hc = env.body[0]
            action = 3 if env.food[1] > hc else (2 if env.food[1] < hc else 0)
        obs, reward, done, _ = env.step(action)
        assert env.food not in env.body  # 食物绝不能长在蛇身上
        if done:
            env.reset()
