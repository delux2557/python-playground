"""共享 DQN 组件(shared/dqn.py)的单元测试。

覆盖:网络输出形状、经验回放池、ε-greedy 探索、目标网络同步、模型存取。
"""

import numpy as np
import pytest
import torch

from shared.dqn import DQNAgent, QNetwork, ReplayBuffer


def make_agent(**kwargs):
    defaults = dict(input_dim=11, n_actions=4, hidden_dims=(32, 32),
                    buffer_capacity=500, batch_size=16,
                    target_update_freq=100, epsilon_end=0.01)
    defaults.update(kwargs)
    return DQNAgent(**defaults)


# ---------------- Q 网络 ----------------
def test_qnetwork_output_shape():
    net = QNetwork(input_dim=11, hidden_dims=(64, 64), output_dim=4)
    out = net(torch.randn(5, 11))
    assert out.shape == (5, 4)


# ---------------- 经验回放 ----------------
def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(capacity=1000)
    for i in range(10):
        buf.push(np.zeros(11, dtype=np.float32), 1, 0.5,
                 np.zeros(11, dtype=np.float32), False)
    s, a, r, s_next, done = buf.sample(4)
    assert s.shape == (4, 11)
    assert a.shape == (4,)
    assert r.shape == (4,)
    assert s_next.shape == (4, 11)
    assert done.shape == (4,)


def test_replay_buffer_drops_oldest():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(np.full(11, i, dtype=np.float32), i, 0.0,
                 np.zeros(11, dtype=np.float32), False)
    assert len(buf) == 3
    s, *_ = buf.sample(3)
    # 只能抽到最后 3 条(编号 2、3、4),最旧的 0、1 已被丢弃
    assert set(s[:, 0].tolist()) == {2.0, 3.0, 4.0}


# ---------------- ε-greedy 探索 ----------------
def test_greedy_action_matches_argmax():
    agent = make_agent()
    state = np.array([0.5, -0.3, 1.0, 0.0, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0, 0.1],
                     dtype=np.float32)
    with torch.no_grad():
        q = agent.online(torch.as_tensor(state).unsqueeze(0))
        expected = int(q.argmax(dim=1).item())
    assert agent.select_action(state, greedy=True) == expected


def test_high_epsilon_explores_beyond_argmax():
    agent = make_agent(n_actions=6)
    agent.epsilon = 1.0
    state = np.zeros(11, dtype=np.float32)
    with torch.no_grad():
        q = agent.online(torch.as_tensor(state).unsqueeze(0))
        argmax = int(q.argmax(dim=1).item())
    actions = {agent.select_action(state, greedy=False) for _ in range(200)}
    # ε=1 应该完全随机:不仅会选到 argmax,还会选到别的动作
    assert actions != {argmax}
    assert len(actions) > 1


def test_legal_mask_respected_in_random_action():
    agent = make_agent(n_actions=6)
    agent.epsilon = 1.0
    state = np.zeros(11, dtype=np.float32)
    legal = np.array([0, 0, 1, 1, 0, 0])
    actions = {agent.select_action(state, greedy=False, legal_mask=legal)
               for _ in range(100)}
    assert actions <= {2, 3}  # 永远只能从合法动作里选


# ---------------- 目标网络 ----------------
def test_hard_update_target_syncs_weights():
    agent = make_agent()
    agent.online.load_state_dict(agent.target.state_dict())  # 先同步成一样
    # 手动把在线网络改乱
    with torch.no_grad():
        agent.online.net[0].weight.add_(1.0)
    assert not _weights_equal(agent.online, agent.target)
    agent.hard_update_target()
    assert _weights_equal(agent.online, agent.target)


# ---------------- 梯度更新 ----------------
def test_update_returns_none_when_buffer_small():
    agent = make_agent()
    assert agent.update() is None


def test_update_returns_loss_after_enough_data():
    agent = make_agent()
    state = np.random.rand(11).astype(np.float32)
    for _ in range(agent.batch_size + 5):
        agent.store(state, 1, 0.0, state, False)
    loss = agent.update()
    assert isinstance(loss, float)
    assert loss >= 0.0


# ---------------- 模型存取 ----------------
def test_save_load_roundtrip(tmp_path):
    agent = make_agent()
    state = np.random.rand(11).astype(np.float32)
    for _ in range(agent.batch_size + 5):
        agent.store(state, 1, 0.0, state, False)
    agent.update()
    path = tmp_path / "agent.pt"
    agent.save(str(path))

    loaded = DQNAgent.load(str(path))
    assert _weights_equal(agent.online, loaded.online)
    assert loaded.epsilon == loaded.epsilon_end  # 加载后默认不探索


def _weights_equal(a, b):
    for pa, pb in zip(a.parameters(), b.parameters()):
        if not torch.equal(pa.data, pb.data):
            return False
    return True
