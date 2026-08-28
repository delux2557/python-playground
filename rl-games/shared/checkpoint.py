"""shared/checkpoint.py —— 训练断点续训的共享工具。

训练动辄几百上千局,中途崩了(断电/OOM/手动停)从零重跑太浪费。
续训方案:
  1. 训练过程中"定期"把模型 + 一个很小的 meta 文件(已训练局数/探索率)
     原子存盘(崩溃最多丢 eval_freq 局);
  2. 下次启动时若 checkpoint 存在,自动加载权重并"接着上次的局数"往下训,
     探索率也从上次的位置继续衰减,而不是退回起点。

模型文件保持 DQNAgent 的标准格式(通用,能被 /api/models/load 加载);
续训专用的 meta 用旁路文件 <checkpoint>.meta.json 存,不影响模型本身。
"""

import json
import os
from pathlib import Path

from shared.dqn import DQNAgent


def save_with_meta(agent: DQNAgent, checkpoint_path, meta: dict):
    """原子保存模型 + 续训 meta。

    agent          : DQNAgent
    checkpoint_path: 模型保存路径(结尾 .pt)
    meta           : 至少含 {"episode": 已训练局数, "epsilon": 当前探索率}
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))                      # DQNAgent.save 内部已是原子写
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    tmp = str(meta_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp, meta_path)


def read_meta(checkpoint_path) -> dict:
    """读续训 meta;不存在(首次训练/还没存过档)或损坏时返回空 dict。"""
    path = Path(checkpoint_path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def load_for_resume(checkpoint_path, resume_epsilon=None):
    """若 checkpoint 存在,加载它用于续训。

    返回 (agent | None, start_ep, resume_epsilon):
      agent          : 加载好的 DQNAgent;checkpoint 不存在则为 None
      start_ep       : 已训练局数(续训的起点,接着往下训)
      resume_epsilon : 要恢复的探索率(优先参数,否则读 meta;都无则 None)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None, 0, None
    agent = DQNAgent.load(str(path))
    meta = read_meta(checkpoint_path)
    start_ep = int(meta.get("episode", 0))
    eps = resume_epsilon
    if eps is None:
        eps = meta.get("epsilon")
    return agent, start_ep, eps
