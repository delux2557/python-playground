"""shared/registry.py —— 模型注册表(平台化的"模型可切换"落地点)。

为什么需要它:
  界面要能"列出训练过的模型、切换用哪个、对比各自水平",
  光有 models/*.pt 文件不够——还得知道每个文件属于哪个游戏、
  用什么算法训的、评估分多少、什么时候训的。
  这份"模型的身份证"就存在 models/registry.json 里。

文件结构(JSON):
  {
    "snake-20260827-1430": {
      "game": "snake",
      "algorithm": "dqn",
      "path": "models/snake.pt",
      "hidden_dims": [64, 64],
      "eval_score": 16.2,
      "episodes": 3000,
      "created_at": "2026-08-27T14:30:00"
    }
  }

训练脚本在保存模型后调 register() 自动登记;
界面通过服务端的 /api/models 列出、/api/models/load 切换。
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

# 注册表文件位置:与项目根目录下的 models/ 平级
_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models" / "registry.json"

# 读写加锁:训练进程和 Web 服务可能同时写
_lock = threading.Lock()


def _read_unlocked() -> dict:
    """读注册表(调用方需持锁)。文件缺失/损坏时返回空表而不是崩溃。"""
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_unlocked(data: dict):
    """原子写注册表(调用方需持锁):先写临时文件再 os.replace 替换。

    为什么必须原子:另一个进程可能随时在读,直接覆盖写会读到
    "写了一半"的坏 JSON。同目录内 os.replace 是原子操作。
    """
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, _REGISTRY_PATH)


def load_registry() -> dict:
    """读取全部登记记录(文件不存在/损坏则返回空)。"""
    with _lock:
        return _read_unlocked()


def save_registry(data: dict):
    """把整本注册表原子写回文件(自动建目录、格式化缩进方便人看)。"""
    with _lock:
        _write_unlocked(data)


def register(game: str, path: str, algorithm: str = "dqn",
             hidden_dims=None, eval_score=None, episodes=None, run_id=None):
    """登记一个模型。返回该模型的唯一 key(供后续切换用)。

    参数(每个影响什么):
      game       : 所属游戏(snake / othello……),界面按游戏筛选
      path       : 模型文件相对项目根的路径(如 models/snake.pt)
      algorithm  : 训练算法(DQN / PPO……),让界面知道模型"出身"
      hidden_dims: 网络隐藏层结构,展示用
      eval_score : 评估分,对比模型强弱的核心指标
      episodes   : 训练局数
      run_id     : 对应 MLflow 实验 run 的 id(实验可追溯,可空)
    """
    # 注意:key 必须唯一。时间戳只到秒的话,同一秒内登记两个模型会互相覆盖,
    # 所以把微秒也拼进来(如 snake-20260827-143012-345678)。
    key = f"{game}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    entry = {
        "game": game,
        "algorithm": algorithm,
        "path": path,
        "hidden_dims": hidden_dims or [],
        "eval_score": eval_score,
        "episodes": episodes,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="microseconds"),
    }
    # 读-改-写必须在同一把锁内完成,否则并发登记会互相覆盖丢记录。
    # 注意:threading.Lock 只保护本进程;跨进程(训练进程 + Web 进程)
    # 的并发写靠"原子替换 + 整本重写"把损坏概率降到最低。
    with _lock:
        data = _read_unlocked()
        data[key] = entry
        _write_unlocked(data)
    return key


def list_models(game: str | None = None) -> list[dict]:
    """列出模型(可只列某个游戏的),按训练时间从新到旧排序。"""
    data = load_registry()
    items = [
        {"key": k, **v}
        for k, v in data.items()
        if game is None or v.get("game") == game
    ]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def resolve(path_or_key: str) -> str | None:
    """把"模型 key 或路径"解析成真实文件路径。

    界面传 key(如 snake-xxx)时查注册表;
    也兼容直接传路径(如 models/snake.pt),便于命令行/脚本直接指定。

    安全:路径必须落在项目根目录内——防止 "../" 之类的路径穿越
    把任意文件喂给 torch.load(name 来自用户请求,不可信)。
    """
    root = _REGISTRY_PATH.parent.parent

    def _safe(p: Path) -> str | None:
        try:
            p.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            return None                    # 跑出项目根外 → 拒绝
        return str(p) if p.exists() else None

    # 1) 先按 key 查注册表
    entry = load_registry().get(path_or_key)
    if entry:
        # 条目里存的路径若已不存在(比如文件被删/测试临时文件),返回 None,
        # 让上层走 404,而不是 torch.load 报一堆底层错误。
        return _safe(root / entry["path"])
    # 2) 否则当作相对路径
    return _safe(root / path_or_key)
