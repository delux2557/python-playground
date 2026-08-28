"""shared/eval.py —— 统一评测基准(让"谁更强"用同一把尺子量)。

为什么要"统一评测基准":
  训练时每局记录一次 eval_score,但那是训练自己跑的、参数各不同,
  两个模型直接比 eval_score 不算公平。统一评测 = 把任何一个模型
  "当场重测",用完全相同的数据分布(固定参照物、固定局数、黑白各半)
  重新打分——这样分高的就是真的强,和训练时间、训练配置无关。

本模块是一个"可插拔的评测中心":
  · 每个游戏注册一个"怎么给模型打分"的评测器(register_evaluator)
  · evaluate(game, path)   : 任意模型 → 统一基准下的分数
  · benchmark(game)        : 把注册表里该游戏所有模型一次性重测,
                             生成排行榜(data/<game>_benchmark.json),
                             并同步刷新注册表里的 eval_score
  · 训练脚本不再独占"评分权",Web 端可随时发起重测
"""

import json
import os
import threading
from pathlib import Path

from shared.registry import list_models, load_registry, resolve, save_registry

_ROOT = Path(__file__).resolve().parent.parent

# 每个游戏一个评测器:game 名 -> 函数 fn(path) -> {"score": 分数, "detail": 说明}
EVALUATORS = {}
_lock = threading.Lock()
# 同一时间只允许一场 benchmark:评测是长耗时同步操作,并发跑会互相
# 践踏全局随机序列(破坏"统一基准"),还会竞写排行榜与注册表。
_benchmark_lock = threading.Lock()


def register_evaluator(game: str):
    """装饰器:把某游戏的评测函数注册进来。"""

    def deco(fn):
        EVALUATORS[game] = fn
        return fn

    return deco


def evaluate(game: str, path: str, **kwargs) -> dict:
    """对任意模型文件跑该游戏的统一评测,返回 {"score", "detail", ...}。

    score 恒为"越高越好"(胜率或平均分),所有模型用同一套默认参数。
    """
    if game not in EVALUATORS:
        return {"score": None, "detail": f"游戏 {game} 未注册评测器"}
    try:
        return EVALUATORS[game](path, **kwargs)
    except Exception as e:                       # 坏模型/缺文件不让整场崩掉
        return {"score": None, "detail": f"评测失败: {e}"}


def benchmark(game: str, games: int | None = None,
              out_path: Path | None = None) -> list[dict]:
    """把注册表里该游戏所有模型,在统一基准下重新打分 → 排行榜。

    返回按 score 从高到低排序的列表,并把结果持久化到
    data/<game>_benchmark.json,同时刷新注册表里的 eval_score
    (训练时的临时分被"统一分"覆盖,保证可比)。

    参数:
      games   : 覆盖默认评测局数(训练脚本/命令行可传大一点,更准)
      out_path: 排行榜保存路径(默认 data/<game>_benchmark.json)

    并发安全:同一时间只跑一场(拿不到 _benchmark_lock 直接抛
    RuntimeError,由路由层转成 400)。
    """
    if game not in EVALUATORS:
        return []
    if not _benchmark_lock.acquire(blocking=False):
        raise RuntimeError("已有一场统一评测在进行中")
    try:
        models = [m for m in list_models(game)]
        rows = []
        for m in models:
            # 用 resolve() 拿绝对路径:注册表里存的是相对项目根的路径,
            # 直接交给评测器会按"当前工作目录"解析,服务不从项目根
            # 启动就全部找不到文件。
            path = resolve(m["key"])
            score = evaluate(game, path, games=games) if games else \
                evaluate(game, path)
            rows.append({
                "key": m["key"], "name": m["key"],
                "path": m["path"], "eval_score": m.get("eval_score"),
                "score": score["score"], "detail": score.get("detail"),
                "created_at": m.get("created_at"),
            })
        rows.sort(key=lambda r: (r["score"] is not None, r["score"]), reverse=True)

        # 同步刷新注册表:统一分覆盖旧分,排行榜永远可比
        with _lock:
            data = load_registry()
            for r in rows:
                if r["score"] is not None and r["key"] in data:
                    data[r["key"]]["eval_score"] = round(r["score"], 4)
            save_registry(data)

        # 持久化排行榜(Web 端重启后还能看到上次的打榜结果)。
        # 原子写:GET /api/benchmark 随时可能并发读,不能读到半截 JSON。
        out = out_path or (_ROOT / "data" / f"{game}_benchmark.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps({"game": game, "rows": rows},
                                  ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, out)
        return rows
    finally:
        _benchmark_lock.release()


def load_benchmark(game: str) -> dict:
    """读取上次持久化的排行榜;没有(或文件损坏)则返回空结构(前端友好)。"""
    path = _ROOT / "data" / f"{game}_benchmark.json"
    if not path.exists():
        return {"game": game, "rows": [], "saved_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"game": game, "rows": [], "saved_at": None}
    return {**data, "saved_at": path.stat().st_mtime}
