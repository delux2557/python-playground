"""othello/train.py —— 黑白棋自对弈训练入口(同时是被测试复用的核心逻辑)。

用法:
    python othello/train.py --episodes 2000 \
        --checkpoint models/othello.pt --curve data/othello_curve.json

和贪吃蛇 train.py 的差异(核心是"自对弈"):
  · 不是"模型 vs 环境",而是"模型 vs 对手池"(见 selfplay.py)
  · 评估指标从"每局得分"换成"对战纯随机的胜率"
  · 每局可能没有任何奖励信号,由自对弈循环在终局统一回传
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# 项目根目录(用于把模型路径存成相对路径,注册表里统一格式)
# 并插入 sys.path:直接运行 "python othello/train.py" 时,Python 只把
# 脚本所在目录(othello/)放进 sys.path[0],项目根不在——没有这一句,
# 下面的 `from othello.dqn import ...` 会报 ModuleNotFoundError
# (与 snake/train.py、serve.py 的处理保持一致)。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from othello.dqn import OthelloDQNConfig  # noqa: E402
from othello.selfplay import SelfPlayTrainer, evaluate_win_rate  # noqa: E402
from shared.checkpoint import load_for_resume, save_with_meta
from shared.experiment import track_training  # noqa: E402
from shared.registry import register  # noqa: E402

# 训练进度文件(训练/Web 解耦的核心:训练进程只写文件,Web 只读文件)
_PROGRESS_PATH = _ROOT / "data" / "othello_progress.json"
_CURVE_PATH = _ROOT / "data" / "othello_curve.json"


def set_seed(seed):
    """固定所有随机源,保证训练可复现(最佳实践)。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _atomic_write_json(path: Path, obj: dict):
    """原子写 JSON:先写临时文件再改名替换。

    为什么必须原子:Web 端可能随时在读这个文件,直接覆盖会读到
    "写了一半"的坏 JSON。先写 .tmp 再 os.replace(同目录内是原子操作),
    保证 Web 读到的永远是完整内容。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


class ProgressReporter:
    """把训练进度写到 data/*.json,让 Web 端实时可见(训练/Web 解耦)。

    训练进程和 Web 进程之间不共享内存,只通过两份文件通信:
      progress.json : 状态(还没跑/训练中/完成/出错)+ 当前局数 + 胜率
      curve.json    : 胜率曲线(每次评估追加一个点,Web 端曲线跟着动)

    Web 端由 serve.py 的 /api/train/status 读 progress.json,
    由 /api/curve 读 curve.json。训练跑完(或出错)后文件里会写最终状态。
    """

    def __init__(self, progress_path: Path, curve_path: Path,
                 game: str = "othello", initial_curve: dict | None = None):
        self.progress_path = Path(progress_path)
        self.curve_path = Path(curve_path)
        self.game = game
        self._total_episodes = 0
        self._started_at = _now()
        self._last_epsilon = None      # 最近一次评估的 ε(finish 写真实终值)
        self._last_pool = 0            # 最近一次评估的对手池大小
        # 曲线记录评估点;续训(initial_curve)时保留旧点,新点往后接,Web 曲线连续
        self._curve = {"episodes": [], "win_rates": []}
        if initial_curve:
            self._curve["episodes"] = list(initial_curve.get("episodes", []))
            self._curve["win_rates"] = list(initial_curve.get("win_rates", []))
        self._write_progress({
            "status": "starting",
            "message": "正在初始化…",
            "episode": 0, "episodes": 0,
            "win_rate": None, "epsilon": None, "opponent_pool": 0,
            "started_at": self._started_at, "updated_at": _now(),
        })

    # ---------------- 对外接口 ----------------
    def start(self, total_episodes: int):
        """训练开始:记录总局数,状态置为 running。"""
        self._total_episodes = total_episodes
        self._write_progress({
            "status": "running",
            "message": f"训练中 · 目标 {total_episodes} 局",
            "episode": 0, "episodes": total_episodes,
            "win_rate": None, "epsilon": None, "opponent_pool": 0,
            "started_at": self._started_at, "updated_at": _now(),
        })

    def on_eval(self, info: dict):
        """每次评估后:更新进度 + 往曲线追加一个点。"""
        self._curve["episodes"].append(info["episode"])
        self._curve["win_rates"].append(round(float(info["win_rate"]), 4))
        _atomic_write_json(self.curve_path, self._curve)   # 曲线增量可见
        # 记住最近一次的 ε/对手池:训练结束时写真实终值用
        self._last_epsilon = round(float(info["epsilon"]), 4)
        self._last_pool = info["opponent_pool"]
        self._write_progress({
            "status": "running",
            "message": f"训练中 · 第 {info['episode']} 局",
            "episode": info["episode"], "episodes": self._total_episodes,
            "win_rate": round(float(info["win_rate"]), 4),
            "epsilon": round(float(info["epsilon"]), 4),
            "opponent_pool": info["opponent_pool"],
            "started_at": self._started_at, "updated_at": _now(),
        })

    def finish(self, final_win_rate: float, epsilon: float | None = None,
               opponent_pool: int | None = None):
        """训练结束:状态置为 done,写入最终胜率。

        epsilon/opponent_pool 传真实终值(不传则保留最近一次评估的值),
        不再硬写 0——否则训练面板展示的最后 ε/对手池失真。
        """
        self._write_progress({
            "status": "done",
            "message": f"训练完成 · 对随机最终胜率 {final_win_rate:.2f}",
            "episode": self._curve_episodes, "episodes": self._total_episodes,
            "win_rate": round(float(final_win_rate), 4),
            "epsilon": epsilon, "opponent_pool": opponent_pool,
            "started_at": self._started_at, "updated_at": _now(),
        })

    def fail(self, message: str):
        """训练异常退出:状态置为 error,方便 Web 端提示。"""
        self._write_progress({
            "status": "error",
            "message": f"训练出错: {message}",
            "episode": self._curve_episodes, "episodes": self._total_episodes,
            "win_rate": None, "epsilon": None, "opponent_pool": 0,
            "started_at": self._started_at, "updated_at": _now(),
        })

    # ---------------- 内部工具 ----------------
    @property
    def _curve_episodes(self) -> int:
        return self._curve["episodes"][-1] if self._curve["episodes"] else 0

    @property
    def curve(self) -> dict:
        """累积曲线(含续训保留的历史点)。写曲线文件用这个,
        而不是训练循环本次返回的点——否则续训时会把历史点覆盖掉。"""
        return {"episodes": list(self._curve["episodes"]),
                "win_rates": list(self._curve["win_rates"])}

    def _write_progress(self, data: dict):
        _atomic_write_json(self.progress_path, {**data, "game": self.game})


def _now() -> str:
    """本地时间的 ISO 字符串(带秒,方便前端展示)。"""
    return datetime.now().isoformat(timespec="seconds")


def run_training(cfg, checkpoint_path=None, curve_path=None,
                 progress_path=None, verbose=True):
    """核心训练函数(被命令行和测试共用)。

    参数:
      cfg            : OthelloDQNConfig(超参数都在这)
      checkpoint_path: 保存模型的路径(结尾 .pt)
      curve_path     : 保存胜率曲线的 JSON 路径
      progress_path  : 保存实时训练进度的 JSON 路径(不传则和曲线同目录)
    返回:
      {"final_win_rate": 最终对随机胜率, "curve": {"episodes", "win_rates"}}
    """
    set_seed(cfg.seed)
    curve_path = Path(curve_path) if curve_path else _CURVE_PATH
    if progress_path is None:
        progress_path = curve_path.with_name("othello_progress.json")
    progress_path = Path(progress_path)

    # ---- 断点续训:检测已有 checkpoint,接着上次的局数往下训 ----
    resume_agent, start_ep, resume_eps = (
        load_for_resume(checkpoint_path) if checkpoint_path else (None, 0, None))
    if resume_agent is not None and resume_eps is not None:
        resume_agent.epsilon = resume_eps   # 恢复探索率,从上次位置继续衰减
    if resume_agent is not None and verbose:
        print(f"[续训] 检测到 checkpoint,从第 {start_ep} 局续训到 "
              f"{cfg.episodes} 局(ε={resume_agent.epsilon:.3f})")

    # 续训时保留旧曲线,新评估点往后接(Web 端曲线连续)
    initial_curve = None
    if resume_agent is not None and curve_path.exists():
        try:
            initial_curve = json.loads(curve_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            initial_curve = None

    reporter = ProgressReporter(progress_path, curve_path,
                                initial_curve=initial_curve)
    reporter.start(cfg.episodes)

    trainer = SelfPlayTrainer(cfg)
    if resume_agent is not None:
        trainer.agent = resume_agent        # 用 checkpoint 的权重继续训练
        trainer.agent.epsilon = resume_agent.epsilon
    try:
        # 训练主循环:每次评估把进度喂给 reporter(写进度 + 增量曲线);
        # 同时定期存档 checkpoint,崩溃后最多丢 eval_freq 局。
        def _progress_cb(info):
            reporter.on_eval(info)
            if checkpoint_path:
                save_with_meta(trainer.agent, checkpoint_path,
                               {"episode": info["episode"],
                                "epsilon": info["epsilon"]})

        curve = trainer.run(verbose=False, progress_cb=_progress_cb,
                            start_ep=start_ep)

        # 最终评估:换一批随机种子,测模型真实水平
        final_wr = evaluate_win_rate(trainer.agent, games=cfg.eval_games,
                                     seed=cfg.seed + 999_999)

        # 保存模型和曲线
        run_id = None
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            save_with_meta(trainer.agent, checkpoint_path,
                           {"episode": cfg.episodes,
                            "epsilon": trainer.agent.epsilon})
            # 实验跟踪:把本次训练的超参/胜率曲线/模型档案记进 MLflow
            run_id = track_training(
                game="othello",
                experiment="othello-selfplay",
                params=dict(cfg.__dict__),
                curves={"win_rate": (reporter.curve["episodes"],
                                     reporter.curve["win_rates"])},
                checkpoint_path=checkpoint_path,
                final_metrics={"final_win_rate": round(final_wr, 4)},
            )
            # 登记到模型注册表:前端 /api/models 就能列出并切换这个模型
            model_rel = Path(checkpoint_path)
            if model_rel.is_absolute():
                try:
                    model_rel = model_rel.relative_to(_ROOT)
                except ValueError:
                    pass
            register(
                game="othello",
                path=str(model_rel),
                algorithm="dqn-selfplay",
                hidden_dims=list(cfg.hidden_dims),
                eval_score=round(final_wr, 3),
                episodes=cfg.episodes,
                run_id=run_id,
            )
        if curve_path:
            # 写累积曲线(续训时含历史点),而不是 trainer 本次返回的点——
            # 否则续训会把第一段的曲线点覆盖掉,Web 端曲线就不连续了。
            _atomic_write_json(curve_path, reporter.curve)

        # 写真实终值:ε 停在 epsilon_end 附近、对手池为实际大小
        reporter.finish(final_wr, epsilon=trainer.agent.epsilon,
                        opponent_pool=len(trainer.opponent_pool))
    except Exception as e:
        # 训练循环之后的任何一步(评估/保存/登记)失败也要把进度置为
        # error——否则 progress.json 永远停在 "running",Web 端会
        # 一直显示"训练中"。
        reporter.fail(str(e))
        raise

    if verbose:
        print(f"完成!对随机最终胜率 = {final_wr:.2f} "
              f"→ 模型已保存到 {checkpoint_path}")

    return {"final_win_rate": final_wr, "curve": curve}


def main():
    parser = argparse.ArgumentParser(description="黑白棋自对弈 DQN 训练")
    parser.add_argument("--episodes", type=int, default=OthelloDQNConfig.episodes)
    parser.add_argument("--checkpoint", default="models/othello.pt")
    parser.add_argument("--curve", default=str(_CURVE_PATH))
    parser.add_argument("--progress", default=None,
                        help="实时进度 JSON 路径(默认与 --curve 同目录)")
    parser.add_argument("--seed", type=int, default=OthelloDQNConfig.seed)
    parser.add_argument("--eval-freq", type=int, default=OthelloDQNConfig.eval_freq)
    parser.add_argument("--eval-games", type=int, default=OthelloDQNConfig.eval_games)
    args = parser.parse_args()

    cfg = OthelloDQNConfig(episodes=args.episodes, seed=args.seed,
                           eval_freq=args.eval_freq, eval_games=args.eval_games)
    run_training(cfg, checkpoint_path=args.checkpoint, curve_path=args.curve,
                 progress_path=args.progress)


if __name__ == "__main__":
    main()
