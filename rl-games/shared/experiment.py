"""shared/experiment.py —— 实验跟踪(MLflow),让每次训练留下完整档案。

行业最佳实践:RL 的产物不能只有模型文件,还得有"实验档案"——
  用了哪些超参、每次评估的指标曲线、模型 artifact 快照、时间线。
MLflow 是业界的实验跟踪事实标准(超参 + 指标 + artifact + UI 一站式)。

本模块对 MLflow 做了一层薄封装,核心设计:
  · 可选依赖:MLflow 未安装/导入失败时自动降级为"无操作"。
    训练流程完全不受影响(有 MLflow 用 MLflow,没有也不报错)。
  · 存储位置:默认 sqlite:///mlflow.db(单文件数据库,零配置);
    通过环境变量 MLFLOW_TRACKING_URI 可指向远程/容器内的 MLflow 服务。
    (不用文件存储 ./mlruns:MLflow 3.x 已把文件存储置为维护模式,
     且 sqlite 单文件更利于 Docker 里挂卷持久化。)
  · 一站式入口 track_training():训练脚本一行调用,把整场训练记录下来。

配套命令(查看实验 UI):
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
                                    # 打开浏览器看所有实验/指标/模型
"""

import os
from datetime import datetime
from pathlib import Path

# 默认用 sqlite 单文件数据库;Docker Compose 里通过环境变量指向 mlflow 服务
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(Path(__file__).resolve().parent.parent / "mlflow.db"),
)


def _mlflow():
    """惰性导入 mlflow;未安装/导入失败返回 None(跟踪自动降级)。"""
    try:
        import mlflow
        return mlflow
    except Exception:
        return None


class ExperimentTracker:
    """一次训练实验的跟踪器(薄封装 MLflow)。

    用法(推荐用下面的 track_training 一站式入口):
        tracker = ExperimentTracker("snake-dqn")
        run_id = tracker.start(run_name="...", tags={...})
        tracker.log_params({...})          # 超参数
        tracker.log_metrics({...}, step=n) # 指标(可带 step 画曲线)
        tracker.log_model("models/x.pt")   # 模型 artifact
        tracker.end()

    MLflow 不可用时所有方法静默跳过,返回 None/False——训练照跑。
    """

    def __init__(self, experiment: str, tracking_uri: str | None = None):
        self.experiment = experiment
        self.uri = tracking_uri or MLFLOW_TRACKING_URI
        self._mlflow = _mlflow()
        self._active = False
        self.run_id = None

    # ---------------- 生命周期 ----------------
    def start(self, run_name: str | None = None,
              tags: dict | None = None) -> str | None:
        """开启一次 run,返回 run_id(MLflow 不可用时返回 None)。"""
        if not self._mlflow:
            return None
        try:
            self._mlflow.set_tracking_uri(self.uri)
            self._mlflow.set_experiment(self.experiment)
            run = self._mlflow.start_run(run_name=run_name)
            self._active = True          # run 已开启:先标记,保证异常时
            self.run_id = run.info.run_id  # end() 一定会去关闭它,不泄漏
            if tags:
                self._mlflow.set_tags(tags)
            return self.run_id
        except Exception:
            self.end()                   # 尽力关闭可能已开启的 run
            return None

    def end(self):
        """结束本次 run(记录保持;不删除)。"""
        if self._active and self._mlflow:
            try:
                self._mlflow.end_run()
            except Exception:
                pass
        self._active = False

    def __enter__(self) -> "ExperimentTracker":
        self.start()
        return self

    def __exit__(self, *exc):
        self.end()

    # ---------------- 记录 ----------------
    def log_params(self, params: dict):
        """记录超参数(MLflow 里可按参数筛选/对比实验)。"""
        if self._active:
            try:
                self._mlflow.log_params(params)
            except Exception:
                pass

    def log_metrics(self, metrics: dict, step: int | None = None):
        """记录指标;传 step 时 MLflow 会画出随时间/局数的曲线。"""
        if self._active:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

    def log_curve(self, metric: str, episodes, values):
        """把一条评估曲线逐点记录(如每 N 局的得分/胜率)。"""
        for ep, v in zip(episodes, values):
            self.log_metrics({metric: round(float(v), 4)}, step=int(ep))

    def log_model(self, path: str, artifact_path: str = "model"):
        """把模型文件作为 artifact 快照进本次 run(可追溯每一版)。"""
        if not self._active:
            return
        try:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception:
            pass


def track_training(*, game: str, experiment: str, params: dict,
                   curves: dict | None = None, checkpoint_path=None,
                   final_metrics: dict | None = None,
                   tags: dict | None = None,
                   tracking_uri: str | None = None) -> str | None:
    """一站式记录一次完整训练到 MLflow(训练脚本调这一个就够了)。

    参数(含义即用途):
      game            : 所属游戏(snake / othello),写入 run 标签
      experiment      : MLflow 实验名(如 "snake-dqn"),同主题实验归一组
      params          : 超参数字典(整个 cfg 转成 dict 即可)
      curves          : {"指标名": (episodes列表, 值列表)} 逐点记成曲线
      checkpoint_path : 模型文件路径,作为 artifact 快照
      final_metrics   : 最终指标(如最终平均分 / 最终胜率)
      tags            : 额外标签
      tracking_uri    : 覆盖 MLFLOW_TRACKING_URI(一般不用传)

    返回:run_id;MLflow 不可用时返回 None(训练不受影响)。
    """
    tracker = ExperimentTracker(experiment, tracking_uri)
    run_id = tracker.start(
        run_name=f"{game}-{datetime.now():%Y%m%d-%H%M%S}",
        tags={"game": game, **(tags or {})})
    if not run_id:
        return None
    try:
        tracker.log_params(params or {})
        for name, (eps, vals) in (curves or {}).items():
            tracker.log_curve(name, eps, vals)
        if final_metrics:
            tracker.log_metrics(final_metrics)
        if checkpoint_path and Path(checkpoint_path).exists():
            tracker.log_model(str(checkpoint_path))
    finally:
        tracker.end()
    return run_id
