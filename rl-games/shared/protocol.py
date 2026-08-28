"""shared/protocol.py —— 统一 Agent 服务协议(轻量平台化的核心)。

目标:让"任意游戏"的 Web 服务都实现同一个接口,前端只需读一份
"元数据"就能渲染任何游戏的驾驶舱。这样:

  · 换游戏     → 新写一个 AgentService 实现,协议层零改动
  · 换算法     → 服务内部换 agent,前端零改动
  · 换模型     → 通过 /api/models/load 从注册表切换

三层契约:
  1. StepRequest   : 前端"走一步"请求的统一格式(ai 或 action)
  2. AgentService  : 抽象基类。每个游戏只实现"游戏特有"的 6 个方法,
                     模型注册等通用部分基类已写好
  3. build_app()   : 把任意 AgentService 包装成一套标准 FastAPI 路由,
                     所有游戏共用同一套 URL 和返回结构

标准路由(所有游戏一致):
  GET  /api/meta            元数据(棋盘/动作/观察含义) + 已登记模型
  GET  /api/state           当前局面 + 模型看到的一切(观察 + Q 值 + ε)
  POST /api/reset           开局
  POST /api/step            走一步 {"ai":true} 或 {"action":0~3}
  GET  /api/curve           训练曲线
  GET  /api/config          训练超参数
  GET  /api/train/status    训练进程实时状态(训练/Web 解耦的观察入口)
  GET  /api/model           当前加载的模型信息
  GET  /api/models          模型注册表列表
  POST /api/models/load     切换模型 {"name": "模型key或路径"}

会话隔离:
  state/reset/step 三个"对局路由"会读取请求头 X-Session-Id,并把它传给
  服务方法。每个浏览器标签页用自己的会话 ID → 玩各自独立的一局,
  多个页面同时打开也不会互相踩(模型/训练曲线等仍是全局共享)。
"""

from abc import ABC, abstractmethod
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared.registry import list_models, resolve

# 共享前端层(所有游戏驾驶舱共用的 common.js / common.css)
_SHARED_STATIC = Path(__file__).resolve().parent / "static"


class StepRequest(BaseModel):
    """前端"走一步"请求。二选一:
      ai: True      让模型决策
      action: 0~3   人类指定动作
    """
    ai: bool = False
    action: int | None = None


class ModelLoadRequest(BaseModel):
    """前端"切换模型"请求。name 可以是注册表里的 key,也可以是路径。"""
    name: str


class AgentService(ABC):
    """每个游戏的服务基类。子类只需实现下面 6 个"游戏特有"方法,
    模型注册表、通用路由都由协议层负责。
    """

    game_name: str = "unknown"   # 子类覆盖:snake / othello……

    # ---------------- 子类必须实现 ----------------
    @abstractmethod
    def meta(self) -> dict:
        """自报家门:棋盘怎么画、动作有哪些、观察每维什么意思。
        返回结构见 snake/serve.py 里的实现示例。"""

    @abstractmethod
    def snapshot(self, session: str | None = None) -> dict:
        """当前局面 + 模型看到的一切(观察向量 + Q 值 + ε + 模型信息)。
        session: 会话 ID,用于区分不同浏览器标签页各自的对局。"""

    @abstractmethod
    def reset(self, session: str | None = None) -> dict:
        """开局,返回和 snapshot() 同结构的数据。"""

    @abstractmethod
    def step(self, req: StepRequest, session: str | None = None) -> dict:
        """走一步,返回"这次决策的完整记录"(观察/Q值/动作/奖励/是否结束)。"""

    @abstractmethod
    def curve(self) -> dict:
        """训练曲线 {"episodes": [...], "scores": [...]}。"""

    @abstractmethod
    def config(self) -> dict:
        """训练超参数字典(前端展示用)。"""

    def train_status(self) -> dict:
        """训练进程的实时状态(训练/Web 解耦的可视化入口)。

        默认实现:没有训练信息(该游戏还没接训练进度上报)。
        游戏可覆盖:去读自己训练进程写的 progress.json,返回
          {"status": "idle|starting|running|done|error",
           "running": bool, "episode": 局数, "episodes": 总局数,
           "win_rate": 最近胜率, "epsilon": ..., "message": 文字说明}
        这样"训练在独立进程跑、Web 只读状态"就成了平台能力。
        """
        return {"status": "idle", "running": False,
                "message": "该游戏未接入训练进度上报",
                "episode": 0, "episodes": 0,
                "win_rate": None, "epsilon": None}

    @abstractmethod
    def _load_weights(self, path: str) -> dict:
        """把 checkpoint 文件加载进自己的 agent。返回模型信息 dict。
        这是"换模型"的唯一入口:子类实现如何把 .pt 塞进自己的网络。"""

    # ---------------- 基类已实现(子类不用动) ----------------
    def models(self) -> list[dict]:
        """当前游戏在注册表里登记过的模型(按时间从新到旧)。"""
        return list_models(self.game_name)

    def load_model(self, name: str, session: str | None = None) -> dict:
        """按 key 或路径切换模型。成功后会返回最新快照,前端可立即刷新。

        session: 透传给 snapshot()——返回"发起请求的那个会话"的局面,
        而不是默认会话的(模型是全局共享的,但局面按会话隔离)。
        """
        path = resolve(name)
        if not path:
            raise HTTPException(404, f"找不到模型: {name}")
        info = self._load_weights(path)
        # 注意展开顺序:快照自带 "model" 键,这里用 _load_weights 返回的
        # info 覆盖它,保证返回的是本次加载的权威模型信息。
        return {**self.snapshot(session), "loaded": True, "model": info}


def build_app(service: AgentService, static_dir: Path) -> FastAPI:
    """把任意 AgentService 包装成一套标准 FastAPI 应用。

    static_dir: 该游戏前端页面(HTML/CSS/JS)所在目录。
    """
    app = FastAPI(title=f"{service.game_name} · RL 驾驶舱")
    # 允许跨域:方便前端开发时用别的端口直接调试
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    # 前端静态资源(每个游戏自己的 static/ 目录)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    # 共享前端层:common.js / common.css 挂在 /shared/frontend/...,
    # 供所有游戏的 <script type="module"> import 引用(消除跨游戏重复代码)
    app.mount("/shared", StaticFiles(directory=_SHARED_STATIC), name="shared")

    # ---------------- 路由:所有游戏完全一致 ----------------
    @app.get("/api/meta")
    def api_meta():
        """元数据 + 已登记模型,前端一进来先读它。"""
        return {**service.meta(), "models": service.models()}

    @app.get("/api/state")
    def api_state(x_session_id: str | None = Header(default=None)):
        """当前局面 + 模型"看到"的一切。"""
        return service.snapshot(session=x_session_id)

    @app.get("/api/model")
    def api_model():
        """当前加载的模型信息。"""
        return service.meta()["model"]

    @app.get("/api/models")
    def api_models():
        """模型注册表列表。"""
        return service.models()

    @app.post("/api/models/load")
    def api_model_load(req: ModelLoadRequest,
                       x_session_id: str | None = Header(default=None)):
        """切换模型(前端下拉框选一个)。"""
        return service.load_model(req.name, session=x_session_id)

    @app.post("/api/reset")
    def api_reset(x_session_id: str | None = Header(default=None)):
        """重新开局。"""
        return service.reset(session=x_session_id)

    @app.post("/api/step")
    def api_step(req: StepRequest, x_session_id: str | None = Header(default=None)):
        """走一步:模型决策(ai)或人类输入(action)。"""
        return service.step(req, session=x_session_id)

    @app.get("/api/curve")
    def api_curve():
        """训练曲线。"""
        return service.curve()

    @app.get("/api/config")
    def api_config():
        """训练超参数。"""
        return service.config()

    @app.get("/api/train/status")
    def api_train_status():
        """训练进程实时状态(训练/Web 解耦:Web 只读,训练进程只写)。"""
        return service.train_status()

    @app.get("/")
    def index():
        """浏览器访问根路径 → 返回前端页面。"""
        return FileResponse(static_dir / "index.html")

    return app
