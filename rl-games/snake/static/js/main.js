/* main.js —— 贪吃蛇驾驶舱入口:对局流程 + 事件绑定 + 启动。
   渲染画布见 board.js;共享工具见 /shared/frontend/common.js。 */

import { state, $ } from "./state.js";
import {
  renderAll, renderDecision, renderGame, renderHud,
  renderObs, setActionNames, setObsLabels,
} from "./board.js";
import { api, drawCurve, renderConfig, renderModelPicker } from "/shared/frontend/common.js";

/* ---------------- 训练曲线(缓存,resize 时重绘) ---------------- */
const curveCanvas = $("curve-canvas");
let curveDataCache = { episodes: [], scores: [] };

function curve(data) {
  curveDataCache = data;
  drawCurve(curveCanvas, data, {
    emptyText: "暂无训练数据 —— 运行 python snake/train.py 生成曲线",
    footerText: `局数(共 ${data.episodes.length} 次评估,最大 ${Math.max(...data.episodes, 0)})`,
    showAvg: true,
    showMax: true,               // 贪吃蛇额外画"历史最高"线
  });
}

/* ---------------- 游戏流程 ---------------- */
async function doStep(action, isAi) {
  if (state.gameOver || state.busy) return;
  state.busy = true;  // 加锁:请求返回前忽略新的触发,防止连按/并发多走
  try {
    const body = isAi ? { ai: true } : { action };
    const res = await api("/api/step", { method: "POST", body: JSON.stringify(body) });
    state.last = res;
    state.snapshot = { state: res.state, obs: res.pre_obs, q_values: res.q_values };
    state.gameOver = res.done;
    if (res.state?.direction !== undefined) state.dir = res.state.direction; // 同步服务端最终方向
    renderAll(res);
    if (res.done) onGameOver(res);
  } catch (e) {
    console.error("step 失败:", e);
  } finally {
    state.busy = false;
  }
}

let gameGen = 0;  // 局代际号:切模式/手动重开时 +1,让旧的自动重开定时器失效

function onGameOver(res) {
  const titles = { crash_wall: "撞墙了!😵", crash_body: "撞到自己了!😵", timeout: "绕圈超时了!🥱" };
  $("overlay-title").textContent = titles[res.reason] || "本局结束";
  $("overlay-sub").textContent = `得分 ${res.state.score} · 步数 ${res.state.steps} · ε ${res.epsilon.toFixed(3)}`;
  $("game-overlay").hidden = false;
  if (state.mode === "ai") {
    // AI 演示:短暂展示后自动重开,形成"连续表演"
    const gen = gameGen;
    setTimeout(() => { if (gen === gameGen && state.mode === "ai") resetGame(); }, 1800);
  }
}

async function resetGame() {
  gameGen++;  // 使旧的自动重开定时器失效
  // 置 busy 挡住主循环:避免 reset 在途时又发出 step,旧 step 的响应
  // 晚到会覆盖 reset 后的新局面。
  state.busy = true;
  try {
    const res = await api("/api/reset", { method: "POST" });
    state.snapshot = res;
    state.last = { q_values: res.q_values, action: -1, reward: 0,
                   epsilon: res.epsilon, state: res.state };
    state.gameOver = false;
    state.dir = res.state.direction ?? 3;  // 同步初始方向(默认向右)
    state.dirQueue = [];                   // 清空方向队列
    $("game-overlay").hidden = true;
    $("chip-action").textContent = "待决策";
    renderGame(); renderObs(res.obs); renderHud();
  } finally {
    state.busy = false;
  }
}

/* ---------------- 模型切换(平台化的"换模型") ---------------- */
async function switchModel(key) {
  if (!key) return;
  try {
    gameGen++;  // 使旧的自动重开定时器失效,换模型后不触发意外重开
    // 切换后返回最新快照(状态 + 观察 + Q 值 + 模型信息),直接刷新
    const res = await api("/api/models/load", { method: "POST", body: JSON.stringify({ name: key }) });
    state.snapshot = res;
    state.last = { q_values: res.q_values, action: -1, reward: 0,
                   epsilon: res.epsilon, state: res.state };
    state.gameOver = false;
    state.dir = res.state.direction ?? 3;  // 同步方向,蛇眼朝向/人玩首发方向才正确
    state.dirQueue = [];
    $("game-overlay").hidden = true;
    $("model-name").textContent = res.model.loaded ? `模型 ${res.model.name}` : "随机初始模型";
    $("badge-model").classList.toggle("fresh", !res.model.loaded);
    $("eps-val").textContent = res.epsilon.toFixed(3);
    renderGame(); renderObs(res.obs); renderHud(); renderDecision(state.last);
  } catch (e) {
    console.error("切换模型失败:", e);
  }
}

/* ---------------- 主循环(AI/人玩 自动走) ---------------- */
const BASE_MS = 500;          // 速度×1 时每步间隔(毫秒)≈ 2 步/秒
let acc = 0, lastTime = 0;

function loop(now) {
  if (!lastTime) lastTime = now;
  acc += now - lastTime;
  lastTime = now;
  const interval = BASE_MS / state.speed;
  // 每帧最多走一步;请求在途(busy)时跳过本拍,不追帧,避免连跳
  if (!state.paused && !state.gameOver && !state.busy && acc >= interval) {
    acc = 0;
    if (state.mode === "ai") {
      doStep(null, true);
    } else {
      // 人玩:从方向队列取一个;没按过键就保持当前方向直行
      const d = state.dirQueue.length ? state.dirQueue.shift() : state.dir;
      doStep(d, false);
    }
  }
  acc = Math.min(acc, interval);  // 封顶,防止积压
  requestAnimationFrame(loop);
}

/* ---------------- 键盘:人玩模式控制方向 ---------------- */
const KEYMAP = {
  ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3,
  w: 0, s: 1, a: 2, d: 3, W: 0, S: 1, A: 2, D: 3,
};

/* ---------------- 启动 ---------------- */
async function init() {
  // 确保遮罩层隐藏(防 CSS 缓存问题)
  $("game-overlay").hidden = true;
  try {
    const [snap, cfg, model, curveData, meta] = await Promise.all([
      api("/api/state"), api("/api/config"), api("/api/model"), api("/api/curve"), api("/api/meta"),
    ]);

    // 元数据驱动:用服务端自报家门的动作名 / 观察含义覆盖默认值
    if (meta.actions?.names?.length) setActionNames(meta.actions.names);
    if (meta.obs?.meaning?.length) setObsLabels(meta.obs.meaning, meta.obs.group);

    state.snapshot = snap;
    state.gridSize = snap.state.grid_size;
    state.last = { q_values: snap.q_values, action: -1, reward: 0,
                   epsilon: snap.epsilon, state: snap.state };

    $("model-name").textContent = model.loaded ? `模型 ${model.name}` : "随机初始模型";
    $("badge-model").classList.toggle("fresh", !model.loaded);
    $("eps-val").textContent = snap.epsilon.toFixed(3);

    renderConfig($("config-grid"), [
      ["学习率 lr", cfg.lr], ["折扣 γ", cfg.gamma],
      ["batch 大小", cfg.batch_size], ["经验池", cfg.buffer_capacity],
      ["目标网同步", cfg.target_update_freq], ["探索 ε 起点", cfg.epsilon_start],
      ["探索 ε 终点", cfg.epsilon_end], ["ε 衰减", cfg.epsilon_decay],
      ["棋盘", `${cfg.grid_size}×${cfg.grid_size}`],
      ["网络结构", `${cfg.input_dim}→${cfg.hidden_dims.join("→")}→${cfg.n_actions}`],
    ]);

    curve(curveData);
    renderModelPicker($("model-select"), meta.models || [], model.name);
    renderGame(); renderObs(snap.obs); renderHud();

    requestAnimationFrame(loop);
  } catch (e) {
    console.error("初始化失败:", e);
    document.body.insertAdjacentHTML("beforeend",
      `<div style="position:fixed;inset:0;display:grid;place-items:center;background:#0c0e16;color:#f87171;font-size:15px">无法连接后端服务,请先运行 python snake/serve.py</div>`);
  }
}

/* ---------------- 事件绑定 ---------------- */
// 模式切换
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.mode;
    $("game-chip").textContent = state.mode === "ai" ? "AI 演示中" : "人玩 · 蛇自动走,方向键控方向";
    // 暂停两种模式都可用:AI=暂停演示,人玩=蛇停下
    if (state.mode === "human") { state.paused = false; $("btn-pause").textContent = "暂停"; }
    resetGame();
  });
});

// 速度控制
document.querySelectorAll(".ctl-btn[data-speed]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".ctl-btn[data-speed]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.speed = parseFloat(btn.dataset.speed);
    acc = 0;
  });
});

// 暂停 / 单步
$("btn-pause").addEventListener("click", () => {
  state.paused = !state.paused;
  $("btn-pause").textContent = state.paused ? "继续" : "暂停";
});
$("btn-step").addEventListener("click", () => {
  // 单步:AI 模式让模型走一步;人玩模式按队列/当前方向走一步
  if (state.mode === "ai") doStep(null, true);
  else doStep(state.dirQueue.length ? state.dirQueue.shift() : state.dir, false);
});
$("btn-restart").addEventListener("click", () => resetGame());

window.addEventListener("keydown", (e) => {
  if (e.key in KEYMAP && state.mode === "human") {
    e.preventDefault();
    if (e.repeat) return;  // 按住不放的自动重复触发,忽略
    if (e.ctrlKey || e.metaKey || e.altKey) return;  // 快捷键组合不拦截
    if (state.gameOver) return;                      // 结束后不再接收输入
    const d = KEYMAP[e.key];
    // 入队前做去重:和"队列末尾(或当前方向)"相同就不重复入队;
    // 队列最多 2 个,够容纳一个 tick 内的连按(如"上+左"两连转)。
    const lastDir = state.dirQueue.length
      ? state.dirQueue[state.dirQueue.length - 1] : state.dir;
    if (d === lastDir || state.dirQueue.length >= 2) return;
    state.dirQueue.push(d);
  }
});

// 模型切换下拉框
$("model-select").addEventListener("change", (e) => switchModel(e.target.value));

/* Tab 切换:模型决策 / 训练进度 */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    if (tab.classList.contains("active")) return;
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach((p) => p.hidden = true);
    const panel = document.getElementById(tab.dataset.tab);
    panel.hidden = false;
    // 训练曲线在 hidden 时 clientWidth=0,切换回来要重绘
    if (tab.dataset.tab === "panel-training") curve(curveDataCache);
  });
});

window.addEventListener("resize", () => curve(curveDataCache));

init();
