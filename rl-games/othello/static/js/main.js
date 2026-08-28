/* main.js —— 黑白棋驾驶舱入口:开局/AI 回合等主流程 + 事件绑定 + 启动。
   职责:
     · 对局流程:doStep / scheduleAi / resetGame / switchModel / setHumanMode
     · 训练状态面板:拉状态 + 自适应轮询
     · 把事件绑定到 board.js(渲染)/ features.js(高级功能)提供的函数
   渲染画布与高级功能面板分别见 board.js / features.js。 */

import { state, $ } from "./state.js";
import {
  cancelScheduled, renderAll, renderBoard, startMoveAnim, updateControls,
} from "./board.js";
import {
  canUndo, clearArena, clearDuel, closeReplay, doUndo, openReplay,
  populateDuelSelects, renderBenchmark, replaySeek, runBenchmark,
  startArena, startDuel, toggleReplayPlay, updateDuelSelectScores,
} from "./features.js";
import {
  api, drawCurve, renderConfig, renderModelPicker, startTrainPolling,
} from "/shared/frontend/common.js";

const N = 8;

/* ---------------- 训练曲线缓存(resize 时重绘用) ---------------- */
const curveCanvas = $("curve-canvas");
let curveDataCache = { episodes: [], scores: [] };

function redrawCurve() {
  drawCurve(curveCanvas, curveDataCache, {
    emptyText: "暂无训练数据 —— 运行 python othello/train.py 生成曲线",
    footerText: `对随机对手胜率(共 ${curveDataCache.episodes.length} 次评估,最大局数 ${Math.max(...curveDataCache.episodes, 0)})`,
    showAvg: true,
    showMax: false,
  });
}

function curve(data) {
  curveDataCache = data;
  redrawCurve();
}

/* ---------------- 训练状态:轮询(训练/Web 解耦的可视化) ---------------- */
const TRAIN_STATUS_TEXT = {
  idle: "未开始", starting: "初始化", running: "训练中",
  done: "已完成", error: "出错",
};

function renderTrain(st) {
  const badge = $("train-badge");
  badge.textContent = TRAIN_STATUS_TEXT[st.status] || st.status;
  badge.className = "chip" + (st.status === "error" ? " chip-error"
    : (st.running ? " chip-running" : " chip-muted"));

  $("train-episode").textContent = st.episodes
    ? `${st.episode} / ${st.episodes}` : "–";
  $("train-winrate").textContent = st.win_rate != null
    ? `${(st.win_rate * 100).toFixed(1)}%` : "–";
  $("train-epsilon").textContent = st.epsilon != null
    ? st.epsilon.toFixed(3) : "–";
  $("train-pool").textContent = st.opponent_pool ?? "–";

  const pct = st.episodes ? Math.min(100, Math.round(st.episode / st.episodes * 100)) : 0;
  $("train-fill").style.width = pct + "%";
  $("train-msg").textContent = st.message || "–";
}

// 自适应轮询:训练中(或刚结束)每 2 秒,空闲 15 秒——避免终生高频打请求
function pollTrain() {
  return (async () => {
    let st;
    try {
      st = await api("/api/train/status");
    } catch (e) {
      return false;
    }
    renderTrain(st);
    if (st.running || st.status === "done" || st.status === "error") {
      try { curve(await api("/api/curve")); } catch (e) { /* 忽略 */ }
    }
    return st.running || st.status === "starting";   // true = 保持高频
  })();
}

/* ---------------- 对战模式(人执黑/白/纯 AI) ---------------- */
function setHumanMode(btn) {
  const human = btn.dataset.human;
  state.human = human;
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  // 纯 AI 对战:自动连下
  $("btn-autoplay").disabled = human !== "none";
  $("btn-autoplay").classList.toggle("off", human !== "none");
  state.autoplay = true;
  $("btn-autoplay").textContent = "自动对战";
  resetGame(human);
}

async function resetGame(human) {
  cancelScheduled();                        // 先清掉所有计时器 + 复盘状态
  closeReplay();
  try {
    if (human) await api("/api/player", {
      method: "POST", body: JSON.stringify({ human }),
    });
    const snap = await api("/api/reset", { method: "POST" });
    state.snapshot = snap;
    state.last = null;
    state.gameOver = false;
    state.anim = null;
    $("game-overlay").hidden = true;
    $("chip-action").textContent = "待决策";
    $("board-desc").textContent = human === "none"
      ? "纯 AI 对战:两个模型(黑/白)轮流自动落子。点「自动对战 / 单步」控制节奏。"
      : "人执" + (human === "black" ? "黑先手" : "白后手") + "。轮到人时点绿点落子;AI 会自动出招。";
    renderAll();
    updateControls();
    scheduleAi();
  } catch (e) {
    console.error("开局失败:", e);
  }
}

/* ---------------- 走一步 & 自动续走 ---------------- */
async function refresh() {
  const snap = await api("/api/state");
  state.snapshot = snap;
  state.gameOver = snap.state.game_over;
  renderAll();
  updateControls();
  if (state.gameOver) onGameOver(snap.state);
  else scheduleAi();
}

async function doStep(body) {
  if (state.busy || state.gameOver || state.replay.active) return;
  state.busy = true;
  try {
    const res = await api("/api/step", { method: "POST", body: JSON.stringify(body) });
    startMoveAnim(res);               // 动画要在 refresh() 覆盖快照前捕获旧棋盘
    state.last = res;
    await refresh();
  } catch (e) {
    console.error("step 失败:", e);
  } finally {
    state.busy = false;
    updateControls();
  }
}

function scheduleAi() {
  // 注意:不能在这里检查 state.busy——refresh() 在 busy 仍为 true 时
  // 就会调用本函数(step 刚完成),否则 AI 永远排不上下一步。
  if (state.timer) return;
  if (state.replay.active) return;   // 复盘时暂停实况推进
  const g = state.snapshot?.state;
  if (!g || g.game_over || !g.ai_turn) return;
  // 纯 AI 对战:只有在"自动对战"开着时才连续走
  if (state.human === "none" && !state.autoplay) return;
  state.timer = setTimeout(async () => {
    state.timer = null;
    await doStep({ ai: true });
  }, state.aiDelay);
}

function onGameOver(g) {
  $("overlay-title").textContent = g.winner_name
    ? `本局结束 · ${g.winner_name}胜` : "本局结束 · 平局";
  $("overlay-sub").textContent =
    `黑 ${g.counts.black} : ${g.counts.white} 白 · 共 ${g.steps} 手`;
  $("game-overlay").hidden = false;
}

/* ---------------- 模型切换(平台化的"换模型") ---------------- */
async function switchModel(key) {
  if (!key) return;
  try {
    cancelScheduled();              // 先清在途的 AI 定时器/动画,避免旧步插进来
    const res = await api("/api/models/load", {
      method: "POST", body: JSON.stringify({ name: key }),
    });
    state.snapshot = res;                 // 切换后返回最新快照,直接刷新
    state.last = null;
    state.gameOver = res.state.game_over;
    state.anim = null;
    $("game-overlay").hidden = true;
    $("model-name").textContent = res.model.loaded ? `模型 ${res.model.name}` : "随机初始模型";
    $("badge-model").classList.toggle("fresh", !res.model.loaded);
    $("eps-val").textContent = res.epsilon.toFixed(3);
    renderAll();
    updateControls();
    if (!state.gameOver) scheduleAi();
  } catch (e) {
    console.error("切换模型失败:", e);
  }
}

/* ---------------- 启动 ---------------- */
async function init() {
  try {
    const [snap, cfg, model, curveData, meta] = await Promise.all([
      api("/api/state"), api("/api/config"), api("/api/model"),
      api("/api/curve"), api("/api/meta"),
    ]);

    state.snapshot = snap;
    state.gameOver = snap.state.game_over;
    state.human = snap.state.human_color === null ? "none"
      : snap.state.human_color === 1 ? "black" : "white";
    document.querySelectorAll(".mode-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.human === state.human);
    });
    $("btn-autoplay").disabled = state.human !== "none";
    $("btn-autoplay").classList.toggle("off", state.human !== "none");

    $("model-name").textContent = model.loaded ? `模型 ${model.name}` : "随机初始模型";
    $("badge-model").classList.toggle("fresh", !model.loaded);
    $("eps-val").textContent = snap.epsilon.toFixed(3);

    renderConfig($("config-grid"), [
      ["学习率 lr", cfg.lr], ["折扣 γ", cfg.gamma],
      ["batch 大小", cfg.batch_size], ["经验池", cfg.buffer_capacity],
      ["目标网同步", cfg.target_update_freq], ["ε 起点", cfg.epsilon_start],
      ["ε 终点", cfg.epsilon_end], ["ε 衰减", cfg.epsilon_decay],
      ["随机对手概率", cfg.random_opponent_prob], ["对手池大小", cfg.opponent_pool_size],
      ["棋盘", "8×8 · 64 格"],
      ["网络结构", `${cfg.input_dim}→${cfg.hidden_dims.join("→")}→${cfg.n_actions}`],
    ]);

    curve(curveData);
    state.metaModels = renderModelPicker($("model-select"), meta.models || [], model.name);
    populateDuelSelects();            // 模型对战面板的选手下拉
    renderAll();
    updateControls();
    scheduleAi();

    // 训练状态面板:自适应轮询(训练中 2s,空闲 15s)
    startTrainPolling(pollTrain);

    // 竞技场:加载上次持久化的统一评测排行榜
    try { renderBenchmark(await api("/api/benchmark")); } catch (e) { /* 忽略 */ }
  } catch (e) {
    console.error("初始化失败:", e);
    document.body.insertAdjacentHTML("beforeend",
      `<div style="position:fixed;inset:0;display:grid;place-items:center;background:#0c0e16;color:#f87171;font-size:15px">无法连接后端服务,请先运行 python othello/serve.py</div>`);
  }
}

/* ---------------- 事件绑定 ---------------- */
// 棋盘点击:轮到人时落子
$("board-canvas").addEventListener("click", (e) => {
  const g = state.snapshot?.state;
  if (!g || g.game_over || !g.human_turn || state.busy || state.replay.active) return;
  const canvas = e.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor((e.clientX - rect.left) / (rect.width / N));
  const r = Math.floor((e.clientY - rect.top) / (rect.height / N));
  const action = r * N + c;
  if (g.legal_moves.includes(action)) doStep({ action });
});

// 鼠标悬停:合法落子位置给"手型"提示
$("board-canvas").addEventListener("mousemove", (e) => {
  const g = state.snapshot?.state;
  const canvas = e.currentTarget;
  if (!g || g.game_over || !g.human_turn || state.replay.active) {
    canvas.style.cursor = "default"; return;
  }
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor((e.clientX - rect.left) / (rect.width / N));
  const r = Math.floor((e.clientY - rect.top) / (rect.height / N));
  canvas.style.cursor = g.legal_moves.includes(r * N + c) ? "pointer" : "default";
});

// 对战模式切换
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => setHumanMode(btn));
});

// AI 间隔(快/中/慢)
document.querySelectorAll(".ctl-btn[data-delay]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".ctl-btn[data-delay]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.aiDelay = parseInt(btn.dataset.delay, 10);
  });
});

// 纯 AI 对战:自动对战 / 单步
$("btn-autoplay").addEventListener("click", () => {
  state.autoplay = !state.autoplay;
  $("btn-autoplay").textContent = state.autoplay ? "自动对战" : "手动单步";
  $("btn-autoplay").classList.toggle("off", !state.autoplay);
  if (state.autoplay) scheduleAi();
});
$("btn-step").addEventListener("click", () => doStep({ ai: true }));
$("btn-restart").addEventListener("click", () => resetGame());

// 悔棋:悔完若轮到 AI 则续走(服务端人机模式会连弹到人的回合,此时自动跳过)
$("btn-undo").addEventListener("click", async () => {
  await doUndo();
  if (!state.gameOver) scheduleAi();
});

// 整局复盘(本局棋谱)
$("btn-replay").addEventListener("click", async () => {
  try {
    const h = await api("/api/history");
    if (!h.moves.length) { console.warn("还没有可复盘的落子"); return; }
    openReplay({ boards: h.boards, moves: h.moves, title: `本局复盘 · 共 ${h.moves.length} 手` });
  } catch (e) { console.error("加载棋谱失败:", e); }
});

// 复盘控制条
$("rp-start").addEventListener("click", () => replaySeek(0));
$("rp-prev").addEventListener("click", () => {
  const rp = state.replay;
  if (rp.playing) toggleReplayPlay();
  replaySeek(rp.index - 1);
});
$("rp-next").addEventListener("click", () => {
  const rp = state.replay;
  if (rp.playing) toggleReplayPlay();
  replaySeek(rp.index + 1);
});
$("rp-end").addEventListener("click", () => {
  replaySeek(state.replay.boards.length - 1);
});
$("rp-play").addEventListener("click", toggleReplayPlay);
$("rp-slider").addEventListener("input", (e) => {
  const rp = state.replay;
  if (rp.playing) toggleReplayPlay();
  replaySeek(parseInt(e.target.value, 10));
});
$("rp-exit").addEventListener("click", () => {
  closeReplay();
  scheduleAi();
});

// 模型切换下拉框
$("model-select").addEventListener("change", (e) => switchModel(e.target.value));

// 模型对战
$("btn-duel-start").addEventListener("click", startDuel);
$("btn-duel-clear").addEventListener("click", clearDuel);
$("duel-black").addEventListener("change", updateDuelSelectScores);
$("duel-white").addEventListener("change", updateDuelSelectScores);

// 竞技场:统一评测 + 循环赛打榜
$("btn-arena-start").addEventListener("click", startArena);
$("btn-arena-clear").addEventListener("click", clearArena);
$("btn-benchmark").addEventListener("click", runBenchmark);

/* Tab 切换:模型决策 / 训练进度 */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    if (tab.classList.contains("active")) return;
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach((p) => p.hidden = true);
    const panel = document.getElementById(tab.dataset.tab);
    panel.hidden = false;
    // 切到训练进度面板时重绘曲线(宽度可能变了)
    if (tab.dataset.tab === "panel-training") redrawCurve();
  });
});

/* 观察输入:展开/折叠 8×8 网格 */
$("btn-obs-expand").addEventListener("click", () => {
  state.obsGridExpanded = !state.obsGridExpanded;
  document.querySelectorAll(".obs-grid").forEach((el) => el.hidden = !state.obsGridExpanded);
  const btn = $("btn-obs-expand");
  btn.textContent = state.obsGridExpanded ? "▾ 收起 8×8 网格" : "▸ 展开 8×8 网格";
  btn.classList.toggle("expanded", state.obsGridExpanded);
});

/* 模型工具折叠/展开 */
$("btn-tools-toggle").addEventListener("click", () => {
  const body = $("tools-body");
  const btn = $("btn-tools-toggle");
  const expanded = btn.getAttribute("aria-expanded") === "true";
  btn.setAttribute("aria-expanded", !expanded);
  body.hidden = expanded;
});

window.addEventListener("resize", redrawCurve);

init();
