/* board.js —— 贪吃蛇驾驶舱的渲染(画面 / 决策 / 观察 / HUD)。
   职责单一:只负责"把 state 画到界面上",不含对局流程/网络逻辑。 */

import { state, $ } from "./state.js";
import { setupCanvas } from "/shared/frontend/common.js";

/* 防御:确保 DOM 已解析再获取画布——模块脚本在极少数环境可能比 DOM 解析
   更早执行,此时 $("game-canvas") 会拿到 null。顶层 await 只作用于
   这个模块,其它模块 import 它会自然等它就绪。 */
await new Promise((resolve) => {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", resolve, { once: true });
  } else resolve();
});

// 动作名 / 观察含义:默认值(与 snake/env.py 对应)。
// 启动后会被 /api/meta 的元数据覆盖——这就是"元数据驱动":前端不写死
// 任何游戏细节,一切以服务端自报家门为准(换游戏时这里自动适配)。
let ACTION_NAMES = ["上", "下", "左", "右"];

// 观察向量每个维度的含义(对应 snake/env.py 的 _get_obs)
let OBS_LABELS = [
  ["危险·上", "danger"], ["危险·下", "danger"],
  ["危险·左", "danger"], ["危险·右", "danger"],
  ["食物·水平 dx", "food"], ["食物·垂直 dy", "food"],
  ["方向·上", "dir"], ["方向·下", "dir"],
  ["方向·左", "dir"], ["方向·右", "dir"],
  ["饥饿度", "hunger"],
];

// 元数据驱动的注入点:main.js 在 init 时用 /api/meta 覆盖默认值
export function setActionNames(names) { ACTION_NAMES = names; }
export function setObsLabels(meaning, group) {
  OBS_LABELS = meaning.map((m, i) => [m, group?.[i] || "x"]);
}

/* ---------------- 画布 ---------------- */
const gameCanvas = $("game-canvas");
const gameCtx = setupCanvas(gameCanvas, 440);

/* ---------------- 游戏画面 ---------------- */
export function renderGame() {
  const g = state.snapshot?.state;
  if (!g) return;
  const size = 440, n = g.grid_size;
  const cell = size / n;
  const ctx = gameCtx;
  ctx.clearRect(0, 0, size, size);

  // 棋盘底色 + 网格线(淡淡的棋盘格更清爽)
  ctx.fillStyle = "#141926";
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = "rgba(148,163,184,.08)";
  ctx.lineWidth = 1;
  for (let i = 1; i < n; i++) {
    ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); ctx.stroke();
  }

  // 蛇身:从尾到头渐变,越靠近头颜色越亮
  const snake = g.snake;
  for (let i = snake.length - 1; i >= 0; i--) {
    const [r, c] = snake[i];
    const t = i / Math.max(1, snake.length - 1); // 0=头 1=尾
    const bright = 1 - t * 0.55;
    ctx.fillStyle = i === 0 ? "#4ade80"
      : `rgba(${34 + Math.round(t * 120)}, ${197 - Math.round(t * 60)}, ${94 + Math.round(t * 90)}, 1)`;
    roundRect(ctx, c * cell + 2, r * cell + 2, cell - 4, cell - 4, 6);
    ctx.fill();
    if (i === 0) {
      // 蛇头眼睛:朝向当前移动方向(经典贪吃蛇做法,直观反馈方向)
      drawHeadEyes(ctx, r, c, cell, state.dir);
    }
  }

  // 食物:带光晕的红果
  const [fr, fc] = g.food;
  const fx = fc * cell + cell / 2, fy = fr * cell + cell / 2;
  const glow = ctx.createRadialGradient(fx, fy, 1, fx, fy, cell * 0.9);
  glow.addColorStop(0, "rgba(251,113,133,.5)");
  glow.addColorStop(1, "rgba(251,113,133,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(fx - cell, fy - cell, cell * 2, cell * 2);
  ctx.fillStyle = "#fb7185";
  ctx.beginPath(); ctx.arc(fx, fy, cell * 0.32, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#fecdd3";
  ctx.beginPath(); ctx.arc(fx - cell * 0.09, fy - cell * 0.11, cell * 0.09, 0, Math.PI * 2); ctx.fill();
}

// 蛇头眼睛:两只白眼球 + 黑瞳孔,瞳孔朝移动方向偏移,直观显示控制方向
function drawHeadEyes(ctx, r, c, cell, dir) {
  const vec = [[-1, 0], [1, 0], [0, -1], [0, 1]][dir] || [1, 0];
  const [dr, dc] = vec;
  const [pr, pc] = [dc, dr];        // 垂直方向(决定两只眼睛的排列方向)
  const cx = c * cell + cell / 2;
  const cy = r * cell + cell / 2;
  const eyeOff = cell * 0.24;        // 眼距中心
  const shift = cell * 0.10;         // 瞳孔朝方向的偏移量
  const eyeR = cell * 0.14, pupilR = cell * 0.07;
  for (const s of [-1, 1]) {
    const ex = cx + pr * s * eyeOff;
    const ey = cy + pc * s * eyeOff;
    ctx.fillStyle = "#f8fafc";
    ctx.beginPath(); ctx.arc(ex, ey, eyeR, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#0b1220";
    ctx.beginPath(); ctx.arc(ex + dc * shift, ey + dr * shift, pupilR, 0, Math.PI * 2); ctx.fill();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* ---------------- 本次决策面板 ---------------- */
const qBars = $("q-bars");

export function renderDecision(last) {
  if (!last) return;
  const q = last.q_values;
  const chosen = last.action;
  const maxQ = Math.max(...q.map(Math.abs), 0.001);
  qBars.innerHTML = "";
  q.forEach((val, i) => {
    const row = document.createElement("div");
    row.className = "q-row" + (i === chosen ? " chosen" : "");
    const width = Math.max(4, (Math.abs(val) / maxQ) * 100);

    row.innerHTML = `
      <span class="q-name">${ACTION_NAMES[i]}</span>
      <div class="q-track"><div class="q-fill" style="width:${width}%;background:${i === chosen ? "#2ee6a8" : "#5f57d9"}"></div></div>
      <span class="q-val">${val.toFixed(3)}</span>`;
    qBars.appendChild(row);
  });
  // action=-1 是"还没走过"的占位(初始/换模型后),显示"待决策"而非 undefined
  $("chip-action").textContent = chosen >= 0 ? "选择 " + ACTION_NAMES[chosen] : "待决策";
  $("meta-epsilon").textContent = `ε ${last.epsilon.toFixed(3)}`;
  $("meta-steps").textContent = `步数 ${last.state.steps}`;
  renderRewardPill(last.reward);
}

function renderRewardPill(reward) {
  const el = $("meta-reward");
  el.textContent = `奖励 ${reward >= 0 ? "+" : ""}${reward.toFixed(1)}`;
  el.className = "meta-pill " + (reward > 0 ? "win" : reward < 0 && Math.abs(reward) > 1 ? "lose" : "");
}

/* ---------------- 观察输入面板 ---------------- */
const obsBars = $("obs-bars");

export function renderObs(obs) {
  if (!obs) return;
  obsBars.innerHTML = "";
  obs.forEach((val, i) => {
    const [label, cls] = OBS_LABELS[i];
    const width = Math.max(2, Math.min(100, Math.abs(val) * 100));
    const item = document.createElement("div");
    item.className = `obs-item ${cls}`;
    item.innerHTML = `
      <span class="obs-label">${label}</span>
      <div class="obs-track"><div class="obs-fill" style="width:${width}%"></div></div>
      <span class="obs-val">${val.toFixed(2)}</span>`;
    obsBars.appendChild(item);
  });
}

/* ---------------- HUD ---------------- */
export function renderHud() {
  const g = state.snapshot?.state;
  if (!g) return;
  $("hud-score").textContent = g.score;
  $("hud-score2").textContent = g.score;
  $("hud-steps").textContent = g.steps;
  const r = state.last?.reward;
  const el = $("hud-reward");
  el.textContent = r === undefined ? "–" : (r >= 0 ? "+" : "") + r.toFixed(1);
  el.className = "hud-val" + (r > 0 ? " pos" : r < 0 && Math.abs(r) > 1 ? " neg" : "");
  $("hud-reason").textContent = state.gameOver ? (state.last.reason || "结束") : "进行中";
}

/* ---------------- 汇总渲染 ---------------- */
export function renderAll(last) {
  renderGame();
  if (last && last.pre_obs) renderObs(last.pre_obs);
  if (last && last.q_values) renderDecision(last);
  renderHud();
}
