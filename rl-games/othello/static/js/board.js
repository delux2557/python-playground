/* board.js —— 黑白棋棋盘渲染 + 决策面板 + 观察输入 + HUD + 动画。
   职责单一:只负责"把 state 画到界面上",不含任何对局流程/网络逻辑。
   也提供流程模块共用的 cancelScheduled / renderAll / updateControls。 */

import { state, $ } from "./state.js";
import { setupCanvas } from "/shared/frontend/common.js";

/* 防御:确保 DOM 已解析再获取画布——模块脚本在极少数环境可能比 DOM 解析
   更早执行,此时 $("board-canvas") 会拿到 null。顶层 await 只作用于
   这个模块,其它模块 import 它会自然等它就绪。 */
await new Promise((resolve) => {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", resolve, { once: true });
  } else resolve();
});

const N = 8;                       // 8×8 棋盘
const BOARD_SIZE = 440;            // canvas CSS 尺寸
const CELL = BOARD_SIZE / N;
const COL_LETTERS = "abcdefgh";    // 列名 a~h(决策面板展示用)

const boardCanvas = $("board-canvas");
const boardCtx = setupCanvas(boardCanvas, BOARD_SIZE);

/* ---------------- 棋盘渲染 ---------------- */
function heatColor(qv, cells) {
  const vals = cells.filter((v) => v != null);
  const min = Math.min(...vals), max = Math.max(...vals);
  const t = max > min ? (qv - min) / (max - min) : 0.5;  // 0=最低 1=最高
  return `hsla(${t * 120}, 85%, 55%, 0.22)`;             // 0°红(低) → 120°绿(高)
}

function drawDisc(ctx, x, y, r, player) {
  const grad = ctx.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r);
  if (player > 0) {                    // 黑子
    grad.addColorStop(0, "#3a3f52");
    grad.addColorStop(1, "#0b0d16");
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  } else {                             // 白子
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(1, "#c2c9d6");
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(120,130,150,.45)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// 通用缩放绘制(落子"长大"、翻子"压扁"都靠它)
function drawScaled(ctx, x, y, r, player, sx, sy) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(sx, sy);
  drawDisc(ctx, 0, 0, r, player);
  ctx.restore();
}

// 3D 翻转棋子:prog 0→1 时棋子沿水平轴"翻面",前一半是旧色、后一半是新色
function drawFlip(ctx, x, y, r, finalPlayer, prog) {
  const sy = Math.abs(Math.cos(prog * Math.PI));
  const cur = prog < 0.5 ? -finalPlayer : finalPlayer;
  drawScaled(ctx, x, y, r, cur, 1, Math.max(0.08, sy));
}

// 弹性回弹缓动:落子先快后慢还带一点点回弹,更生动
function easeOutBack(t) {
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

export function renderBoard() {
  const ctx = boardCtx;
  ctx.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);

  // 数据源:复盘帧 或 实况
  let board, highlight = null;
  if (state.replay.active) {
    board = state.replay.boards[state.replay.index];
    if (state.replay.index > 0) {
      const a = state.replay.moves[state.replay.index - 1];
      highlight = { r: Math.floor(a / N), c: a % N };
    }
  } else {
    const g = state.snapshot?.state;
    if (!g) return;
    board = g.board;
    if (state.last && state.last.action_rc) {
      highlight = { r: state.last.action_rc[0], c: state.last.action_rc[1] };
    }
  }

  const cells = state.snapshot?.q?.cells;     // 仅实况有 Q 值热力
  const showHeat = !state.replay.active && !!cells;

  // 棋盘格底色 + Q 值热力图色块
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      ctx.fillStyle = (r + c) % 2 === 0 ? "#1a2030" : "#141a26";
      ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
      if (showHeat) {
        const qv = cells[r * N + c];
        if (qv != null) {
          ctx.fillStyle = heatColor(qv, cells);
          ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
        }
      }
    }
  }

  // 网格线
  ctx.strokeStyle = "rgba(148,163,184,.10)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, BOARD_SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(BOARD_SIZE, i * CELL); ctx.stroke();
  }

  // 上一步落子:黄色描框高亮
  if (highlight) {
    ctx.strokeStyle = "rgba(251,191,36,.85)";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(highlight.c * CELL + 3, highlight.r * CELL + 3, CELL - 6, CELL - 6);
  }

  // 落子动画进度(仅在实况且动画进行中时生效)
  let animT = 1;
  const anim = state.anim;
  if (anim && !state.replay.active) {
    animT = Math.min(1, (performance.now() - anim.t0) / anim.duration);
  }
  const animCells = new Set();
  if (anim && !state.replay.active) {
    if (anim.placed) animCells.add(anim.placed.r * N + anim.placed.c);
    anim.flips.forEach((f) => animCells.add(f.r * N + f.c));
  }

  // 棋子
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const v = board[r][c];
      if (v === 0) continue;
      const x = c * CELL + CELL / 2, y = r * CELL + CELL / 2, rad = CELL * 0.34;
      const idx = r * N + c;
      if (animCells.has(idx)) {
        if (anim.placed && anim.placed.r === r && anim.placed.c === c) {
          // 落子位:从小到大"弹出"
          const s = easeOutBack(Math.min(1, animT / 0.45));
          drawScaled(ctx, x, y, rad, v, s, s);
        } else {
          // 被夹翻的棋子:3D 翻面
          drawFlip(ctx, x, y, rad, v, Math.min(1, animT / 0.62));
        }
        continue;
      }
      drawDisc(ctx, x, y, rad, v);
    }
  }

  // 轮到人落子时:合法位置画成绿色小圆点(可点击)
  if (!state.replay.active) {
    const g = state.snapshot?.state;
    if (g && !g.game_over && g.human_turn) {
      ctx.fillStyle = "rgba(46,230,168,.6)";
      for (const i of g.legal_moves) {
        const r = Math.floor(i / N), c = i % N;
        ctx.beginPath();
        ctx.arc(c * CELL + CELL / 2, r * CELL + CELL / 2, CELL * 0.10, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

/* ---------------- 落子动画(核心:对比前后棋盘找出新子与翻子) ---------------- */
export function startMoveAnim(res) {
  const prev = state.snapshot?.state?.board;
  const board = res?.state?.board;
  const rc = res?.action_rc;
  if (!board || !rc || rc.length < 2) return;
  const [r, c] = rc;
  const player = board[r][c];                 // 落子后该格子的颜色 = 落子方
  const flips = [];
  if (prev) {
    for (let i = 0; i < N * N; i++) {
      const pr = Math.floor(i / N), pc = i % N;
      if (prev[pr][pc] === -player && board[pr][pc] === player) {
        flips.push({ r: pr, c: pc });
      }
    }
  }
  state.anim = { placed: { r, c, player }, flips, t0: performance.now(), duration: 520 };
  requestAnimationFrame(animFrame);
}

function animFrame(now) {
  const a = state.anim;
  if (!a) return;
  const t = Math.min(1, (now - a.t0) / a.duration);
  renderBoard();
  if (t < 1) requestAnimationFrame(animFrame);
  else { state.anim = null; renderBoard(); }
}

/* ---------------- 子数 / HUD / 回合状态 ---------------- */
export function renderBoardMeta() {
  const g = state.snapshot?.state;
  if (!g) return;
  $("count-black").textContent = g.counts.black;
  $("count-white").textContent = g.counts.white;
  $("hud-turn").textContent = g.game_over ? "终局" : g.current_name;
  $("hud-steps").textContent = g.steps;
  $("hud-flips").textContent = state.last?.flips != null ? state.last.flips : "–";
  $("hud-eps").textContent = state.snapshot.epsilon.toFixed(3);

  const chip = $("turn-chip");
  if (state.replay.active) {
    chip.textContent = "复盘模式";
    chip.className = "chip";
  } else if (g.game_over) {
    chip.textContent = `终局 · ${g.winner_name || "—"}`;
    chip.className = "chip chip-ai";
  } else if (g.human_turn) {
    chip.textContent = `轮到你(${g.current_name})`;
    chip.className = "chip chip-ai";
  } else {
    chip.textContent = `AI 思考中…(${g.current_name})`;
    chip.className = "chip";
  }
}

/* ---------------- 候选落子:Q 值排序条 ---------------- */
const qBars = $("q-bars");

function cellName(rc) {
  if (!rc) return "—";
  return COL_LETTERS[rc[1]] + (rc[0] + 1);   // 如 [2,1] → b3
}

export function renderDecision() {
  const q = state.snapshot?.q;
  if (!q) return;
  const top = q.top || [];
  const chosen = state.last?.action;
  const maxQ = Math.max(...top.map((t) => Math.abs(t.q)), 0.001);
  const shown = top.slice(0, 12);
  qBars.innerHTML = "";
  shown.forEach((t) => {
    const row = document.createElement("div");
    row.className = "q-row" + (t.cell === chosen ? " chosen" : "");
    const width = Math.max(4, (Math.abs(t.q) / maxQ) * 100);
    row.innerHTML = `
      <span class="q-name">${cellName([t.r, t.c])}</span>
      <div class="q-track"><div class="q-fill" style="width:${width}%;background:${t.cell === chosen ? "#2ee6a8" : "#5f57d9"}"></div></div>
      <span class="q-val">${t.q.toFixed(3)}</span>`;
    qBars.appendChild(row);
  });

  const chip = $("chip-action");
  chip.textContent = state.last?.action_rc
    ? `上步落子 ${cellName(state.last.action_rc)}`
    : "待决策";
}

/* ---------------- 模型观察输入:3 通道迷你棋盘 ---------------- */
const obsBars = $("obs-bars");

function channelGrid(key) {
  // 以"当前落子方"视角拆出 3 通道的 0/1 网格
  const g = state.snapshot.state;
  const own = g.current;                 // BLACK=1 / WHITE=-1
  const opp = -own;
  const grid = [];
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const v = g.board[r][c];
      grid.push(key === "own" ? (v === own ? 1 : 0)
                : key === "opp" ? (v === opp ? 1 : 0)
                : (v === 0 ? 1 : 0));
    }
  }
  return grid;
}

export function renderObs() {
  const chans = state.snapshot?.obs?.channels;
  if (!chans) return;
  const maxCount = N * N;
  const colors = { own: "#7c6cff", opp: "#fbbf24", empty: "#334155" };
  obsBars.innerHTML = "";
  chans.forEach((ch) => {
    const block = document.createElement("div");
    block.className = "obs-channel";
    const width = (ch.count / maxCount) * 100;
    const grid = channelGrid(ch.key);
    block.innerHTML = `
      <div class="obs-head">
        <span class="obs-name">${ch.label}</span>
        <span class="obs-count">${ch.count} / 64</span>
      </div>
      <div class="obs-track"><div class="obs-fill" style="width:${width}%;background:${colors[ch.key] || "#5f57d9"}"></div></div>
      <div class="obs-grid"${state.obsGridExpanded ? "" : " hidden"}>
        <div class="mini-grid">${grid.map((v) =>
          `<span class="mini-cell" style="background:${v ? (colors[ch.key] || "#5f57d9") : "transparent"}"></span>`).join("")}</div>
      </div>`;
    obsBars.appendChild(block);
  });
  const btn = $("btn-obs-expand");
  btn.textContent = state.obsGridExpanded ? "▾ 收起 8×8 网格" : "▸ 展开 8×8 网格";
  btn.classList.toggle("expanded", state.obsGridExpanded);
}

/* ---------------- 汇总渲染 / 控件状态 / 定时器清理 ---------------- */
export function renderAll() {
  renderBoard();
  renderBoardMeta();
  renderDecision();
  renderObs();
}

export function updateControls() {
  const inReplay = state.replay.active;
  const g = state.snapshot?.state;
  // 终局后也允许悔棋(撤销最后一手翻盘),所以不再用 game_over 禁用
  $("btn-undo").disabled = inReplay || state.busy || !g || g.steps <= 0;
  $("btn-replay").disabled = inReplay;
}

export function cancelScheduled() {
  if (state.timer) { clearTimeout(state.timer); state.timer = null; }
  const rp = state.replay;
  if (rp.timer) { clearInterval(rp.timer); rp.timer = null; }
  rp.playing = false;
  if (state.anim) state.anim = null;
}
