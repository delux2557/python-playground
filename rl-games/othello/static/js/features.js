/* features.js —— 黑白棋驾驶舱的"高级功能":悔棋 / 整局复盘回放 /
   模型对战 / 分区热力图 / 竞技场打榜 / 统一评测。
   职责:只做"拉数据 + 更新对应面板",不含开局/AI 回合等主流程(在 main.js)。
   依赖 board.js 的渲染能力;main.js 负责在事件里编排主流程。 */

import { state, $ } from "./state.js";
import { cancelScheduled, renderAll, renderBoard, updateControls } from "./board.js";
import { api } from "/shared/frontend/common.js";

/* ---------------- 悔棋 ---------------- */
export function canUndo() {
  // 终局后也允许悔棋:撤销最后一手重新争取翻盘(服务端已支持)
  return !!state.snapshot && !state.replay.active
    && state.snapshot.state.steps > 0;
}

export async function doUndo() {
  if (state.busy) return;
  state.busy = true;
  try {
    cancelScheduled();              // 先停掉在途的 AI 步,悔棋期间不许抢下
    const res = await api("/api/undo", { method: "POST" });
    state.anim = null;
    state.snapshot = res;
    state.last = null;
    state.gameOver = res.state.game_over;
    $("game-overlay").hidden = true;   // 终局悔棋:收起结束遮罩
    renderAll();
    updateControls();
    // 注意:悔棋后是否继续 AI 回合由 main.js 的事件处理器负责(scheduleAi)
  } catch (e) {
    console.error("悔棋失败:", e);
  } finally {
    state.busy = false;
  }
}

/* ---------------- 整局复盘回放(主棋盘复用为"放映机") ---------------- */
export function openReplay({ boards, moves, title }) {
  cancelScheduled();
  const rp = state.replay;
  rp.active = true;
  rp.boards = boards;
  rp.moves = moves;
  rp.index = 0;
  rp.playing = false;
  rp.title = title || "整局复盘";
  $("replay-bar").hidden = false;
  $("replay-title").textContent = rp.title;
  $("rp-exit").textContent = "退出复盘";
  updateReplayUI();
  renderAll();
  updateControls();
}

export function closeReplayUI() {
  const rp = state.replay;
  rp.active = false;
  rp.playing = false;
  if (rp.timer) { clearInterval(rp.timer); rp.timer = null; }
  $("replay-bar").hidden = true;
}

export function closeReplay() {
  closeReplayUI();
  renderAll();
  updateControls();
  // 退出复盘后是否续走 AI 由 main.js 的事件处理器负责(scheduleAi)
}

export function replaySeek(i) {
  const rp = state.replay;
  rp.index = Math.max(0, Math.min(rp.boards.length - 1, i));
  updateReplayUI();
  renderBoard();
}

function updateReplayUI() {
  const rp = state.replay;
  const total = rp.boards.length - 1;
  $("rp-slider").max = total;
  $("rp-slider").value = rp.index;
  $("rp-counter").textContent = `${rp.index} / ${total}`;
  $("rp-play").innerHTML = rp.playing
    ? '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M7 5h3v14H7zM14 5h3v14h-3z"/></svg>'
    : '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M8 5l11 7-11 7z"/></svg>';
}

export function toggleReplayPlay() {
  const rp = state.replay;
  if (rp.playing) {
    rp.playing = false;
    if (rp.timer) { clearInterval(rp.timer); rp.timer = null; }
  } else {
    if (rp.index >= rp.boards.length - 1) rp.index = 0;
    rp.playing = true;
    rp.timer = setInterval(() => {
      if (rp.index >= rp.boards.length - 1) {
        rp.playing = false;
        if (rp.timer) { clearInterval(rp.timer); rp.timer = null; }
        updateReplayUI();
        return;
      }
      rp.index++;
      updateReplayUI();
      renderBoard();
    }, 480);
  }
  updateReplayUI();
}

/* ---------------- 模型对战:黑 vs 白 ---------------- */
const DUEL_POLL_MS = 700;

function duelScoreOf(key) {
  const m = (state.metaModels || []).find((x) => x.key === key);
  return m && m.eval_score != null ? m.eval_score : null;
}

export function updateDuelSelectScores() {
  for (const [id, outId] of [["duel-black", "duel-black-score"], ["duel-white", "duel-white-score"]]) {
    const key = $(id).value;
    const s = duelScoreOf(key);
    $(outId).textContent = s != null ? `评估 ${s}` : "随机权重 · 未评估";
    $(outId).classList.toggle("has-score", s != null);
  }
}

export function populateDuelSelects() {
  const models = state.metaModels || [];
  const defs = [models[0]?.key || "random", models[1]?.key || "random"];
  ["duel-black", "duel-white"].forEach((id, i) => {
    const sel = $(id);
    sel.innerHTML = "";
    const randOpt = document.createElement("option");
    randOpt.value = "random";
    randOpt.textContent = "随机初始模型";
    sel.appendChild(randOpt);
    models.forEach((m) => {
      const o = document.createElement("option");
      o.value = m.key;
      o.textContent = m.key;
      sel.appendChild(o);
    });
    sel.value = defs[i];
  });
  updateDuelSelectScores();
}

export async function startDuel() {
  if (state.duel.running) return;
  const black = $("duel-black").value;
  const white = $("duel-white").value;
  const games = Math.max(1, Math.min(100, parseInt($("duel-games").value, 10) || 10));
  $("duel-games").value = games;
  $("btn-duel-start").disabled = true;
  $("duel-score").hidden = false;
  $("duel-regions").hidden = true;
  state.duel.running = true;
  try {
    const st = await api("/api/duel/start", {
      method: "POST",
      body: JSON.stringify({ black, white, games }),
    });
    renderDuelStatus(st);
    pollDuel();
  } catch (e) {
    console.error("开始对战失败:", e);
    state.duel.running = false;
    $("btn-duel-start").disabled = false;
    $("duel-badge").textContent = "出错了";
    $("duel-badge").className = "chip chip-error";
  }
}

function pollDuel() {
  if (!state.duel.running) return;
  clearTimeout(state.duel.pollTimer);
  state.duel.pollTimer = setTimeout(async () => {
    let st;
    try {
      st = await api("/api/duel/status");
    } catch (e) {
      state.duel.pollTimer = setTimeout(pollDuel, 1500);   // 网络抖动,稍后重试
      return;
    }
    renderDuelStatus(st);
    if (!st.running) {
      state.duel.running = false;
      $("btn-duel-start").disabled = false;
      $("duel-badge").textContent = st.error ? "对局出错" : "已结束";
      $("duel-badge").className = "chip " + (st.error ? "chip-error" : "chip-muted");
      try { renderDuelRegions(await api("/api/duel/regions")); } catch (e) { /* 忽略 */ }
    } else {
      pollDuel();
    }
  }, DUEL_POLL_MS);
}

function renderDuelStatus(st) {
  $("duel-badge").textContent = st.running ? `对局中 ${st.played}/${st.games}` : "已结束";
  $("duel-badge").className = "chip " + (st.running ? "chip-running" : "chip-muted");

  $("duel-black-wins").textContent = st.black_wins;
  $("duel-draws").textContent = st.draws;
  $("duel-white-wins").textContent = st.white_wins;

  // 三段比分条(黑胜 / 平 / 白胜)
  const g = st.games || 1;
  $("dw-black").style.width = (st.black_wins / g * 100).toFixed(2) + "%";
  $("dw-draw").style.width = (st.draws / g * 100).toFixed(2) + "%";
  $("dw-white").style.width = (st.white_wins / g * 100).toFixed(2) + "%";

  const pct = st.games ? Math.round(st.played / st.games * 100) : 0;
  $("duel-progress-fill").style.width = pct + "%";
  $("duel-progress-text").textContent = `${st.played} / ${st.games} 局`;
  $("duel-winrate").textContent = st.black_win_rate != null
    ? `黑胜率 ${(st.black_win_rate * 100).toFixed(1)}%` : "—";

  renderDuelGameList(st.results || []);
}

function renderDuelGameList(results) {
  const el = $("duel-games-list");
  el.innerHTML = "";
  if (!results.length) { el.hidden = true; return; }
  el.hidden = false;
  results.forEach((g) => {
    const winner = g.result === 1 ? "黑胜" : g.result === -1 ? "白胜" : "平局";
    const row = document.createElement("button");
    row.className = "duel-game-row " + (g.result === 1 ? "w-black"
      : g.result === -1 ? "w-white" : "w-draw");
    row.innerHTML = `
      <span class="dg-index">#${g.index + 1}</span>
      <span class="dg-result">${winner}</span>
      <span class="dg-counts">${g.counts.black} : ${g.counts.white}</span>
      <span class="dg-replay">复盘 →</span>`;
    row.title = `查看第 ${g.index + 1} 局 · ${winner}`;
    row.addEventListener("click", async () => {
      try {
        const gg = await api(`/api/duel/game/${g.index}`);
        openReplay({
          boards: gg.boards,
          moves: gg.moves,
          title: `第 ${g.index + 1} 局复盘 · ${winner}`,
        });
      } catch (e) { console.error("加载对局失败:", e); }
    });
    el.appendChild(row);
  });
}

/* ---------------- 角/边/中心 分区胜率热力图 ---------------- */
function winRateColor(v) {
  const t = Math.max(0, Math.min(1, v));
  return `hsla(${t * 120}, 72%, 55%, 0.82)`;   // 0% 红 → 100% 绿
}

export function renderDuelRegions(data) {
  $("duel-regions").hidden = false;
  $("duel-regions-samples").textContent = `样本 ${data.samples}`;
  const grid = data.grid;
  const gridEl = $("region-grid");
  gridEl.innerHTML = "";
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const v = grid[r][c];
      const cell = document.createElement("div");
      cell.className = "region-cell";
      if (v == null) {
        cell.textContent = "—";
        cell.style.background = "rgba(148,163,184,.06)";
        cell.style.color = "var(--text-dim)";
        cell.title = "无样本";
      } else {
        const pct = Math.round(v * 100);
        cell.textContent = pct + "%";
        cell.style.background = winRateColor(v);
        cell.style.color = v >= 0.5 ? "rgba(8,12,22,.8)" : "#fff";
        cell.title = `${String.fromCharCode(97 + c)}${r + 1} · 胜率 ${pct}%`;
      }
      gridEl.appendChild(cell);
    }
  }

  const order = ["corner", "edge", "center"];
  const barsEl = $("region-bars");
  barsEl.innerHTML = "";
  order.forEach((k) => {
    const rg = data.regions[k];
    if (!rg) return;
    const row = document.createElement("div");
    row.className = "region-bar-row";
    const pct = rg.win_rate != null ? Math.round(rg.win_rate * 100) : 0;
    row.innerHTML = `
      <span class="rb-name">${rg.name}</span>
      <div class="rb-track"><div class="rb-fill" style="width:${rg.win_rate != null ? pct : 0}%;background:${rg.win_rate != null ? winRateColor(rg.win_rate) : "var(--surface-3)"}"></div></div>
      <span class="rb-val">${rg.win_rate != null ? pct + "%" : "—"}<small>${rg.win}/${rg.total}</small></span>`;
    barsEl.appendChild(row);
  });
}

export function clearDuel() {
  state.duel.running = false;
  clearTimeout(state.duel.pollTimer);
  $("duel-games-list").innerHTML = "";
  $("duel-games-list").hidden = true;
  $("duel-regions").hidden = true;
  $("duel-score").hidden = true;
  $("btn-duel-start").disabled = false;
  $("duel-badge").textContent = "待开始";
  $("duel-badge").className = "chip chip-muted";
}

/* ---------------- 竞技场:统一评测基准 + 循环赛打榜 ---------------- */
const ARENA_POLL_MS = 800;

function shortKey(name) {
  // 长 key(如 othello-20260827-144315-941011)折行前缩短,排行榜更清爽
  const m = /^(\w+)-(\d{8})-(\d{6})/.exec(name);
  return m ? `${m[1]}·${m[2].slice(2)} ${m[3].slice(0, 4)}` : name;
}

/* ---- 统一评测基准 ---- */
export function renderBenchmark(data) {
  const rows = (data && data.rows) || [];
  const list = $("bench-list");
  if (!rows.length) {
    list.innerHTML = '<div class="bench-empty">尚无已登记模型 —— 先训练再评测</div>';
    return;
  }
  list.innerHTML = rows.map((r, i) => {
    const s = r.score;
    const scoreTxt = s != null ? `${(s * 100).toFixed(1)}%` : "失败";
    return `<div class="bench-row">
      <span class="bench-rank">#${i + 1}</span>
      <span class="bench-name" title="${r.name}">${shortKey(r.name)}</span>
      <span class="bench-score${s != null ? "" : " off"}">${scoreTxt}</span>
    </div>`;
  }).join("");
}

export async function runBenchmark() {
  const btn = $("btn-benchmark");
  btn.disabled = true;
  btn.textContent = "评测中…";
  try {
    const data = await api("/api/benchmark", { method: "POST" });
    renderBenchmark(data);
  } catch (e) {
    console.error("统一评测失败:", e);
    $("bench-list").innerHTML = '<div class="bench-empty">评测失败(后端不可达或对战进行中)</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = "重新评测";
  }
}

/* ---- 循环赛 ---- */
export async function startArena() {
  if (state.arena.running) return;
  const games = Math.max(1, Math.min(50, parseInt($("arena-games").value, 10) || 10));
  $("arena-games").value = games;
  $("btn-arena-start").disabled = true;
  $("arena-progress").hidden = false;
  $("arena-board").hidden = true;
  $("arena-empty").hidden = true;
  state.arena.running = true;
  try {
    const st = await api("/api/arena/start", {
      method: "POST",
      body: JSON.stringify({ games_per_match: games }),
    });
    renderArenaStatus(st);
    pollArena();
  } catch (e) {
    console.error("开始循环赛失败:", e);
    state.arena.running = false;
    $("btn-arena-start").disabled = false;
    $("arena-badge").textContent = "出错了";
    $("arena-badge").className = "chip chip-error";
  }
}

function pollArena() {
  if (!state.arena.running) return;
  clearTimeout(state.arena.pollTimer);
  state.arena.pollTimer = setTimeout(async () => {
    let st;
    try {
      st = await api("/api/arena/status");
    } catch (e) {
      state.arena.pollTimer = setTimeout(pollArena, 1500);
      return;
    }
    renderArenaStatus(st);
    if (!st.running) {
      state.arena.running = false;
      $("btn-arena-start").disabled = false;
      $("arena-badge").textContent = st.error ? "循环赛出错" : "已结束";
      $("arena-badge").className = "chip " + (st.error ? "chip-error" : "chip-muted");
    } else {
      pollArena();
    }
  }, ARENA_POLL_MS);
}

function renderArenaStatus(st) {
  $("arena-badge").textContent = st.running ? `循环赛中 ${st.played}/${st.total}` : "已结束";
  $("arena-badge").className = "chip " + (st.running ? "chip-running" : "chip-muted");

  const pct = st.total ? Math.round(st.played / st.total * 100) : 100;
  $("arena-progress-fill").style.width = pct + "%";
  $("arena-progress-text").textContent = `${st.played} / ${st.total} 场`;

  const rows = st.leaderboard || [];
  $("arena-empty").hidden = rows.length > 0;
  if (!rows.length) return;
  $("arena-board").hidden = false;
  renderArenaBoard(rows);
}

function renderArenaBoard(rows) {
  const tbody = $("arena-tbody");
  tbody.innerHTML = rows.map((r) => {
    const rankCls = r.rank === 1 ? " r1" : r.rank === 2 ? " r2" : r.rank === 3 ? " r3" : "";
    const wr = r.win_rate != null ? (r.win_rate * 100).toFixed(1) + "%" : "—";
    const base = r.eval_score != null ? (r.eval_score * 100).toFixed(1) + "%" : "—";
    const isRandom = r.key === "random";
    return `<tr>
      <td><span class="arena-rank${rankCls}">${r.rank}</span></td>
      <td class="arena-pname" title="${r.name}">${shortKey(r.name)}</td>
      <td class="arena-pts">${r.points.toFixed(1)}</td>
      <td class="arena-wr">${wr}</td>
      <td class="arena-wdl">${r.wins}/${r.draws}/${r.losses}</td>
      <td class="arena-base">${isRandom ? "基准" : base}</td>
    </tr>`;
  }).join("");
}

export function clearArena() {
  state.arena.running = false;
  clearTimeout(state.arena.pollTimer);
  $("arena-board").hidden = true;
  $("arena-empty").hidden = false;
  $("arena-progress").hidden = true;
  $("btn-arena-start").disabled = false;
  $("arena-badge").textContent = "待开始";
  $("arena-badge").className = "chip chip-muted";
  $("arena-tbody").innerHTML = "";
}
