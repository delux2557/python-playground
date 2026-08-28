/* =========================================================================
   common.js —— 前端共享层(所有游戏驾驶舱共用)
   消除两个游戏 app.js 之间"逐行复制"的通用逻辑:
     1. 会话 ID(SESSION_ID):每个标签页一局独立的棋
     2. api():fetch 封装,自动带 X-Session-Id 头
     3. setupCanvas():高清屏画布初始化
     4. drawLine / drawCurve:训练曲线绘制
     5. renderConfig / renderModelPicker:训练参数面板 + 模型切换下拉
     6. startTrainPolling:训练状态自适应轮询(训练中高频/空闲降频)
   ========================================================================= */

"use strict";

/* ---------------- 会话 ID:每个标签页一局独立的棋 ----------------
   用 sessionStorage(每个标签页各自独立)存随机会话 ID,所有请求都带上
   X-Session-Id 头——服务端按它隔离对局,多个标签页各玩各的。
   关键细节:浏览器"复制标签页"时会把 sessionStorage 一并复制过去,
   两个标签页若共用同一个会话 ID 就会操作同一盘棋、互相抢步。
   所以只有"刷新"(reload)才沿用原 ID 续局,其余导航方式(新开/复制
   标签页等)一律重新签发新 ID。 */
export const SESSION_ID = (() => {
  const nav = performance.getEntriesByType("navigation")[0];
  const isReload = nav ? nav.type === "reload"
    : (performance.navigation && performance.navigation.type === 1);
  let id = isReload ? sessionStorage.getItem("session-id") : null;
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID()
          : "s-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10));
    sessionStorage.setItem("session-id", id);
  }
  return id;
})();

/* ---------------- 请求后端 ---------------- */
export async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json",
               "X-Session-Id": SESSION_ID, ...(opts.headers || {}) },
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

/* ---------------- 画布初始化(支持高清屏) ---------------- */
export function setupCanvas(canvas, cssSize) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssSize * dpr;
  canvas.height = cssSize * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return ctx;
}

/* ---------------- 训练曲线绘制 ----------------
   两个游戏共用同一套坐标映射/网格轴,差异只有:
     · 空数据提示文案(emptyText)、底部说明(footerText)
     · 是否画"累计平均"线(showAvg)与"历史最高"线(showMax)
   调用方把画布与曲线数据传进来,数据缓存由调用方管理(resize 时重绘)。 */
export function drawLine(ctx, pts, color, width) {
  if (pts.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();
}

export function drawCurve(canvas, data, opts = {}) {
  const {
    emptyText = "暂无训练数据",
    footerText = "",
    showAvg = true,
    showMax = false,
    avgColor = "#fbbf24",
    maxColor = "#2ee6a8",
    pointColor = "#7c6cff",
  } = opts;
  const wrap = canvas.parentElement;
  const cssW = wrap.clientWidth || 560;
  const cssH = 220;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const episodes = data.episodes || [];
  const scores = data.scores || [];
  if (!episodes.length) {
    ctx.fillStyle = "#8b93a7";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(emptyText, cssW / 2, cssH / 2);
    return;
  }

  const padL = 34, padR = 12, padT = 14, padB = 24;
  const plotW = cssW - padL - padR, plotH = cssH - padT - padB;
  const maxX = Math.max(...episodes);
  const maxY = Math.max(...scores, 1);

  // 坐标轴与网格
  ctx.strokeStyle = "rgba(148,163,184,.12)";
  ctx.fillStyle = "#8b93a7";
  ctx.font = "10px sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH - (plotH * i) / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(cssW - padR, y); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(((maxY * i) / 4).toFixed(0), padL - 5, y + 3);
  }
  const x = (ep) => padL + (ep / maxX) * plotW;
  const y = (s) => padT + plotH - (s / maxY) * plotH;

  // 辅助线:历史最高(可选,贪吃蛇用) / 累计平均(比原始折线更能看出趋势)
  if (showMax) {
    let cumMax = -Infinity;
    const maxLine = [];
    scores.forEach((s, i) => { cumMax = Math.max(cumMax, s); maxLine.push([x(episodes[i]), y(cumMax)]); });
    drawLine(ctx, maxLine, maxColor, 1.6);
  }
  if (showAvg) {
    let cumSum = 0;
    const avgLine = [];
    scores.forEach((s, i) => { cumSum += s; avgLine.push([x(episodes[i]), y(cumSum / (i + 1))]); });
    drawLine(ctx, avgLine, avgColor, 1.6);
  }

  // 原始点
  ctx.fillStyle = pointColor;
  scores.forEach((s, i) => {
    ctx.beginPath(); ctx.arc(x(episodes[i]), y(s), 2.4, 0, Math.PI * 2); ctx.fill();
  });

  if (footerText) {
    ctx.textAlign = "left";
    ctx.fillStyle = "#8b93a7";
    ctx.fillText(footerText, padL, cssH - 6);
  }
}

/* ---------------- 训练参数面板 ---------------- */
export function renderConfig(container, items) {
  container.innerHTML = items
    .map(([k, v]) => `<div class="config-item"><div class="cfg-label">${k}</div><div class="cfg-val">${v}</div></div>`)
    .join("");
}

/* ---------------- 模型切换下拉(平台化的"换模型") ----------------
   返回 models 列表,方便调用方缓存(如模型对战面板要复用)。 */
export function renderModelPicker(sel, models, currentName) {
  sel.innerHTML = "";
  if (!models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "暂无已登记模型(先跑训练)";
    opt.disabled = true;
    sel.appendChild(opt);
    return models;
  }
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.key;
    const score = m.eval_score != null ? ` · 评估 ${m.eval_score}` : "";
    opt.textContent = `${m.key}${score}`;
    sel.appendChild(opt);
  });
  // 默认选中当前加载的那个模型:按路径反查注册表
  const cur = currentName || "";
  const hit = models.find((m) => m.path === cur)
    || models.find((m) => cur.includes(m.path) || m.path.includes(cur));
  if (hit) sel.value = hit.key;
  return models;
}

/* ---------------- 训练状态自适应轮询 ----------------
   pollFn: 每次轮询要做的事(拉状态 + 渲染),返回 true = 训练在跑(保持高频),
           否则降频。训练中每 fastMs 轮询一次,空闲降到 idleMs——
           避免"训练结束后仍终生每 2 秒打一次请求"。 */
export function startTrainPolling(pollFn, { fastMs = 2000, idleMs = 15000 } = {}) {
  let timer = null;
  let fast = true;
  const tick = async () => {
    const needFast = await pollFn();
    if (needFast !== fast) { fast = needFast; restart(); }
  };
  const restart = () => {
    if (timer) clearInterval(timer);
    timer = setInterval(tick, fast ? fastMs : idleMs);
  };
  restart();
  return {
    stop() { if (timer) clearInterval(timer); timer = null; },
  };
}
