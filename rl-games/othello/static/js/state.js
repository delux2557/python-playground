/* state.js —— 黑白棋驾驶舱的共享状态(被 board / features / main 引用)。
   抽成独立模块是为了避免多文件之间的循环依赖:所有模块只从这里读状态。 */

export const state = {
  human: "black",                  // black | white | none(纯 AI 对战)
  snapshot: null,                  // 最新 /api/state(棋盘 + Q 值 + 观察)
  last: null,                      // 最近一次 step 的响应(决策 + 翻转数)
  gameOver: false,
  aiDelay: 400,                    // AI 落子间隔(毫秒)
  autoplay: true,                  // 纯 AI 对战模式:是否自动连下
  busy: false,                     // 防并发:一次只走一步
  timer: null,                     // 已排定的 AI 步
  anim: null,                      // 落子动画:{placed, flips, t0, duration}
  replay: {                        // 整局复盘回放(复用主棋盘)
    active: false, boards: [], moves: [], index: 0,
    playing: false, timer: null, title: "",
  },
  duel: { models: [], running: false, pollTimer: null },
  arena: { running: false, pollTimer: null },
  obsGridExpanded: false,          // 观察输入迷你网格是否展开
  metaModels: [],                  // 注册表模型列表(模型对战面板复用)
};

export const $ = (id) => document.getElementById(id);
