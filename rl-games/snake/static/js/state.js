/* state.js —— 贪吃蛇驾驶舱的共享状态(被 board / main 引用)。 */

export const state = {
  mode: "ai",          // ai | human
  speed: 1,            // 速度倍率(1/4/8)
  paused: false,
  gridSize: 11,
  gameOver: false,
  busy: false,         // 上一步请求进行中(防连发)
  dir: 3,              // 蛇"实际"移动方向(以服务端为准,每步后同步)
  dirQueue: [],        // 玩家按键方向队列(最多存 2 个,每节拍消费一个)。
                       // 单格缓冲会吞掉快速连按的转向(如"上+左"连按),
                       // 队列保证两次输入都能生效。
  last: null,          // 最近一次 step 的结果(驾驶舱数据)
  snapshot: null,      // 当前局面 + 观察 + Q 值
};

export const $ = (id) => document.getElementById(id);
