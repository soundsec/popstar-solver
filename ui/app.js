/* PopStar 统一页面：游玩 + 求解 + 图片识别。
 *
 * 布局约定：棋盘常驻左侧，任何时候都能直接点着玩；右侧用标签切换
 * 「游玩 / 分析」两套工具；图片识别是同页面的弹层，不跳转。
 *
 * 规则**一律由服务端裁决**——这里不实现任何游戏规则。连通块检测只用于
 * 高亮和可点判定，真正的消除都走 /api/apply。
 */

const PALETTE = [
  '#e5484d', '#3e63dd', '#46a758', '#f0b429',
  '#8e4ec6', '#e5832c', '#0d9488', '#d6409f',
];
const EMPTY = -1;
const V = window.Vision;

const $ = window.UI.$;
const api = window.UI.api;

const S = {
  // 棋盘与回放
  grid: null,
  initial: null,
  frames: null,
  path: null,
  step: 0,
  comps: [],
  timer: null,
  // 游玩
  history: [],
  moves: 0,
  best: null,
  hint: null,
  solution: null,
  // 分析
  pareto: null,
  profile: null,
  upperBound: null,
  generatorLabel: 'uniform',
  groupScore: 0,
  bonusScore: 0,
  // 图片识别
  img: null,
  image: null,
  rect: null,
  scanGrid: null,
  scanK: 5,
  clusters: [],
};

const scoring = () => ({});

/* =======================================================================
 * 棋盘渲染
 * ===================================================================== */

function components(grid) {
  const h = grid.length, w = grid[0].length;
  const seen = new Set();
  const out = [];
  const key = (r, c) => r * w + c;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = grid[r][c];
      if (v < 0 || seen.has(key(r, c))) continue;
      const stack = [[r, c]];
      const cells = [];
      seen.add(key(r, c));
      while (stack.length) {
        const [cr, cc] = stack.pop();
        cells.push([cr, cc]);
        for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          const nr = cr + dr, nc = cc + dc;
          if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
          if (grid[nr][nc] !== v || seen.has(key(nr, nc))) continue;
          seen.add(key(nr, nc));
          stack.push([nr, nc]);
        }
      }
      out.push({ color: v, size: cells.length, cells });
    }
  }
  return out;
}

function renderBoard() {
  const board = $('board');
  board.innerHTML = '';
  if (!S.grid) return;
  const h = S.grid.length, w = S.grid[0].length;
  board.style.gridTemplateColumns = `repeat(${w}, 1fr)`;
  /* 10×10 大约 72px 一格；更小的棋盘不要被拉成同样的总宽 */
  board.style.maxWidth = `${w * 72}px`;

  S.comps = components(S.grid);
  const compOf = new Map();
  for (const comp of S.comps) {
    for (const [r, c] of comp.cells) compOf.set(`${r},${c}`, comp);
  }
  window.__compOf = compOf;

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = S.grid[r][c];
      const el = document.createElement('div');
      el.className = 'cell';
      el.dataset.r = r; el.dataset.c = c;
      if (v < 0) {
        el.classList.add('empty');
      } else {
        el.style.background = PALETTE[v % PALETTE.length];
        el.textContent = String.fromCharCode(65 + (v % 26));
        const comp = compOf.get(`${r},${c}`);
        if (comp && comp.size >= 2) {
          el.classList.add('playable');
          el.dataset.comp = comp.cells.map(([cr, cc]) => `${cr},${cc}`).join(';');
        }
      }
      board.appendChild(el);
    }
  }
  // 分析回放优先：标出这一步即将消除的整块。没有回放时才用「提示」。
  const next = nextMoveGroup();
  const mark = next ? next.cells : S.hint;
  if (mark) {
    for (const [r, c] of mark) {
      const el = document.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
      if (el) el.classList.add('nextmove');
    }
  }
  refreshStats();
}

/** 当前回放步即将消除的连通块；走完或没有解时返回 null。 */
function nextMoveGroup() {
  if (!S.path || !S.grid || S.step >= S.path.length) return null;
  const target = S.path[S.step];
  const tr = Number(target[0]);
  const tc = Number(target[1]);
  return (S.comps || []).find((comp) =>
    comp.cells.some(([r, c]) => r === tr && c === tc)) || null;
}

function describePlayback() {
  if (!S.frames) return;
  const total = S.frames.length - 1;
  const next = nextMoveGroup();
  if (!next) {
    $('step-label').textContent = `第 ${S.step} / ${total} 步，没有下一步了`;
    return;
  }
  const target = S.path[S.step];
  $('step-label').textContent =
    `第 ${S.step} / ${total} 步 · 下一步消除 (${target[0]},${target[1]}) 所在 ${next.size} 块（棋盘上已标出）`;
}

function refreshStats() {
  if (!S.grid) return;
  let remaining = 0, left = 0;
  for (const row of S.grid) for (const v of row) if (v >= 0) remaining += 1;
  for (const comp of S.comps) if (comp.size >= 2) left += 1;
  $('s-remaining').textContent = remaining;
  $('s-left').textContent = left;
}

/* =======================================================================
 * 手动消除
 * ===================================================================== */

$('board').addEventListener('mouseover', (e) => {
  const cell = e.target.closest('.cell');
  if (!cell) return;
  document.querySelectorAll('.cell.hl').forEach((el) => el.classList.remove('hl'));
  const comp = cell.dataset.comp;
  if (!comp) return;
  for (const rc of comp.split(';')) {
    const [r, c] = rc.split(',');
    const el = document.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
    if (el) el.classList.add('hl');
  }
});

$('board').addEventListener('mouseleave', () => {
  document.querySelectorAll('.cell.hl').forEach((el) => el.classList.remove('hl'));
});

$('board').addEventListener('click', UI.guard(async (e) => {
  const cell = e.target.closest('.cell');
  if (!cell || !cell.classList.contains('playable')) return;
  S.hint = null;
  const res = await api('/api/apply', {
    grid: S.grid,
    cell: [Number(cell.dataset.r), Number(cell.dataset.c)],
    ...scoring(),
  });
  // 手动走子后，先前的解法回放不再对应当前局面
  S.history.push(S.grid);
  S.grid = res.grid;
  S.frames = null; S.path = null; S.solution = null; S.step = 0;
  S.moves += 1;
  $('gameover').hidden = true;
  renderBoard();
  // 服务端这次只知道「这一步」的分数。本局累计要在这里加上去，
  // 终局奖励用 total−group（两边都只含这一步的 group 分，相减即 bonus）。
  const bonus = res.finished ? (res.total - res.group) : 0;
  setPlayScore(S.groupScore + res.score, bonus);
  $('s-moves').textContent = S.moves;
  addLog(res.size, res.score);
  if (res.finished) finish(res);
  else $('step-label').textContent =
    `消除 ${res.size} 块，+${Math.round(res.score)} 分，本局累计 ${Math.round(S.groupScore)}（还剩 ${res.moves_left} 个可消块）`;
}, 'step-label'));

function setPlayScore(group, bonus) {
  S.groupScore = group;
  S.bonusScore = bonus;
  $('s-group').textContent = Math.round(group);
  $('s-bonus').textContent = Math.round(bonus);
  $('s-total').textContent = Math.round(group + bonus);
  updateTightness();
}

function addLog(size, score) {
  const li = document.createElement('li');
  li.dataset.score = String(score);
  li.textContent = `第 ${S.moves} 步：消除 ${size} 块，+${Math.round(score)} 分`;
  $('log').appendChild(li);
  $('log').scrollTop = $('log').scrollHeight;
}

function finish(res) {
  const total = S.groupScore + S.bonusScore;
  $('gameover').hidden = false;
  $('gameover').innerHTML =
    `<strong>本局结束</strong>：总分 <strong>${Math.round(total)}</strong>` +
    ` = group ${Math.round(S.groupScore)} + bonus ${Math.round(S.bonusScore)}，` +
    `剩余 ${res.remaining} 块，共 ${S.moves} 步` +
    (res.remaining === 0 ? ' —— 清盘！' : '');
  $('step-label').textContent = '已终局，无法继续消除。';
  if (!S.best || total > S.best.total) {
    S.best = {
      total,
      label: S.generatorLabel,
      seed: $('in-seed').value || '(随机)',
      remaining: res.remaining,
    };
  }
  renderBest();
}

function renderBest() {
  $('best').textContent = S.best
    ? `最高分 ${Math.round(S.best.total)}（${S.best.label}，种子 ${S.best.seed}，剩余 ${S.best.remaining}）`
    : '—';
}

/* =======================================================================
 * 新一局 / 撤销 / 重开 / 提示
 * ===================================================================== */

function adoptBoard(grid) {
  S.grid = grid;
  S.initial = grid.map((row) => row.slice());
  S.history = [];
  S.frames = null; S.path = null; S.solution = null; S.step = 0;
  S.hint = null;
  $('gameover').hidden = true;
}

function resetCounters() {
  S.moves = 0;
  S.groupScore = 0;
  S.bonusScore = 0;
  $('log').innerHTML = '';
  $('s-total').textContent = '0';
  $('s-group').textContent = '0';
  $('s-bonus').textContent = '0';
  $('s-moves').textContent = '0';
  $('pareto-table').querySelector('tbody').innerHTML = '';
  $('pareto-detail').textContent = '—';
  $('lambda-verdict').textContent = '—';
}

$('btn-random').addEventListener('click', UI.guard(async () => {
  const seedRaw = $('in-seed').value.trim();
  S.generatorLabel = $('in-generator').value;
  const res = await api('/api/board', {
    height: Number($('in-h').value),
    width: Number($('in-w').value),
    colors: Number($('in-colors').value),
    seed: seedRaw === '' ? null : Number(seedRaw),
    generator: S.generatorLabel,
    ...scoring(),
  });
  adoptBoard(res.grid);
  S.upperBound = res.upper_bound;
  S.profile = res.profile;
  S.pareto = null;
  resetCounters();
  renderBoard();
  renderProfile();
  renderBest();
  $('step-label').textContent = `新一局：${S.generatorLabel}，${res.moves} 个可消块`;
}, 'step-label'));

$('btn-restart').addEventListener('click', () => {
  if (!S.initial) return;
  adoptBoard(S.initial.map((row) => row.slice()));
  S.moves = 0;
  $('log').innerHTML = '';
  setPlayScore(0, 0);
  $('s-moves').textContent = '0';
  renderBoard();
  $('step-label').textContent = '已重开本局。';
  renderBest();
});

$('btn-undo').addEventListener('click', () => {
  if (!S.history.length) { $('step-label').textContent = '没有可撤销的步数。'; return; }
  S.grid = S.history.pop();
  S.moves = Math.max(0, S.moves - 1);
  S.hint = null;
  $('gameover').hidden = true;
  const list = $('log');
  if (list.lastChild) list.removeChild(list.lastChild);
  $('s-moves').textContent = S.moves;
  // 从剩余日志重算 group 分（避免再打一次服务端）
  let group = 0;
  for (const li of list.children) group += Number(li.dataset.score || 0);
  setPlayScore(group, 0);
  renderBoard();
  $('step-label').textContent = '已撤销一步。';
});

$('btn-hint').addEventListener('click', UI.guard(async () => {
  const btn = $('btn-hint');
  btn.disabled = true;
  $('step-label').textContent = '计算提示中…';
  try {
    const res = await api('/api/solve', {
      grid: S.grid, mode: 'fast', beam_width: 64, time_limit: 5, ...scoring(),
    });
    if (!res.moves.length) {
      $('step-label').textContent = '当前局面已无步可走。';
      S.hint = null;
    } else {
      const first = res.moves[0];
      const comp = components(S.grid).find((c) =>
        c.cells.some(([r, cc]) => r === first.cell[0] && cc === first.cell[1]));
      S.hint = comp ? comp.cells : [first.cell];
      $('step-label').textContent =
        `建议消除 (${first.cell[0]},${first.cell[1]}) 所在 ${first.size} 块，` +
        `+${first.score} 分；该解后续可得 ${Math.round(res.total)} 分（beam64，仅供参考）`;
    }
    renderBoard();
  } finally {
    btn.disabled = false;
  }
}, 'step-label'));

/* =======================================================================
 * 盘面画像与上界紧度
 * ===================================================================== */

function renderProfile() {
  const p = S.profile;
  const box = $('board-profile');
  if (!p) { box.textContent = '—'; return; }
  box.innerHTML =
    `生成方式 <strong>${S.generatorLabel}</strong> · ` +
    `色数分布 ${JSON.stringify(p.counts)} · ` +
    `数量均衡度 ${p.balance.toFixed(3)} · 空间聚集比 ${p.adjacency_ratio.toFixed(2)}` +
    `<br>上界 U(S₀) = ${Math.round(S.upperBound || 0)}<span id="tightness"></span>`;
  updateTightness();
}

function updateTightness() {
  const el = $('tightness');
  if (!el) return;
  const raw = $('s-total').textContent;
  const total = Number(raw);
  if (!S.upperBound || raw === '—' || !isFinite(total) || total <= 0) {
    el.textContent = '';
    return;
  }
  const ratio = S.upperBound / total;
  if (ratio <= 1.001) {
    el.innerHTML = ` · <span class="tight">U/V = ${ratio.toFixed(3)} → 已证明最优</span>`;
  } else if (ratio <= 1.05) {
    el.innerHTML =
      ` · <span class="tight">U/V = ${ratio.toFixed(3)} → 距最优 ≤${((ratio - 1) * 100).toFixed(1)}%</span>`;
  } else {
    el.innerHTML =
      ` · <span class="loose">U/V = ${ratio.toFixed(3)} → 上界松，无法证明</span>`;
  }
}

/* =======================================================================
 * 求解与回放
 * ===================================================================== */

$('btn-solve').addEventListener('click', UI.guard(async () => {
  const btn = $('btn-solve');
  btn.disabled = true;
  $('solve-status').textContent = '求解中…';
  try {
    const res = await api('/api/solve', {
      grid: S.initial || S.grid,
      mode: $('in-mode').value,
      beam_width: Number($('in-beam').value),
      time_limit: Number($('in-time').value),
      ...scoring(),
    });
    S.frames = res.frames;
    S.path = res.path;
    S.solution = res;
    S.step = 0;
    S.hint = null;
    S.grid = res.frames[0];
    $('gameover').hidden = true;
    renderBoard();
    showSolutionScore(0);
    describePlayback();
    $('solve-status').textContent =
      `${res.mode} · ${res.elapsed.toFixed(2)}s` +
      (res.proven ? ' · 已证明最优' : ' · best found');
    $('solve-detail').textContent =
      `展开 ${res.stats.states_expanded}  缓存命中 ${res.stats.cache_hits}` +
      `  缓存 ${res.stats.cache_size}  深度 ${res.stats.max_depth}\n` +
      `动作：${res.moves.map((m) => `(${m.cell[0]},${m.cell[1]})×${m.size}`).join(' ')}`;
  } finally {
    btn.disabled = false;
  }
}, 'solve-status'));

/** 按当前回放步数显示累计得分（随播放逐步增长）。 */
function showSolutionScore(k) {
  const res = S.solution;
  if (!res) return;
  let group = 0;
  const moves = res.moves || [];
  for (let i = 0; i < k && i < moves.length; i++) group += moves[i].score || 0;
  const done = k >= moves.length;
  const bonus = done ? (res.bonus || 0) : 0;
  setPlayScore(group, bonus);
  $('s-moves').textContent = k;
  updateTightness();
}

function gotoStep(k) {
  if (!S.frames) return;
  S.step = Math.max(0, Math.min(S.frames.length - 1, k));
  S.grid = S.frames[S.step];
  renderBoard();
  showSolutionScore(S.step);
  describePlayback();
}

$('btn-next').addEventListener('click', () => gotoStep(S.step + 1));
$('btn-prev').addEventListener('click', () => gotoStep(S.step - 1));
$('btn-reset').addEventListener('click', () => gotoStep(0));
$('btn-play').addEventListener('click', () => {
  if (S.timer) {
    clearInterval(S.timer); S.timer = null;
    $('btn-play').textContent = '自动播放';
    return;
  }
  if (!S.frames) return;
  $('btn-play').textContent = '暂停';
  S.timer = setInterval(() => {
    if (S.step >= S.frames.length - 1) {
      clearInterval(S.timer); S.timer = null;
      $('btn-play').textContent = '自动播放';
      return;
    }
    gotoStep(S.step + 1);
  }, 420);
});

/* =======================================================================
 * Pareto 前沿
 * ===================================================================== */

$('btn-pareto').addEventListener('click', UI.guard(async () => {
  const btn = $('btn-pareto');
  btn.disabled = true;
  $('pareto-status').textContent = '计算中（多组 profile，约数秒）…';
  try {
    const res = await api('/api/pareto', {
      grid: S.initial || S.grid,
      beam_width: Number($('in-pbeam').value),
      ...scoring(),
    });
    S.pareto = res;
    renderPareto();
    $('pareto-status').textContent =
      `${res.points.length} 个前沿点 · ${res.elapsed.toFixed(1)}s · ${res.expanded} 次展开`;
    $('pareto-detail').textContent =
      `U_group=${res.u_group.toFixed(0)}   U_nonclear=${res.u_nonclear.toFixed(0)}` +
      `   （非清盘上界比 U_group 小 ${(res.u_group - res.u_nonclear).toFixed(0)}）\n` +
      `profile：${res.profiles.join(', ')}\n` +
      `切换点：${res.switches.length
        ? res.switches.map((s) => `λ=${s.lam.toFixed(2)}: R${s.from}→R${s.to}`).join('  ')
        : '通行规则的奖励不是系数的倍数，前沿不再按 λ 切开'}`;
  } finally {
    btn.disabled = false;
  }
}, 'pareto-status'));

function renderPareto() {
  const res = S.pareto;
  if (!res) return;
  const best = bestAt();
  const tbody = $('pareto-table').querySelector('tbody');
  tbody.innerHTML = '';
  for (const p of res.points) {
    const tr = document.createElement('tr');
    if (p.pareto) tr.classList.add('pareto');
    if (best && p.r === best.r && Math.abs(p.g - best.g) < 1e-9) tr.classList.add('best');
    tr.innerHTML =
      `<td>${p.r}</td><td>${p.g.toFixed(0)}</td><td>${(p.eta * 100).toFixed(0)}%</td>` +
      `<td>${p.bonus == null ? '—' : Number(p.bonus).toFixed(0)}</td>` +
      `<td>${p.pareto ? '<span class="tag">是</span>' : '<span class="tag muted">被支配</span>'}</td>` +
      `<td><button data-r="${p.r}">载入</button></td>`;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', UI.guard(
      () => loadFrontier(Number(btn.dataset.r)), 'pareto-status'));
  });
  updateVerdict();
}

function bestAt() {
  const res = S.pareto;
  if (!res || !res.points.length) return null;
  let best = null;
  for (const p of res.points) {
    const bonus = Number(p.bonus) || 0;
    const total = p.g + bonus;
    if (!best || total > best.total + 1e-9 ||
        (Math.abs(total - best.total) <= 1e-9 && p.r < best.r)) {
      best = { r: p.r, g: p.g, bonus, total };
    }
  }
  return best;
}

function updateVerdict() {
  const res = S.pareto;
  if (!res) return;
  const best = bestAt();
  if (!best) { $('lambda-verdict').textContent = '—'; return; }
  $('lambda-verdict').innerHTML =
    `通行规则下最优终局 <strong>R = ${best.r}</strong>，` +
    `group = ${best.g.toFixed(0)}，bonus = ${best.bonus.toFixed(0)}，` +
    `总分 <strong>${best.total.toFixed(0)}</strong>` +
    (best.r === 0 ? '（清盘）' : '（未清盘）');
}

async function loadFrontier(r) {
  const res = S.pareto;
  const path = res.paths[String(r)];
  if (!path) return;
  $('pareto-status').textContent = `载入 R=${r} 的解…`;
  const rep = await api('/api/replay', {
    grid: S.initial || S.grid, path, ...scoring(),
  });
  S.frames = rep.frames; S.path = path; S.step = 0;
  // 前沿解只拿到总分拆分，逐步得分未知，回放时按最终值显示
  S.solution = {
    moves: path.map((c) => ({ cell: c, size: 0, score: 0 })),
    bonus: rep.bonus, mode: 'frontier',
  };
  S.grid = rep.frames[0];
  $('gameover').hidden = true;
  renderBoard();
  describePlayback();
  setPlayScore(rep.group, rep.bonus);
  $('s-moves').textContent = rep.move_count;
  updateTightness();
  $('pareto-status').textContent =
    `已载入 R=${r} 的解（${rep.move_count} 步，总分 ${Math.round(rep.total)}）`;
}

/* =======================================================================
 * 图片识别弹层
 * ===================================================================== */

function openImport() {
  $('import-modal').hidden = false;
  if (S.image) drawScan();
}

function closeImport() { $('import-modal').hidden = true; }

$('btn-import').addEventListener('click', openImport);
$('btn-import-close').addEventListener('click', closeImport);
$('import-modal').addEventListener('click', (e) => {
  if (e.target === $('import-modal')) closeImport();   // 点遮罩关闭
});
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('import-modal').hidden) closeImport();
});

function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    $('scan-status').textContent = '请选择图片文件。';
    return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    S.img = img;
    const nat = document.createElement('canvas');
    nat.width = img.naturalWidth;
    nat.height = img.naturalHeight;
    const ctx = nat.getContext('2d');
    ctx.drawImage(img, 0, 0);
    S.image = {
      data: ctx.getImageData(0, 0, nat.width, nat.height).data,
      width: nat.width, height: nat.height,
    };
    S.rect = null;
    S.scanGrid = null;
    $('preview').innerHTML = '';
    $('palette').innerHTML = '';
    drawScan();
    autoDetect();
    $('scan-status').textContent = S.rect
      ? `已载入 ${nat.width}×${nat.height}，自动框出 ${S.rect.w}×${S.rect.h}；确认后点「识别棋盘」。`
      : `已载入 ${nat.width}×${nat.height}；请手动在图上拖动框选棋盘。`;
  };
  img.onerror = () => { $('scan-status').textContent = '图片解码失败。'; };
  img.src = url;
}

$('file').addEventListener('change', (e) => loadFile(e.target.files[0]));

const drop = $('drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('hot'); });
drop.addEventListener('dragleave', () => drop.classList.remove('hot'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('hot');
  loadFile(e.dataTransfer.files[0]);
});
window.addEventListener('paste', (e) => {
  for (const item of (e.clipboardData || {}).items || []) {
    if (item.type.startsWith('image/')) {
      openImport();
      loadFile(item.getAsFile());
      break;
    }
  }
});

let dragFrom = null;

function drawScan() {
  const canvas = $('canvas');
  if (!S.img) { canvas.width = 0; canvas.height = 0; return; }
  const scale = Math.min(1, 520 / S.img.naturalWidth);
  const w = Math.round(S.img.naturalWidth * scale);
  const h = Math.round(S.img.naturalHeight * scale);
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(S.img, 0, 0, w, h);

  if (S.rect) {
    const r = S.rect;
    ctx.strokeStyle = '#1a7f37';
    ctx.lineWidth = 3;
    ctx.strokeRect(r.x * scale, r.y * scale, r.w * scale, r.h * scale);
    const rows = Number($('in-rows').value);
    const cols = Number($('in-cols').value);
    ctx.strokeStyle = 'rgba(26,127,55,.55)';
    ctx.lineWidth = 1;
    for (let i = 1; i < cols; i++) {
      const x = (r.x + (r.w * i) / cols) * scale;
      ctx.beginPath(); ctx.moveTo(x, r.y * scale); ctx.lineTo(x, (r.y + r.h) * scale); ctx.stroke();
    }
    for (let j = 1; j < rows; j++) {
      const y = (r.y + (r.h * j) / rows) * scale;
      ctx.beginPath(); ctx.moveTo(r.x * scale, y); ctx.lineTo((r.x + r.w) * scale, y); ctx.stroke();
    }
  }
}

function scanPoint(e) {
  const canvas = $('canvas');
  const box = canvas.getBoundingClientRect();
  const scale = canvas.width / S.img.naturalWidth;
  return {
    x: ((e.clientX - box.left) / box.width) * canvas.width / scale,
    y: ((e.clientY - box.top) / box.height) * canvas.height / scale,
  };
}

$('canvas').addEventListener('mousedown', (e) => {
  if (!S.img) return;
  dragFrom = scanPoint(e);
});
$('canvas').addEventListener('mousemove', (e) => {
  if (!dragFrom || !S.img) return;
  const p = scanPoint(e);
  S.rect = {
    x: Math.round(Math.min(dragFrom.x, p.x)),
    y: Math.round(Math.min(dragFrom.y, p.y)),
    w: Math.round(Math.abs(p.x - dragFrom.x)),
    h: Math.round(Math.abs(p.y - dragFrom.y)),
  };
  drawScan();
});
window.addEventListener('mouseup', () => {
  if (dragFrom && S.rect) {
    S.rectManual = true;
    $('scan-status').textContent = `已框选 ${S.rect.w}×${S.rect.h}，点「识别棋盘」。`;
  }
  dragFrom = null;
});

function autoDetect() {
  if (!S.image) return;
  S.rectManual = false;
  S.rect = V.detectBoardRect(S.image, 55, {
    rows: Number($('in-rows').value) || 10,
    cols: Number($('in-cols').value) || 10,
  });
  drawScan();
}

$('btn-detect').addEventListener('click', UI.guard(() => {
  autoDetect();
  $('scan-status').textContent = S.rect
    ? `已框选 ${S.rect.w}×${S.rect.h}。`
    : '自动框选失败，请手动拖动。';
}, 'scan-status'));

function extractScan() {
  if (!S.image || !S.rect) {
    $('scan-status').textContent = '请先载入图片并框选棋盘区域。';
    return;
  }
  const rows = Number($('in-rows').value);
  const cols = Number($('in-cols').value);
  const out = V.extractGrid(S.image, S.rect, {
    rows, cols,
    k: Number($('in-k').value),
    emptyThreshold: Number($('in-thr').value),
  });
  S.scanGrid = out.grid;
  S.scanK = out.clusters.length;
  S.clusters = out.clusters;
  renderScanPreview();

  const r = S.rect;
  $('scan-status').textContent =
    `识别完成：${rows}×${cols}，${out.clusters.length} 种颜色，空格 ${out.empties.length} 个。`;
  $('scan-detail').textContent =
    `区域 ${r.w}×${r.h} @ (${r.x},${r.y})   格子 ${(r.w / cols).toFixed(1)}×${(r.h / rows).toFixed(1)} px\n` +
    `背景 rgb(${out.background.map(Math.round).join(',')})   空格阈值 ${$('in-thr').value}\n` +
    '聚类中心：\n' +
    out.clusters.map((c, i) =>
      `  #${i} rgb(${c.centroid.map(Math.round).join(',')}) ×${c.count}`).join('\n') +
    `\n非空格子 ${rows * cols - out.empties.length} 个。点击格子可修正。`;
}

$('btn-extract').addEventListener('click', UI.guard(extractScan, 'scan-status'));
$('in-thr').addEventListener('input', UI.guard(() => {
  $('out-thr').textContent = $('in-thr').value;
  if (S.scanGrid) extractScan();
}, 'scan-status'));
for (const id of ['in-rows', 'in-cols', 'in-k']) {
  $(id).addEventListener('change', UI.guard(() => {
    // 没手动拖过框时，行/列变化要按新的尺寸重新拟合（残局会往上补空行）
    if (!S.rectManual) autoDetect();
    else drawScan();
    if (S.scanGrid) extractScan();
  }, 'scan-status'));
}

function renderScanPreview() {
  const box = $('preview');
  box.innerHTML = '';
  if (!S.scanGrid) return;
  const rows = S.scanGrid.length, cols = S.scanGrid[0].length;
  box.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  box.style.maxWidth = `${Math.min(520, cols * 46)}px`;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = S.scanGrid[r][c];
      const el = document.createElement('div');
      el.className = 'cell' + (v === EMPTY ? ' empty' : ' playable');
      el.dataset.r = r; el.dataset.c = c;
      if (v !== EMPTY) {
        const rgb = (S.clusters[v] && S.clusters[v].centroid) || [180, 180, 180];
        const L = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        el.style.background = `rgb(${rgb.map((n) => Math.round(n)).join(',')})`;
        el.style.color = L > 165 ? '#1b1f24' : '#fff';
        el.textContent = String.fromCharCode(65 + (v % 26));
      }
      box.appendChild(el);
    }
  }
  const pal = $('palette');
  pal.innerHTML = '';
  S.clusters.forEach((cl, i) => {
    const sw = document.createElement('span');
    sw.className = 'swatch';
    const rgb = cl.centroid.map((n) => Math.round(n)).join(',');
    sw.innerHTML = `<i style="background:rgb(${rgb})"></i>#${i} ×${cl.count}`;
    pal.appendChild(sw);
  });
  const sw = document.createElement('span');
  sw.className = 'swatch';
  sw.innerHTML = '<i style="background:#f0f2f5"></i>空格';
  pal.appendChild(sw);
}

// 点击循环切换颜色：0 → … → k−1 → 空格 → 0
$('preview').addEventListener('click', (e) => {
  const cell = e.target.closest('.cell');
  if (!cell || !S.scanGrid) return;
  const r = Number(cell.dataset.r), c = Number(cell.dataset.c);
  const v = S.scanGrid[r][c];
  S.scanGrid[r][c] = (v === EMPTY) ? 0 : (v + 1 >= S.scanK ? EMPTY : v + 1);
  renderScanPreview();
});

$('btn-load-board').addEventListener('click', UI.guard(async () => {
  if (!S.scanGrid) { $('import-status').textContent = '请先识别棋盘。'; return; }
  $('import-status').textContent = '载入中…';
  const res = await api('/api/board', { grid: S.scanGrid, ...scoring() });
  S.generatorLabel = '图片导入';
  adoptBoard(res.grid);
  S.upperBound = res.upper_bound;
  S.profile = res.profile;
  S.pareto = null;
  resetCounters();
  $('in-h').value = res.grid.length;
  $('in-w').value = res.grid[0].length;
  renderBoard();
  renderProfile();
  renderBest();
  closeImport();
  $('step-label').textContent = `已载入图片中的棋盘，${res.moves} 个可消块`;
}, 'import-status'));

/* =======================================================================
 * 标签切换
 * ===================================================================== */

function switchTab(name) {
  for (const btn of document.querySelectorAll('.tab')) {
    btn.classList.toggle('active', btn.dataset.tab === name);
  }
  $('tab-play').hidden = name !== 'play';
  $('tab-analyze').hidden = name !== 'analyze';
}

for (const btn of document.querySelectorAll('.tab')) {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
}

/* =======================================================================
 * 启动
 * ===================================================================== */

(async function boot() {
  if (!(await UI.ping())) return;   // 服务不在就先提示，别让后续点击静默失败

  const params = new URLSearchParams(location.search);

  // ?board=<id>：从别处（或旧链接）带过来的棋盘
  const boardId = params.get('board');
  if (boardId) {
    try {
      const resp = await fetch(`/api/board/${encodeURIComponent(boardId)}`);
      const data = await resp.json();
      if (!data.error) {
        S.generatorLabel = '导入棋盘';
        adoptBoard(data.grid);
        S.upperBound = null;
        S.profile = null;
        $('in-h').value = data.height;
        $('in-w').value = data.width;
        renderBoard();
        $('step-label').textContent =
          `已载入导入的棋盘（${data.height}×${data.width}），${data.moves} 个可消块`;
      }
    } catch (err) {
      $('step-label').textContent = `载入失败：${err.message}`;
    }
  } else {
    $('btn-random').click();
  }

  const tab = params.get('tab');
  if (tab === 'analyze') switchTab('analyze');
  if (tab === 'scan') openImport();
})();
