/* 图片 → 棋盘的核心识别逻辑（**不依赖 DOM**，可在 Node 里直接测试）。
 *
 * 抽成独立模块的原因：识别正确性必须能被自动验证，而不是只能靠肉眼看。
 * 这样可以用合成图片（已知真值）跑回归测试。
 *
 * 实机截图不能靠「非背景像素的包围盒」：顶栏、道具按钮和底部广告都会被框进来。
 * 这里改成先找大小一致的方块，再拟合等间距格子；空行补在顶部，空列补在右侧。
 * 取色用方块本体的环带（避开中心星星），按色相分成最多 k 种颜色。
 *
 * 输入统一为 ``{data: Uint8ClampedArray(RGBA), width, height}``。
 */
(function (root) {
  'use strict';

  const EMPTY = -1;

  function makeImage(width, height, fill) {
    const data = new Uint8ClampedArray(width * height * 4);
    if (fill) {
      for (let i = 0; i < width * height; i++) {
        data[i * 4] = fill[0];
        data[i * 4 + 1] = fill[1];
        data[i * 4 + 2] = fill[2];
        data[i * 4 + 3] = 255;
      }
    }
    return { data, width, height };
  }

  const dist = (a, b) =>
    Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);

  // ITU-R BT.601 亮度权重，用来区分亮色方块和暗背景。
  const lum = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];

  function medianRGB(px) {
    if (!px.length) return [0, 0, 0];
    const rs = px.map((p) => p[0]).sort((a, b) => a - b);
    const gs = px.map((p) => p[1]).sort((a, b) => a - b);
    const bs = px.map((p) => p[2]).sort((a, b) => a - b);
    const mid = px.length >> 1;
    return px.length % 2
      ? [rs[mid], gs[mid], bs[mid]]
      : [
          (rs[mid - 1] + rs[mid]) / 2,
          (gs[mid - 1] + gs[mid]) / 2,
          (bs[mid - 1] + bs[mid]) / 2,
        ];
  }

  /** 最常见颜色（量化后计数），用作背景色兜底。 */
  function dominantColor(image) {
    const { data, width: w, height: h } = image;
    const counter = new Map();
    const step = Math.max(1, Math.round(Math.min(w, h) / 120));
    for (let y = 0; y < h; y += step) {
      for (let x = 0; x < w; x += step) {
        const i = (y * w + x) * 4;
        const key = ((data[i] >> 4) << 8) | ((data[i + 1] >> 4) << 4) | (data[i + 2] >> 4);
        let cell = counter.get(key);
        if (!cell) { cell = [0, 0, 0, 0]; counter.set(key, cell); }
        cell[0] += 1; cell[1] += data[i]; cell[2] += data[i + 1]; cell[3] += data[i + 2];
      }
    }
    let best = null;
    for (const cell of counter.values()) if (!best || cell[0] > best[0]) best = cell;
    return [best[1] / best[0], best[2] / best[0], best[3] / best[0]];
  }

  /** 四角小块的中位色；四角彼此差异过大时返回 null（表示不可信）。 */
  function cornerColor(image) {
    const { data, width: w, height: h } = image;
    const patch = (cx, cy) => {
      const px = [];
      const n = Math.max(4, Math.round(Math.min(w, h) * 0.03));
      for (let y = cy; y < Math.min(h, cy + n); y++) {
        for (let x = cx; x < Math.min(w, cx + n); x++) {
          const i = (y * w + x) * 4;
          px.push([data[i], data[i + 1], data[i + 2]]);
        }
      }
      return px.length ? medianRGB(px) : null;
    };
    const n = Math.max(4, Math.round(Math.min(w, h) * 0.03));
    const corners = [
      patch(0, 0), patch(Math.max(0, w - n), 0),
      patch(0, Math.max(0, h - n)), patch(Math.max(0, w - n), Math.max(0, h - n)),
    ].filter(Boolean);
    if (corners.length < 4) return null;
    const spread = Math.max(...corners.map((c) => dist(c, corners[0])));
    return spread > 90 ? null : corners[0];
  }

  /** 背景色：优先信四角，否则退化为最常见颜色。 */
  function backgroundOf(image) {
    return cornerColor(image) || dominantColor(image);
  }

  const satOf = (c) => {
    const max = Math.max(c[0], c[1], c[2]);
    const min = Math.min(c[0], c[1], c[2]);
    return max > 0 ? (max - min) / max : 0;
  };

  const valueOf = (c) => Math.max(c[0], c[1], c[2]) / 255;

  /** 色相（度）。红在 0° 两侧，比较时必须走环形距离。 */
  function rgbToHue(c) {
    const r = c[0] / 255, g = c[1] / 255, b = c[2] / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    if (d < 1e-6) return 0;
    let h;
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    if (h < 0) h += 6;
    return h * 60;
  }

  function hueDist(a, b) {
    const d = Math.abs(a - b) % 360;
    return Math.min(d, 360 - d);
  }

  /**
   * 饱和色用色相距离（红/黄/绿/蓝/紫靠色相分开，明暗和星星高光不影响），
   * 灰暗色退回 RGB 距离。threshold 对色相表示「度」，对 RGB 表示欧氏距离。
   */
  function colorDist(a, b) {
    if (satOf(a) > 0.2 && satOf(b) > 0.2) return hueDist(rgbToHue(a), rgbToHue(b));
    return dist(a, b);
  }

  /** 同一格子里若混入了第二种颜色（例如中央的播放按钮），只保留色相最多的那簇。 */
  function dominantRGB(pixels) {
    if (pixels.length < 8) return medianRGB(pixels);
    const nbin = 24;
    const buckets = new Array(nbin);
    for (let i = 0; i < nbin; i++) buckets[i] = [];
    let used = 0;
    for (const px of pixels) {
      if (satOf(px) < 0.22) continue;
      const bin = Math.floor(rgbToHue(px) / (360 / nbin)) % nbin;
      buckets[bin].push(px);
      used += 1;
    }
    if (used < 8) return medianRGB(pixels);
    let best = 0;
    for (let i = 1; i < nbin; i++) {
      if (buckets[i].length > buckets[best].length) best = i;
    }
    const chosen = buckets[best].concat(
      buckets[(best + nbin - 1) % nbin],
      buckets[(best + 1) % nbin],
    );
    return medianRGB(chosen.length ? chosen : pixels);
  }

  function medianNum(nums) {
    if (!nums.length) return 0;
    const a = nums.slice().sort((x, y) => x - y);
    const m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }

  /** 旧的包围盒检测。实机截图上界面/广告会把框撑满全图，只作兜底。 */
  function detectBoardRectByBackground(image, bgThreshold) {
    const { data, width: w, height: h } = image;
    const bg = backgroundOf(image);
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        if (dist([data[i], data[i + 1], data[i + 2]], bg) > bgThreshold) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
    return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
  }

  /**
   * 找出大小一致的方块中心。
   * 实机图里方块是高饱和、偏亮的圆角矩形；按钮和广告大小不同，星空光点则小得多。
   */
  function findTileCenters(image) {
    const { data, width: W, height: H } = image;
    const step = Math.max(1, Math.min(4, Math.round(Math.min(W, H) / 480)));
    const mw = Math.floor(W / step);
    const mh = Math.floor(H / step);
    if (mw < 4 || mh < 4) return [];
    const mask = new Uint8Array(mw * mh);
    for (let y = 0; y < mh; y++) {
      const sy = y * step;
      for (let x = 0; x < mw; x++) {
        const i = (sy * W + x * step) * 4;
        const rgb = [data[i], data[i + 1], data[i + 2]];
        // 方块又亮又饱和；夜空虽饱和但很暗，星星高光偏白。
        if (satOf(rgb) > 0.42 && valueOf(rgb) > 0.40 && valueOf(rgb) < 0.985) {
          mask[y * mw + x] = 1;
        }
      }
    }

    const seen = new Uint8Array(mw * mh);
    const stack = [];
    const comps = [];
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        const start = y * mw + x;
        if (!mask[start] || seen[start]) continue;
        let minX = x, maxX = x, minY = y, maxY = y, area = 0;
        stack.push(start);
        seen[start] = 1;
        while (stack.length) {
          const p = stack.pop();
          const py = (p / mw) | 0;
          const px = p - py * mw;
          area += 1;
          if (px < minX) minX = px;
          if (px > maxX) maxX = px;
          if (py < minY) minY = py;
          if (py > maxY) maxY = py;
          if (px > 0 && mask[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack.push(p - 1); }
          if (px + 1 < mw && mask[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack.push(p + 1); }
          if (py > 0 && mask[p - mw] && !seen[p - mw]) { seen[p - mw] = 1; stack.push(p - mw); }
          if (py + 1 < mh && mask[p + mw] && !seen[p + mw]) { seen[p + mw] = 1; stack.push(p + mw); }
        }
        const bw = (maxX - minX + 1) * step;
        const bh = (maxY - minY + 1) * step;
        comps.push({
          area,
          w: bw,
          h: bh,
          x: (minX + maxX + 1) * step / 2,
          y: (minY + maxY + 1) * step / 2,
        });
      }
    }

    const square = comps.filter((c) =>
      c.w >= 6 && c.h >= 6 && c.w / c.h > 0.65 && c.h / c.w > 0.65);
    if (square.length < 8) return [];
    const maxArea = square.reduce((m, c) => Math.max(m, c.area), 0);
    const pool = square.filter((c) => c.area >= maxArea * 0.35);
    if (pool.length < 8) return [];

    let bestN = -1, bestW = pool[0].w, bestH = pool[0].h;
    for (const c of pool) {
      let n = 0;
      for (const o of pool) {
        if (Math.abs(o.w - c.w) <= c.w * 0.14 && Math.abs(o.h - c.h) <= c.h * 0.14) n += 1;
      }
      if (n > bestN) { bestN = n; bestW = c.w; bestH = c.h; }
    }
    return pool.filter((o) =>
      Math.abs(o.w - bestW) <= bestW * 0.14 && Math.abs(o.h - bestH) <= bestH * 0.14);
  }

  function nearestPitch(points) {
    const ds = [];
    for (let i = 0; i < points.length; i++) {
      let best = Infinity;
      for (let j = 0; j < points.length; j++) {
        if (i === j) continue;
        const d = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
        if (d > 4 && d < best) best = d;
      }
      if (best < Infinity) ds.push(best);
    }
    return medianNum(ds);
  }

  function mergeClose(points, minDist) {
    const used = new Array(points.length).fill(false);
    const out = [];
    for (let i = 0; i < points.length; i++) {
      if (used[i]) continue;
      let sx = points[i].x, sy = points[i].y, n = 1;
      used[i] = true;
      for (let j = i + 1; j < points.length; j++) {
        if (used[j]) continue;
        if (Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y) <= minDist) {
          used[j] = true;
          sx += points[j].x;
          sy += points[j].y;
          n += 1;
        }
      }
      out.push({ x: sx / n, y: sy / n });
    }
    return out;
  }

  function groupSorted(values, gap) {
    const sorted = values.slice().sort((a, b) => a - b);
    const groups = [];
    for (const v of sorted) {
      const g = groups[groups.length - 1];
      if (!g || v - g.values[g.values.length - 1] > gap) groups.push({ values: [v] });
      else g.values.push(v);
    }
    return groups.map((g) => ({ center: medianNum(g.values), n: g.values.length }));
  }

  /** 间距约等于 pitch 的最长一段。棋盘在这段里，零散按钮不在。 */
  function bestChain(groups, pitch) {
    let best = null;
    let cur = [];
    let score = 0;
    const flush = () => {
      if (cur.length && (!best || score > best.score)) best = { score, groups: cur.slice() };
    };
    for (const g of groups) {
      if (!cur.length) { cur = [g]; score = g.n; continue; }
      const gap = g.center - cur[cur.length - 1].center;
      if (gap > pitch * 0.55 && gap < pitch * 1.5) {
        cur.push(g);
        score += g.n;
      } else {
        flush();
        cur = [g];
        score = g.n;
      }
    }
    flush();
    return best;
  }

  function medianDiff(centers) {
    if (centers.length < 2) return null;
    const d = [];
    for (let i = 1; i < centers.length; i++) d.push(centers[i] - centers[i - 1]);
    return medianNum(d);
  }

  /**
   * 用方块格子拟合棋盘，而不是「所有非背景像素的包围盒」。
   * 后者会把顶栏、道具按钮、底部广告一起框进来。
   *
   * 重力让空行出现在顶部、空列出现在右侧，所以不足 rows/cols 时往这两个方向补。
   */
  function fitBoardLattice(image, options) {
    const opts = options || {};
    let tiles = findTileCenters(image);
    if (tiles.length < 8) return null;
    let pitch = nearestPitch(tiles);
    if (!(pitch > 6)) return null;
    tiles = mergeClose(tiles, pitch * 0.45);
    pitch = nearestPitch(tiles);
    if (!(pitch > 6) || tiles.length < 8) return null;

    const rowGroups = groupSorted(tiles.map((t) => t.y), pitch * 0.45);
    const rowChain = bestChain(rowGroups, pitch);
    if (!rowChain || rowChain.score < 8 || rowChain.groups.length < 2) return null;
    const rowCenters0 = rowChain.groups.map((g) => g.center);

    const onBoard = tiles.filter((t) =>
      rowCenters0.some((cy) => Math.abs(t.y - cy) < pitch * 0.45));
    const colGroups = groupSorted(onBoard.map((t) => t.x), pitch * 0.45);
    const colChain = bestChain(colGroups, pitch);
    if (!colChain || colChain.groups.length < 2) return null;

    let rowCenters = rowCenters0.slice();
    let colCenters = colChain.groups.map((g) => g.center);
    let pitchY = medianDiff(rowCenters) || pitch;
    let pitchX = medianDiff(colCenters) || pitch;
    if (Math.abs(pitchX - pitchY) / Math.max(pitchX, pitchY) > 0.35) return null;

    const wantRows = Number(opts.rows) > 1 ? Number(opts.rows) : rowCenters.length;
    const wantCols = Number(opts.cols) > 1 ? Number(opts.cols) : colCenters.length;
    // 空行在顶部（方块向下落）
    while (rowCenters.length < wantRows) rowCenters.unshift(rowCenters[0] - pitchY);
    if (rowCenters.length > wantRows) rowCenters = rowCenters.slice(rowCenters.length - wantRows);
    // 空列在右侧（空列会向左收拢）
    while (colCenters.length < wantCols) colCenters.push(colCenters[colCenters.length - 1] + pitchX);
    if (colCenters.length > wantCols) colCenters = colCenters.slice(0, wantCols);

    const rect = {
      x: Math.round(colCenters[0] - pitchX / 2),
      y: Math.round(rowCenters[0] - pitchY / 2),
      w: Math.round(colCenters.length * pitchX),
      h: Math.round(rowCenters.length * pitchY),
    };
    if (rect.w < 8 || rect.h < 8) return null;
    return { rect, pitchX, pitchY, rowCenters, colCenters };
  }

  /**
   * 棋盘区域。能拟合出方块格子时用格子（可包含顶部空行）；
   * 否则退回背景色包围盒。
   *
   * ``options.rows / options.cols`` 用来把残局补成完整棋盘。
   */
  function detectBoardRect(image, bgThreshold, options) {
    if (typeof bgThreshold === 'object' && bgThreshold) {
      options = bgThreshold;
      bgThreshold = 55;
    }
    const lattice = fitBoardLattice(image, options || {});
    if (lattice) return lattice.rect;
    return detectBoardRectByBackground(image, bgThreshold == null ? 55 : bgThreshold);
  }

  /** 贪心颜色聚类。饱和色按色相归并，避免同一颜色的明暗被拆开。 */
  function clusterColors(colors, threshold) {
    const clusters = [];
    for (const col of colors) {
      let hit = null;
      for (const cl of clusters) {
        if (colorDist(col, cl.centroid) <= threshold) { hit = cl; break; }
      }
      if (hit) {
        hit.items.push(col);
        const n = hit.items.length;
        for (let c = 0; c < 3; c++) {
          hit.centroid[c] += (col[c] - hit.centroid[c]) / n;
        }
      } else {
        clusters.push({ centroid: col.slice(), items: [col], count: 1 });
      }
    }
    for (const cl of clusters) cl.count = cl.items.length;
    return clusters;
  }

  /** 反复合并最近的两簇，直到簇数 ≤ k。 */
  function mergeToK(clusters, k) {
    let list = clusters.slice();
    while (list.length > k) {
      let bi = 0, bj = 1, bd = Infinity;
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length; j++) {
          const d = colorDist(list[i].centroid, list[j].centroid);
          if (d < bd) { bd = d; bi = i; bj = j; }
        }
      }
      const a = list[bi], b = list[bj];
      const total = a.count + b.count;
      const merged = {
        centroid: [0, 1, 2].map(
          (c) => (a.centroid[c] * a.count + b.centroid[c] * b.count) / total),
        items: a.items.concat(b.items),
        count: total,
      };
      list = list.filter((_, idx) => idx !== bi && idx !== bj).concat([merged]);
    }
    return list;
  }

  /**
   * 逐格取方块本体的中位色，避开中心星星和格子间隙。
   * 环带大约在格子半宽的 34%–46%：星星在更内侧，圆角和缝隙在更外侧。
   * 只留又亮又饱和的像素，夜空和白高光进不来。
   */
  function sampleCells(image, rect, rows, cols) {
    const { data, width: iw, height: ih } = image;
    const cw = rect.w / cols;
    const ch = rect.h / rows;
    const out = [];
    for (let ry = 0; ry < rows; ry++) {
      const row = [];
      for (let cx = 0; cx < cols; cx++) {
        const ccx = rect.x + (cx + 0.5) * cw;
        const ccy = rect.y + (ry + 0.5) * ch;
        const rad = Math.min(cw, ch);
        const rIn = rad * 0.34;
        const rOut = rad * 0.46;
        const rIn2 = rIn * rIn;
        const rOut2 = rOut * rOut;
        const x0 = Math.max(0, Math.floor(ccx - rOut));
        const x1 = Math.min(iw, Math.ceil(ccx + rOut));
        const y0 = Math.max(0, Math.floor(ccy - rOut));
        const y1 = Math.min(ih, Math.ceil(ccy + rOut));
        const satPx = [];
        const all = [];
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const dx = x + 0.5 - ccx;
            const dy = y + 0.5 - ccy;
            const d2 = dx * dx + dy * dy;
            if (d2 < rIn2 || d2 > rOut2) continue;
            const i = (y * iw + x) * 4;
            const rgb = [data[i], data[i + 1], data[i + 2]];
            all.push(rgb);
            if (satOf(rgb) >= 0.35 && Math.max(rgb[0], rgb[1], rgb[2]) >= 80) satPx.push(rgb);
          }
        }
        // 夜空里的小亮点也会又亮又饱和，但只占环带的极少像素。
        // 真正的方块几乎整圈都是本体颜色。播放按钮混进来的第二种颜色
        // 再用色相上最大的那一簇去掉，避免通道中位数把蓝和橙拌成青色。
        const tileLike = satPx.length >= 12 && satPx.length >= all.length * 0.35;
        const pool = tileLike ? satPx : all;
        row.push(pool.length ? dominantRGB(pool) : [0, 0, 0]);
      }
      out.push(row);
    }
    return out;
  }

  /**
   * 空格：离背景够近，或（夜空那种暗而饱和的底）亮度很低，
   * 或（浅色底）几乎没有饱和度。方块本体两者都不是。
   */
  function isEmptyColor(col, bg, emptyThreshold) {
    if (dist(col, bg) <= emptyThreshold) return true;
    const bgV = valueOf(bg);
    if (bgV < 0.45) return valueOf(col) < 0.34;
    return satOf(col) < 0.15;
  }

  /**
   * 完整识别：取样 → 聚类 → 判空格 → 输出棋盘。
   *
   * 返回 ``{grid, clusters, empties, cellColors, background}``。
   */
  function extractGrid(image, rect, options) {
    const opts = options || {};
    const rows = opts.rows || 10;
    const cols = opts.cols || 10;
    const k = opts.k || 5;
    const emptyThreshold = opts.emptyThreshold == null ? 60 : opts.emptyThreshold;
    const clusterThreshold = opts.clusterThreshold == null ? 42 : opts.clusterThreshold;

    const cellColors = sampleCells(image, rect, rows, cols);
    const bg = opts.background || backgroundOf(image);

    // 空格不参与聚类，否则夜空会占掉一个颜色名额
    const isEmpty = [];
    const nonEmptyColors = [];
    for (let ry = 0; ry < rows; ry++) {
      const line = [];
      for (let cx = 0; cx < cols; cx++) {
        const near = isEmptyColor(cellColors[ry][cx], bg, emptyThreshold);
        line.push(near);
        if (!near) nonEmptyColors.push(cellColors[ry][cx]);
      }
      isEmpty.push(line);
    }

    let clusters = clusterColors(
      nonEmptyColors.length ? nonEmptyColors : cellColors.flat(), clusterThreshold);
    clusters = mergeToK(clusters, k);
    // 按亮度排序，使颜色编号稳定可复现
    clusters.sort((a, b) => lum(a.centroid) - lum(b.centroid));

    const grid = [];
    const empties = [];
    for (let ry = 0; ry < rows; ry++) {
      const line = [];
      for (let cx = 0; cx < cols; cx++) {
        if (isEmpty[ry][cx]) { line.push(EMPTY); empties.push([ry, cx]); continue; }
        let best = 0, bestD = Infinity;
        clusters.forEach((cl, idx) => {
          const d = colorDist(cellColors[ry][cx], cl.centroid);
          if (d < bestD) { bestD = d; best = idx; }
        });
        line.push(best);
      }
      grid.push(line);
    }
    return { grid, clusters, empties, cellColors, background: bg };
  }

  const Vision = {
    EMPTY,
    makeImage, dist, lum, medianRGB, dominantColor, cornerColor, backgroundOf,
    detectBoardRect, fitBoardLattice, clusterColors, mergeToK, sampleCells, extractGrid,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = Vision;
  if (typeof window !== 'undefined') window.Vision = Vision;
})(typeof globalThis !== 'undefined' ? globalThis : this);
