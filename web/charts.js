/**
 * A very small charting layer for the playground.
 *
 * No dependencies and no build step: SVG for the line/bar charts (crisp text,
 * easy hit-testing) and canvas for the search wavefront, which draws thousands
 * of nodes per frame. Colours come from CSS custom properties, so the whole
 * thing follows the page's light/dark theme without any JS branching.
 */

const NS = "http://www.w3.org/2000/svg";

function css(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (value && value.trim()) || fallback;
}

export function palette() {
  return {
    surface: css("--surface-1", "#fcfcfb"),
    text: css("--text-primary", "#0b0b0b"),
    secondary: css("--text-secondary", "#52514e"),
    muted: css("--text-muted", "#898781"),
    grid: css("--grid", "#e1e0d9"),
    axis: css("--axis", "#c3c2b7"),
    series: [
      css("--series-1", "#2a78d6"),
      css("--series-2", "#eb6834"),
      css("--series-3", "#1baf7a"),
      css("--series-4", "#eda100"),
      css("--series-5", "#e87ba4"),
      css("--series-6", "#008300"),
    ],
    good: css("--good", "#0ca30c"),
    seqLow: css("--seq-low", "#cde2fb"),
    seqHigh: css("--seq-high", "#0d366b"),
  };
}

function el(name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

function niceTicks(min, max, count = 4) {
  if (!isFinite(min) || !isFinite(max) || max === min) return [min || 0];
  const span = max - min;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 0.001; v += step) {
    out.push(Math.round(v * 1e6) / 1e6);
  }
  return out;
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const n = Number(value);
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e4) return `${Math.round(n / 1000)}k`;
  if (!Number.isInteger(n)) return n.toFixed(Math.abs(n) < 10 ? 2 : 0);
  return n.toLocaleString("en-US");
}

/* ------------------------------------------------------------------ */
/* Line chart with a crosshair + tooltip                              */
/* ------------------------------------------------------------------ */
export class LineChart {
  constructor(container, options = {}) {
    this.container = container;
    this.options = Object.assign(
      { yLabel: "", xLabel: "", area: false, minHeight: 170 },
      options,
    );
    this.series = [];
    this.svg = el("svg", { class: "chart" });
    this.container.innerHTML = "";
    this.container.appendChild(this.svg);
    this.tooltip = document.createElement("div");
    this.tooltip.className = "chart-tooltip";
    this.tooltip.setAttribute("role", "status");
    this.container.appendChild(this.tooltip);
    this._bindHover();
  }

  setSeries(series) {
    this.series = series;
    this.render();
  }

  _bindHover() {
    this.svg.addEventListener("mousemove", (event) => this._hover(event));
    this.svg.addEventListener("mouseleave", () => {
      this.tooltip.style.opacity = "0";
      if (this.crosshair) this.crosshair.setAttribute("opacity", "0");
      for (const dot of this.dots || []) dot.setAttribute("opacity", "0");
    });
  }

  _hover(event) {
    if (!this.plot || !this.series.length) return;
    const box = this.svg.getBoundingClientRect();
    const x = event.clientX - box.left;
    const { left, width, top, height } = this.plot;
    if (x < left || x > left + width) return;
    const fraction = (x - left) / width;
    const value = this.xMin + fraction * (this.xMax - this.xMin);

    const rows = [];
    this.series.forEach((serie, index) => {
      if (!serie.points.length) return;
      let best = serie.points[0];
      let bestDelta = Infinity;
      for (const point of serie.points) {
        const delta = Math.abs(point[0] - value);
        if (delta < bestDelta) { bestDelta = delta; best = point; }
      }
      rows.push({ name: serie.name, color: serie.color, point: best });
      const dot = this.dots[index];
      if (dot) {
        dot.setAttribute("cx", this._px(best[0]));
        dot.setAttribute("cy", this._py(best[1]));
        dot.setAttribute("opacity", "1");
      }
    });
    if (!rows.length) return;

    this.crosshair.setAttribute("x1", x);
    this.crosshair.setAttribute("x2", x);
    this.crosshair.setAttribute("y1", top);
    this.crosshair.setAttribute("y2", top + height);
    this.crosshair.setAttribute("opacity", "1");

    const header = `${this.options.xLabel || "x"} ${fmt(rows[0].point[0])}`;
    this.tooltip.innerHTML =
      `<div class="tt-head">${header}</div>` +
      rows
        .map(
          (row) =>
            `<div class="tt-row"><span class="tt-swatch" style="background:${row.color}"></span>` +
            `<span class="tt-name">${row.name}</span>` +
            `<span class="tt-value">${fmt(row.point[1])}</span></div>`,
        )
        .join("");
    this.tooltip.style.opacity = "1";
    const offset = x > left + width * 0.6 ? -12 - this.tooltip.offsetWidth : 12;
    this.tooltip.style.transform = `translate(${x + offset}px, ${top + 6}px)`;
  }

  _px(value) {
    const { left, width } = this.plot;
    if (this.xMax === this.xMin) return left;
    return left + ((value - this.xMin) / (this.xMax - this.xMin)) * width;
  }

  _py(value) {
    const { top, height } = this.plot;
    if (this.yMax === this.yMin) return top + height;
    return top + height - ((value - this.yMin) / (this.yMax - this.yMin)) * height;
  }

  render() {
    const pal = palette();
    const width = this.container.clientWidth || 460;
    const height = Math.max(this.options.minHeight, this.container.clientHeight || 0);
    const pad = { top: 12, right: 14, bottom: 26, left: 46 };
    this.plot = {
      left: pad.left,
      top: pad.top,
      width: Math.max(10, width - pad.left - pad.right),
      height: Math.max(10, height - pad.top - pad.bottom),
    };
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.setAttribute("width", width);
    this.svg.setAttribute("height", height);
    while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);

    const points = this.series.flatMap((s) => s.points);
    this.xMin = 0;
    this.xMax = Math.max(1, ...points.map((p) => p[0]));
    this.yMin = 0;
    this.yMax = Math.max(1, ...points.map((p) => p[1])) * 1.12;

    // gridlines + y ticks
    for (const tick of niceTicks(this.yMin, this.yMax, 4)) {
      const y = this._py(tick);
      this.svg.appendChild(
        el("line", {
          x1: this.plot.left, x2: this.plot.left + this.plot.width, y1: y, y2: y,
          stroke: pal.grid, "stroke-width": 1,
        }),
      );
      const label = el("text", {
        x: this.plot.left - 8, y: y + 3.5, "text-anchor": "end",
        fill: pal.secondary, "font-size": 10.5,
      });
      label.textContent = fmt(tick);
      this.svg.appendChild(label);
    }
    for (const tick of niceTicks(this.xMin, this.xMax, 4)) {
      const label = el("text", {
        x: this._px(tick), y: height - 8, "text-anchor": "middle",
        fill: pal.secondary, "font-size": 10.5,
      });
      label.textContent = fmt(tick);
      this.svg.appendChild(label);
    }

    for (const serie of this.series) {
      if (!serie.points.length) continue;
      const d = serie.points
        .map((p, i) => `${i ? "L" : "M"}${this._px(p[0]).toFixed(1)},${this._py(p[1]).toFixed(1)}`)
        .join("");
      if (this.options.area) {
        const base = this._py(this.yMin);
        const first = this._px(serie.points[0][0]);
        const last = this._px(serie.points[serie.points.length - 1][0]);
        this.svg.appendChild(
          el("path", {
            d: `${d}L${last.toFixed(1)},${base}L${first.toFixed(1)},${base}Z`,
            fill: serie.color, opacity: 0.13, stroke: "none",
          }),
        );
      }
      this.svg.appendChild(
        el("path", {
          d, fill: "none", stroke: serie.color, "stroke-width": 2,
          "stroke-linejoin": "round", "stroke-linecap": "round",
        }),
      );
    }

    this.crosshair = el("line", {
      stroke: pal.axis, "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0,
    });
    this.svg.appendChild(this.crosshair);
    this.dots = this.series.map((serie) => {
      const dot = el("circle", {
        r: 4, fill: serie.color, stroke: pal.surface, "stroke-width": 2, opacity: 0,
      });
      this.svg.appendChild(dot);
      return dot;
    });
  }
}

/* ------------------------------------------------------------------ */
/* Search wavefront (canvas)                                          */
/* ------------------------------------------------------------------ */
export class Wavefront {
  constructor(canvas) {
    this.canvas = canvas;
    this.nodes = [];
    this.maxDepth = 1;
    this.maxH = 1;
  }

  reset() {
    this.nodes = [];
    this.maxDepth = 1;
    this.maxH = 1;
    this.draw();
  }

  /** points: [step, g, h, f, depth, open, generated, evaluated] */
  add(points) {
    for (const point of points) {
      const depth = point[4];
      const h = point[2];
      this.maxDepth = Math.max(this.maxDepth, depth);
      this.maxH = Math.max(this.maxH, h);
      this.nodes.push({ depth, h });
    }
  }

  draw() {
    const canvas = this.canvas;
    const pal = palette();
    const ratio = window.devicePixelRatio || 1;
    const size = Math.max(160, Math.min(canvas.clientWidth, canvas.clientHeight || 1e9));
    canvas.width = size * ratio;
    canvas.height = size * ratio;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const radius = size / 2 - 10;

    // faint depth rings for scale
    ctx.strokeStyle = pal.grid;
    ctx.lineWidth = 1;
    for (let ring = 1; ring <= 4; ring += 1) {
      ctx.beginPath();
      ctx.arc(cx, cy, (radius * ring) / 4, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (!this.nodes.length) return;

    // Group by depth so each ring spreads its own nodes evenly.
    const rings = new Map();
    for (const node of this.nodes) {
      if (!rings.has(node.depth)) rings.set(node.depth, []);
      rings.get(node.depth).push(node);
    }
    const low = pal.seqLow;
    const high = pal.seqHigh;
    for (const [depth, group] of rings) {
      const r = (depth / Math.max(1, this.maxDepth)) * radius;
      group.forEach((node, index) => {
        const angle = (2 * Math.PI * (index + 0.5)) / group.length;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        ctx.fillStyle = mix(low, high, node.h / (this.maxH || 1));
        ctx.beginPath();
        ctx.arc(x, y, 2.4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "").trim();
  const full = clean.length === 3 ? clean.split("").map((c) => c + c).join("") : clean;
  return [
    parseInt(full.slice(0, 2), 16),
    parseInt(full.slice(2, 4), 16),
    parseInt(full.slice(4, 6), 16),
  ];
}

function mix(a, b, t) {
  const clamped = Math.max(0, Math.min(1, t || 0));
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  const out = ca.map((v, i) => Math.round(v + (cb[i] - v) * clamped));
  return `rgb(${out[0]},${out[1]},${out[2]})`;
}

/* ------------------------------------------------------------------ */
/* Horizontal bar chart with rounded data-ends                        */
/* ------------------------------------------------------------------ */
export class BarChart {
  constructor(container, options = {}) {
    this.container = container;
    this.options = Object.assign({ valueLabel: (v) => fmt(v), log: false }, options);
    this.svg = el("svg", { class: "chart" });
    container.innerHTML = "";
    container.appendChild(this.svg);
  }

  setData(items) {
    this.items = items;
    this.render();
  }

  render() {
    const pal = palette();
    const items = this.items || [];
    const width = this.container.clientWidth || 420;
    const rowHeight = 30;
    const pad = { top: 6, right: 62, bottom: 6, left: 108 };
    const height = pad.top + pad.bottom + items.length * rowHeight;
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.setAttribute("width", width);
    this.svg.setAttribute("height", height);
    while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);

    const plotWidth = Math.max(10, width - pad.left - pad.right);
    const scale = (value) => {
      const max = Math.max(...items.map((i) => i.value), 1);
      if (this.options.log) {
        const lv = Math.log10(Math.max(value, 1));
        const lm = Math.log10(Math.max(max, 10));
        return (lv / lm) * plotWidth;
      }
      return (value / max) * plotWidth;
    };

    items.forEach((item, index) => {
      const y = pad.top + index * rowHeight;
      const barHeight = 17;
      const barY = y + (rowHeight - barHeight) / 2;
      const length = Math.max(0, scale(item.value));
      const radius = Math.min(4, length / 2);

      const label = el("text", {
        x: pad.left - 10, y: barY + barHeight / 2 + 4, "text-anchor": "end",
        fill: pal.secondary, "font-size": 11.5,
      });
      label.textContent = item.name;
      this.svg.appendChild(label);

      if (length > 0.5) {
        // square baseline, rounded data-end
        const path =
          `M${pad.left},${barY}` +
          `H${pad.left + length - radius}` +
          `Q${pad.left + length},${barY} ${pad.left + length},${barY + radius}` +
          `V${barY + barHeight - radius}` +
          `Q${pad.left + length},${barY + barHeight} ${pad.left + length - radius},${barY + barHeight}` +
          `H${pad.left}Z`;
        const bar = el("path", { d: path, fill: item.color });
        const title = el("title");
        title.textContent = `${item.name}: ${this.options.valueLabel(item.value)}`;
        bar.appendChild(title);
        this.svg.appendChild(bar);
      }

      const value = el("text", {
        x: pad.left + length + 8, y: barY + barHeight / 2 + 4,
        fill: pal.secondary, "font-size": 11.5,
      });
      value.textContent = item.note || this.options.valueLabel(item.value);
      this.svg.appendChild(value);
    });
  }
}

export { fmt };
