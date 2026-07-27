/**
 * Playground UI: wires the controls to the Pyodide worker and animates the
 * charts as search progress streams in.
 *
 * Progress messages are coalesced into one repaint per animation frame, so the
 * page stays smooth even when the planner is reporting thousands of expansions
 * a second.
 */

import { LineChart, BarChart, Wavefront, palette, fmt } from "./charts.js";

const LOCAL_PYODIDE = "vendor/pyodide/";
const CDN_PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";

const RACE_CONFIGS = [
  { planner: "astar", heuristic: "lmcut" },
  { planner: "astar", heuristic: "hmax" },
  { planner: "gbfs", heuristic: "hff" },
  { planner: "bfs", heuristic: null },
];

const $ = (id) => document.getElementById(id);

const state = {
  demos: [],
  worker: null,
  running: false,
  started: 0,
  points: [],
  peakOpen: 0,
  pending: false,
  raceRows: [],
};

/* ------------------------------------------------------------------ charts */
const costChart = new LineChart($("chart-cost"), { xLabel: "expanded" });
const openChart = new LineChart($("chart-open"), { xLabel: "expanded", area: true });
const wavefront = new Wavefront($("wavefront"));
const raceExpanded = new BarChart($("race-expanded"), { log: true });
const raceRuntime = new BarChart($("race-runtime"), {
  valueLabel: (v) => `${v.toFixed(0)} ms`,
});
const raceCost = new BarChart($("race-cost"));

function seriesForCost() {
  const pal = palette();
  return [
    { name: "f = g + h", color: pal.series[0], points: state.points.map((p) => [p[0], p[3]]) },
    { name: "g (path cost)", color: pal.series[1], points: state.points.map((p) => [p[0], p[1]]) },
    { name: "h (estimate)", color: pal.series[2], points: state.points.map((p) => [p[0], p[2]]) },
  ];
}

function renderLegend() {
  const pal = palette();
  $("legend-cost").innerHTML = [
    ["f = g + h", pal.series[0]],
    ["g", pal.series[1]],
    ["h", pal.series[2]],
  ]
    .map(([name, color]) =>
      `<span class="legend-item"><i style="background:${color}"></i>${name}</span>`)
    .join("");
}

function repaint() {
  state.pending = false;
  costChart.setSeries(seriesForCost());
  const pal = palette();
  openChart.setSeries([
    { name: "frontier", color: pal.series[0], points: state.points.map((p) => [p[0], p[5]]) },
  ]);
  wavefront.draw();

  const last = state.points[state.points.length - 1];
  if (last) {
    $("s-expanded").textContent = fmt(last[0]);
    $("s-generated").textContent = fmt(last[6]);
    $("s-open").textContent = fmt(last[5]);
  }
  $("s-time").textContent = `${((performance.now() - state.started) / 1000).toFixed(2)}s`;
}

function scheduleRepaint() {
  if (state.pending) return;
  state.pending = true;
  requestAnimationFrame(repaint);
}

/* ------------------------------------------------------------------ worker */
function bootMessage(text) {
  $("boot-text").textContent = text;
}

async function boot() {
  renderLegend();
  const [sources, demos, build] = await Promise.all([
    fetch("dist/jupyddl-sources.json").then((r) => r.json()),
    fetch("dist/demos.json").then((r) => r.json()),
    fetch("dist/build.json").then((r) => r.json()).catch(() => ({})),
  ]);
  state.demos = demos;
  fillDemos();

  // Prefer a vendored runtime (offline / self-hosted); fall back to the CDN.
  let base = CDN_PYODIDE;
  try {
    const probe = await fetch(`${LOCAL_PYODIDE}pyodide.mjs`, { method: "HEAD" });
    if (probe.ok) base = LOCAL_PYODIDE;
  } catch (e) { /* no local copy, use the CDN */ }

  bootMessage("Loading Python (about 10 MB, cached after the first visit)…");
  state.worker = new Worker("worker.js", { type: "module" });
  state.worker.onmessage = onWorkerMessage;
  state.worker.onerror = (event) => {
    const message = `Worker failed: ${event.message || "unknown error"}`;
    if ($("app").hidden) {
      // Still booting: keep the boot screen so an empty UI never looks ready.
      bootMessage(message);
      $("boot").querySelector(".spinner").style.display = "none";
    } else {
      showError(message);
      setRunning(false);
    }
  };
  state.worker.postMessage({ type: "init", pyodideBase: base, sources });
  $("build-info").textContent = build.version ? `jupyddl ${build.version}` : "";
}

function onWorkerMessage(event) {
  const { type, payload } = event.data;
  if (type === "ready") {
    fillSelect($("planner"), payload.planners, "astar");
    fillSelect($("heuristic"), payload.heuristics.concat(["none"]), "lmcut");
    $("build-info").textContent =
      `jupyddl ${payload.version} · CPython ${payload.python} · WebAssembly`;
    $("boot").hidden = true;
    $("app").hidden = false;
    requestAnimationFrame(repaint);
  } else if (type === "progress") {
    state.points.push(...payload.points);
    state.peakOpen = Math.max(state.peakOpen, payload.peak_open || 0);
    wavefront.add(payload.points);
    scheduleRepaint();
  } else if (type === "result") {
    finishSolve(payload);
  } else if (type === "race") {
    finishRace(payload);
  } else if (type === "error") {
    if ($("app").hidden) {
      // The error box lives inside the app pane, which is not on screen yet.
      bootMessage(payload.message);
      $("boot").querySelector(".spinner").style.display = "none";
    } else {
      showError(payload.message);
      setRunning(false);
    }
  }
}

/* --------------------------------------------------------------------- UI */
function fillSelect(select, values, preferred) {
  select.innerHTML = values
    .map((v) => `<option value="${v}"${v === preferred ? " selected" : ""}>${v}</option>`)
    .join("");
}

function fillDemos() {
  $("demo").innerHTML = state.demos
    .map((d, i) => `<option value="${i}">${d.title}</option>`)
    .join("");
  loadDemo(0);
}

function loadDemo(index) {
  const demo = state.demos[index];
  if (!demo) return;
  $("domain").value = demo.domain;
  $("problem").value = demo.problem;
  $("blurb").textContent = demo.blurb;
}

function showError(message) {
  const box = $("error");
  box.hidden = false;
  box.textContent = message;
}

function clearError() {
  $("error").hidden = true;
}

function setRunning(running) {
  state.running = running;
  $("run").disabled = running;
  $("race").disabled = running;
  $("run").textContent = running ? "Searching…" : "Run search";
}

function resetRun() {
  clearError();
  state.points = [];
  state.peakOpen = 0;
  state.started = performance.now();
  wavefront.reset();
  $("plan").hidden = true;
  $("plan").innerHTML = "";
  $("plan-empty").hidden = false;
  $("plan-empty").textContent = "Searching…";
  $("verdict").textContent = "";
  $("task-info").textContent = "";
  // Zero the tiles too: leaving the previous run's numbers beside empty charts
  // reads as if this run had already done the work.
  $("s-expanded").textContent = "0";
  $("s-generated").textContent = "0";
  $("s-open").textContent = "0";
  $("s-time").textContent = "0.00s";
  repaint();
}

/* ----------------------------------------------------------------- actions */
function run() {
  if (state.running) return;
  setRunning(true);
  resetRun();
  state.worker.postMessage({
    type: "solve",
    domain: $("domain").value,
    problem: $("problem").value,
    planner: $("planner").value,
    heuristic: $("heuristic").value,
    weight: 2.0,
  });
}

function finishSolve(payload) {
  setRunning(false);
  repaint();
  const { task, stats } = payload;
  $("task-info").textContent =
    `${task.facts} facts · ${task.operators} ground actions · ${task.goals} goals`;
  $("s-expanded").textContent = fmt(stats.expanded);
  $("s-generated").textContent = fmt(stats.generated);
  $("s-time").textContent = `${stats.runtime.toFixed(2)}s`;

  if (!payload.solved) {
    $("plan-empty").textContent = "No plan found — the goal is unreachable here.";
    $("verdict").innerHTML = '<span class="bad">unsolvable</span>';
    return;
  }
  $("plan-empty").hidden = true;
  const list = $("plan");
  list.hidden = false;
  list.innerHTML = payload.plan.map((action) => `<li>${escapeHtml(action)}</li>`).join("");
  const validity = payload.valid
    ? '<span class="ok">plan validated</span>'
    : '<span class="bad">plan did not validate</span>';
  $("verdict").innerHTML = `${validity} · cost ${payload.cost} · ${payload.plan.length} actions`;
}

function race() {
  if (state.running) return;
  setRunning(true);
  resetRun();
  // A race runs each configuration to completion rather than streaming one
  // search, so the live pane has nothing to plot — say so instead of leaving
  // empty axes next to zeroed counters.
  $("task-info").textContent = "planner race — results below";
  $("plan-empty").textContent = "Racing four configurations…";
  state.worker.postMessage({
    type: "race",
    domain: $("domain").value,
    problem: $("problem").value,
    configs: RACE_CONFIGS,
  });
}

function finishRace(payload) {
  setRunning(false);
  const pal = palette();
  const rows = payload.results;
  state.raceRows = rows;
  $("race-pane").hidden = false;
  $("plan-empty").textContent = "Run a search to see the plan.";

  const colored = (pick, note) =>
    rows.map((row, i) => ({
      name: row.label,
      value: pick(row) || 0,
      color: pal.series[i % pal.series.length],
      note: note ? note(row) : undefined,
    }));

  raceExpanded.setData(colored((r) => r.stats.expanded || 0));
  raceRuntime.setData(colored((r) => (r.stats.runtime || 0) * 1000));
  raceCost.setData(
    colored(
      (r) => (r.valid ? r.cost || 0 : 0),
      (r) => (r.valid ? String(r.cost) : "no plan"),
    ),
  );

  $("race-table").querySelector("tbody").innerHTML = rows
    .map(
      (row) =>
        `<tr><td>${escapeHtml(row.label)}</td>` +
        `<td>${fmt(row.stats.expanded || 0)}</td>` +
        `<td>${fmt(row.stats.generated || 0)}</td>` +
        `<td>${((row.stats.runtime || 0) * 1000).toFixed(0)}</td>` +
        `<td>${row.cost ?? "–"}</td>` +
        `<td>${row.valid ? "yes" : "no"}</td></tr>`,
    )
    .join("");
  $("race-pane").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------------------ events */
$("run").addEventListener("click", run);
$("race").addEventListener("click", race);
$("demo").addEventListener("change", (e) => loadDemo(Number(e.target.value)));

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    const which = tab.dataset.tab;
    for (const other of document.querySelectorAll(".tab")) {
      const active = other === tab;
      other.classList.toggle("active", active);
      other.setAttribute("aria-selected", String(active));
    }
    $("domain").hidden = which !== "domain";
    $("problem").hidden = which !== "problem";
  });
}

$("race-table-toggle").addEventListener("click", () => {
  const table = $("race-table");
  table.hidden = !table.hidden;
  $("race-table-toggle").textContent = table.hidden ? "Show table" : "Hide table";
});

$("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.getAttribute("data-theme") === "dark"
    || (!root.hasAttribute("data-theme")
        && window.matchMedia("(prefers-color-scheme: dark)").matches);
  root.setAttribute("data-theme", dark ? "light" : "dark");
  renderLegend();
  repaint();
  if (state.raceRows.length) finishRaceRepaint();
});

function finishRaceRepaint() {
  finishRace({ results: state.raceRows });
  setRunning(false);
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    repaint();
    if (state.raceRows.length) finishRaceRepaint();
  }, 120);
});

boot().catch((error) => showError(String(error)));
