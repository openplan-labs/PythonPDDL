/**
 * Workbench UI: four views over one Pyodide worker.
 *
 * Solve streams a single search and animates it; Experiment sweeps an
 * instance x configuration matrix and fills a sortable table; PDDL support
 * renders the requirement matrix straight from the library; Generate produces
 * reproducible instances and hands them to Solve.
 *
 * Progress messages are coalesced into one repaint per animation frame, so the
 * page stays smooth even when the planner reports thousands of expansions a
 * second.
 */

import { LineChart, BarChart, Wavefront, palette, fmt } from "./charts.js";

const LOCAL_PYODIDE = "vendor/pyodide/";
const CDN_PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";

const $ = (id) => document.getElementById(id);

const state = {
  demos: [],
  capabilities: null,
  worker: null,
  running: false,
  started: 0,
  points: [],
  pending: false,
  experiment: { rows: [], expected: 0, sortKey: null, sortDir: 1 },
  generated: null,
};

/* ------------------------------------------------------------------ charts */
const costChart = new LineChart($("chart-cost"), { xLabel: "expanded" });
const openChart = new LineChart($("chart-open"), { xLabel: "expanded", area: true });
const wavefront = new Wavefront($("wavefront"));
const expChart = new BarChart($("exp-chart"), { log: true });

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
  const pal = palette();
  costChart.setSeries([
    { name: "f = g + h", color: pal.series[0], points: state.points.map((p) => [p[0], p[3]]) },
    { name: "g (path cost)", color: pal.series[1], points: state.points.map((p) => [p[0], p[1]]) },
    { name: "h (estimate)", color: pal.series[2], points: state.points.map((p) => [p[0], p[2]]) },
  ]);
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
  if (state.running) {
    $("s-time").textContent = `${((performance.now() - state.started) / 1000).toFixed(2)}s`;
  }
}

function scheduleRepaint() {
  if (state.pending) return;
  state.pending = true;
  requestAnimationFrame(repaint);
}

/* ------------------------------------------------------------------- boot */
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
    onReady(payload);
  } else if (type === "progress") {
    state.points.push(...payload.points);
    wavefront.add(payload.points);
    scheduleRepaint();
  } else if (type === "result") {
    finishSolve(payload);
  } else if (type === "experiment-row") {
    addExperimentRow(payload);
  } else if (type === "experiment-done") {
    finishExperiment();
  } else if (type === "inspect") {
    showInspection(payload);
  } else if (type === "generated") {
    showGenerated(payload);
  } else if (type === "error") {
    if ($("app").hidden) {
      bootMessage(payload.message);
      $("boot").querySelector(".spinner").style.display = "none";
    } else {
      showError(payload.message);
      setRunning(false);
    }
  }
}

function onReady(payload) {
  state.capabilities = payload;
  const planners = payload.planners.map((p) => p.name);
  fillSelect($("planner"), planners, "astar");
  fillSelect($("heuristic"), payload.heuristics.concat(["none"]), "lmcut");
  fillSelect($("gen-kind"), payload.generators.map((g) => g.name), "gripper");
  updateGeneratorBlurb();

  buildChecklist("pick-instances", state.demos.map((d) => ({
    value: d.id, label: d.title,
    checked: ["gripper", "blocksworld8", "hanoi", "logistics"].includes(d.id),
  })));
  buildChecklist("pick-planners", planners.map((name) => ({
    value: name, label: name, checked: ["astar", "gbfs", "bfs"].includes(name),
  })));
  buildChecklist("pick-heuristics", payload.heuristics.map((name) => ({
    value: name, label: name, checked: ["lmcut", "hff"].includes(name),
  })));
  renderRequirements();

  $("build-info").textContent =
    `jupyddl ${payload.version} · CPython ${payload.python} · WebAssembly`;
  $("boot").hidden = true;
  $("app").hidden = false;
  requestAnimationFrame(repaint);
}

/* --------------------------------------------------------------------- UI */
function fillSelect(select, values, preferred) {
  select.innerHTML = values
    .map((v) => `<option value="${v}"${v === preferred ? " selected" : ""}>${v}</option>`)
    .join("");
}

function buildChecklist(id, items) {
  $(id).innerHTML = items
    .map(
      (item) =>
        `<label class="check"><input type="checkbox" value="${escapeHtml(item.value)}"` +
        `${item.checked ? " checked" : ""} /> ${escapeHtml(item.label)}</label>`,
    )
    .join("");
}

function checkedValues(id) {
  return [...$(id).querySelectorAll("input:checked")].map((input) => input.value);
}

function fillDemos() {
  $("demo").innerHTML = state.demos
    .map((d, i) => `<option value="${i}">${escapeHtml(d.title)}</option>`)
    .join("");
  loadDemo(0);
}

function loadDemo(index) {
  const demo = state.demos[index];
  if (!demo) return;
  $("domain").value = demo.domain;
  $("problem").value = demo.problem;
  $("blurb").textContent = demo.blurb;
  $("tagrow").innerHTML = (demo.tags || [])
    .map((tag) => `<span class="chip">${escapeHtml(tag)}</span>`)
    .join("");
  // Each demo carries the configuration it is actually pleasant to run with:
  // the hard ones would look like a hang under an optimal planner.
  if (demo.planner && $("planner").options.length) $("planner").value = demo.planner;
  if (demo.heuristic && $("heuristic").options.length) {
    $("heuristic").value = demo.heuristic;
  }
}

function showError(message) {
  const box = $("error");
  box.hidden = false;
  box.textContent = message;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearError() {
  $("error").hidden = true;
}

function setRunning(running) {
  state.running = running;
  $("run").disabled = running;
  $("run-experiment").disabled = running;
  $("generate").disabled = running;
  $("inspect").disabled = running;
  $("run").textContent = running ? "Searching…" : "Run search";
}

function limits(nodesId, secondsId) {
  const nodes = Number($(nodesId).value);
  const seconds = Number($(secondsId).value);
  const out = {};
  if (nodes > 0) out.max_expansions = nodes;
  if (seconds > 0) out.time_limit = seconds;
  return out;
}

/* ------------------------------------------------------------------ solve */
function resetRun() {
  clearError();
  state.points = [];
  state.started = performance.now();
  wavefront.reset();
  $("plan").hidden = true;
  $("plan").innerHTML = "";
  $("plan-empty").hidden = false;
  $("plan-empty").textContent = "Searching…";
  $("verdict").textContent = "";
  $("verdict").className = "verdict";
  $("copy-plan").hidden = true;
  for (const [id, value] of [
    ["s-expanded", "0"], ["s-generated", "0"], ["s-open", "0"], ["s-time", "0.00s"],
  ]) {
    $(id).textContent = value;
  }
  repaint();
}

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
    limits: limits("limit-nodes", "limit-seconds"),
  });
}

function describeTask(task) {
  const parts = [
    `${fmt(task.facts)} facts`,
    `${fmt(task.operators)} ground actions`,
    `${task.goals} goals`,
  ];
  if (task.axioms) parts.push(`${task.axioms} axioms`);
  if (task.numeric && task.numeric.length) {
    parts.push(`${task.numeric.length} numeric fluents`);
  }
  if (task.temporal) parts.push("durative");
  return parts.join(" · ");
}

function finishSolve(payload) {
  setRunning(false);
  repaint();
  const { task, stats } = payload;
  $("task-info").textContent = describeTask(task);
  $("s-expanded").textContent = fmt(stats.expanded);
  $("s-generated").textContent = fmt(stats.generated);
  $("s-time").textContent = `${stats.runtime.toFixed(2)}s`;

  if (!payload.solved) {
    $("plan-empty").textContent = payload.truncated
      ? "Stopped at the budget before finding a plan — raise the limits and try again."
      : "No plan found: the goal is unreachable in this instance.";
    $("verdict").innerHTML = payload.truncated
      ? '<span class="warn">truncated</span> — no plan found within the budget'
      : '<span class="bad">unsolvable</span> — the search space was exhausted';
    return;
  }
  $("plan-empty").hidden = true;
  const list = $("plan");
  list.hidden = false;
  list.innerHTML = payload.plan.map((a) => `<li>${escapeHtml(a)}</li>`).join("");
  $("copy-plan").hidden = false;

  const bits = [
    payload.valid
      ? '<span class="ok">plan validated</span>'
      : '<span class="bad">plan did not validate</span>',
    `cost ${payload.cost}`,
    `${payload.plan.length} actions`,
  ];
  if (payload.makespan != null) bits.push(`makespan ${payload.makespan}`);
  $("verdict").innerHTML = bits.join(" · ");
}

function showInspection(payload) {
  setRunning(false);
  const { task } = payload;
  $("task-info").textContent = describeTask(task);
  $("plan-empty").hidden = false;
  $("plan").hidden = true;
  $("verdict").innerHTML =
    `<span class="ok">grounded</span> — not solved` +
    (task.requirements.length
      ? ` · requirements: ${escapeHtml(task.requirements.join(" "))}`
      : "");
  $("plan-empty").textContent =
    `First ground actions: ${payload.sample_operators.join(", ") || "none"}`;
}

/* ------------------------------------------------------------- experiment */
function runExperiment() {
  if (state.running) return;
  const instanceIds = checkedValues("pick-instances");
  const planners = checkedValues("pick-planners");
  const heuristics = checkedValues("pick-heuristics");
  if (!instanceIds.length || !planners.length) {
    showError("Pick at least one instance and one planner.");
    return;
  }

  const informed = new Set(
    (state.capabilities.planners || [])
      .filter((p) => p.requires_heuristic)
      .map((p) => p.name),
  );
  const configs = [];
  for (const planner of planners) {
    if (informed.has(planner)) {
      if (!heuristics.length) {
        showError(`${planner} needs a heuristic — pick at least one.`);
        return;
      }
      for (const heuristic of heuristics) configs.push({ planner, heuristic });
    } else {
      // An uninformed planner would run identically for every heuristic, so it
      // contributes exactly one row rather than one per heuristic.
      configs.push({ planner, heuristic: null });
    }
  }

  const instances = state.demos.filter((d) => instanceIds.includes(d.id));
  clearError();
  setRunning(true);
  state.experiment = {
    rows: [], expected: instances.length * configs.length, sortKey: null, sortDir: 1,
  };
  $("exp-results-pane").hidden = false;
  $("exp-table").querySelector("tbody").innerHTML = "";
  $("exp-progress").hidden = false;
  $("exp-progress-fill").style.width = "0%";
  $("exp-status").textContent =
    `Running ${instances.length} × ${configs.length} = ${state.experiment.expected} runs…`;
  $("export-csv").disabled = true;
  $("export-json").disabled = true;

  state.worker.postMessage({
    type: "experiment",
    instances: instances.map((d) => ({
      id: d.id, domain: d.domain, problem: d.problem,
    })),
    configs,
    limits: limits("exp-nodes", "exp-seconds"),
  });
}

function addExperimentRow(row) {
  state.experiment.rows.push(row);
  const done = state.experiment.rows.length;
  const total = state.experiment.expected || done;
  $("exp-progress-fill").style.width = `${Math.round((done / total) * 100)}%`;
  $("exp-status").textContent = `${done} / ${total} runs`;
  renderExperimentTable();
}

function configOf(row) {
  return row.heuristic ? `${row.planner}/${row.heuristic}` : row.planner;
}

function renderExperimentTable() {
  const { rows, sortKey, sortDir } = state.experiment;
  const sorted = [...rows];
  if (sortKey) {
    sorted.sort((a, b) => {
      const av = sortKey === "config" ? configOf(a) : a[sortKey];
      const bv = sortKey === "config" ? configOf(b) : b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;   // unsolved rows sink to the bottom
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return (av - bv) * sortDir;
      }
      return String(av).localeCompare(String(bv)) * sortDir;
    });
  }

  $("exp-table").querySelector("tbody").innerHTML = sorted
    .map((row) => {
      let status = '<span class="bad">no</span>';
      if (row.error) status = '<span class="bad">error</span>';
      else if (row.truncated && !row.solved) status = '<span class="warn">truncated</span>';
      else if (row.solved && row.valid) status = '<span class="ok">yes</span>';
      else if (row.solved) status = '<span class="warn">invalid</span>';
      return (
        `<tr${row.error ? ' class="row-error" title="' + escapeHtml(row.error) + '"' : ""}>` +
        `<td>${escapeHtml(row.instance)}</td>` +
        `<td>${escapeHtml(configOf(row))}</td>` +
        `<td>${status}</td>` +
        `<td class="num">${row.cost ?? "–"}</td>` +
        `<td class="num">${row.length ?? "–"}</td>` +
        `<td class="num">${fmt(row.expanded)}</td>` +
        `<td class="num">${fmt(row.generated)}</td>` +
        `<td class="num">${fmt(row.evaluated)}</td>` +
        `<td class="num">${(row.runtime * 1000).toFixed(1)}</td>` +
        `</tr>`
      );
    })
    .join("");
}

function finishExperiment() {
  setRunning(false);
  const rows = state.experiment.rows;
  const solved = rows.filter((r) => r.valid).length;
  $("exp-status").textContent =
    `${rows.length} runs · ${solved} validated plans`;
  $("exp-caption").textContent =
    `Experiment results — ${rows.length} runs, ${solved} validated plans`;
  $("exp-progress").hidden = true;
  $("export-csv").disabled = rows.length === 0;
  $("export-json").disabled = rows.length === 0;

  // Total effort per configuration, which is the headline comparison.
  const pal = palette();
  const totals = new Map();
  for (const row of rows) {
    const key = configOf(row);
    totals.set(key, (totals.get(key) || 0) + (row.expanded || 0));
  }
  expChart.setData(
    [...totals.entries()].map(([name, value], i) => ({
      name, value, color: pal.series[i % pal.series.length],
    })),
  );
}

const CSV_COLUMNS = [
  "instance", "planner", "heuristic", "solved", "valid", "truncated",
  "cost", "length", "makespan", "expanded", "generated", "evaluated",
  "runtime", "facts", "operators", "error",
];

function download(name, text, mime) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function exportCsv() {
  const escape = (value) => {
    const text = value == null ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const lines = [CSV_COLUMNS.join(",")];
  for (const row of state.experiment.rows) {
    lines.push(CSV_COLUMNS.map((key) => escape(row[key])).join(","));
  }
  download("jupyddl-experiment.csv", lines.join("\n"), "text/csv");
}

function exportJson() {
  download(
    "jupyddl-experiment.json",
    JSON.stringify(
      { generated: new Date().toISOString(), rows: state.experiment.rows },
      null,
      2,
    ),
    "application/json",
  );
}

/* ----------------------------------------------------------- requirements */
const SUPPORT_BLURB = {
  native: "modelled directly",
  compiled: "compiled into the core representation",
  partial: "supported with a documented restriction",
  rejected: "refused with a clear error",
};

function renderRequirements(filter = "all") {
  const caps = state.capabilities;
  if (!caps) return;
  const counts = caps.requirement_summary;
  $("req-summary").textContent =
    `${counts.native} native · ${counts.compiled} compiled · ` +
    `${counts.partial} partial · ${counts.rejected} rejected`;

  const levels = ["all", "native", "compiled", "partial", "rejected"];
  $("req-filters").innerHTML = levels
    .map(
      (level) =>
        `<button class="filter${level === filter ? " active" : ""}" ` +
        `data-level="${level}">${level}` +
        `${level === "all" ? "" : ` (${counts[level]})`}</button>`,
    )
    .join("");

  const rows = caps.requirements.filter(
    (r) => filter === "all" || r.support === filter,
  );
  $("req-table").querySelector("tbody").innerHTML = rows
    .map(
      (r) =>
        `<tr>` +
        `<td><code>${escapeHtml(r.name)}</code></td>` +
        `<td>${escapeHtml(r.pddl)}</td>` +
        `<td><span class="pill ${r.support}" title="${SUPPORT_BLURB[r.support]}">` +
        `${r.support}</span></td>` +
        `<td><strong>${escapeHtml(r.summary)}</strong>` +
        (r.note ? `<br /><span class="note">${escapeHtml(r.note)}</span>` : "") +
        `</td></tr>`,
    )
    .join("");
}

/* --------------------------------------------------------------- generate */
function updateGeneratorBlurb() {
  const caps = state.capabilities;
  if (!caps) return;
  const kind = $("gen-kind").value;
  const entry = caps.generators.find((g) => g.name === kind);
  $("gen-blurb").textContent = entry ? entry.summary : "";
}

function generate() {
  if (state.running) return;
  clearError();
  setRunning(true);
  $("gen-domain").textContent = "generating…";
  $("gen-problem").textContent = "";
  state.worker.postMessage({
    type: "generate",
    kind: $("gen-kind").value,
    size: Number($("gen-size").value),
    seed: Number($("gen-seed").value),
  });
}

function showGenerated(payload) {
  setRunning(false);
  state.generated = payload;
  $("gen-domain").textContent = payload.domain;
  $("gen-problem").textContent = payload.problem;
  $("gen-to-solve").disabled = false;
}

function openGeneratedInSolve() {
  if (!state.generated) return;
  $("domain").value = state.generated.domain;
  $("problem").value = state.generated.problem;
  $("blurb").textContent =
    `generated: ${state.generated.kind}, size ${state.generated.size}, ` +
    `seed ${state.generated.seed}`;
  $("tagrow").innerHTML = '<span class="chip">generated</span>';
  switchView("solve");
}

/* ------------------------------------------------------------------ views */
function switchView(name) {
  for (const section of document.querySelectorAll(".view")) {
    section.classList.toggle("active", section.id === `view-${name}`);
  }
  for (const link of document.querySelectorAll(".navlink")) {
    const active = link.dataset.view === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (name === "solve") requestAnimationFrame(repaint);
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ----------------------------------------------------------------- events */
$("run").addEventListener("click", run);
$("run-experiment").addEventListener("click", runExperiment);
$("generate").addEventListener("click", generate);
$("gen-to-solve").addEventListener("click", openGeneratedInSolve);
$("gen-kind").addEventListener("change", updateGeneratorBlurb);
$("demo").addEventListener("change", (e) => loadDemo(Number(e.target.value)));
$("export-csv").addEventListener("click", exportCsv);
$("export-json").addEventListener("click", exportJson);

$("inspect").addEventListener("click", () => {
  if (state.running) return;
  clearError();
  setRunning(true);
  state.worker.postMessage({
    type: "inspect", domain: $("domain").value, problem: $("problem").value,
  });
});

$("copy-plan").addEventListener("click", () => {
  const text = [...$("plan").querySelectorAll("li")]
    .map((li, i) => `${i + 1}. ${li.textContent}`)
    .join("\n");
  navigator.clipboard?.writeText(text);
  $("copy-plan").textContent = "Copied";
  setTimeout(() => { $("copy-plan").textContent = "Copy plan"; }, 1200);
});

for (const link of document.querySelectorAll(".navlink")) {
  link.addEventListener("click", () => switchView(link.dataset.view));
}

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

document.addEventListener("click", (event) => {
  const filter = event.target.closest(".filter");
  if (filter) renderRequirements(filter.dataset.level);

  const all = event.target.closest("[data-all]");
  if (all) {
    for (const input of $(all.dataset.all).querySelectorAll("input")) {
      input.checked = true;
    }
  }
  const none = event.target.closest("[data-none]");
  if (none) {
    for (const input of $(none.dataset.none).querySelectorAll("input")) {
      input.checked = false;
    }
  }

  const header = event.target.closest(".sortable th[data-key]");
  if (header) {
    const key = header.dataset.key;
    const exp = state.experiment;
    exp.sortDir = exp.sortKey === key ? -exp.sortDir : 1;
    exp.sortKey = key;
    for (const th of document.querySelectorAll(".sortable th")) {
      th.classList.remove("sorted-asc", "sorted-desc");
    }
    header.classList.add(exp.sortDir === 1 ? "sorted-asc" : "sorted-desc");
    renderExperimentTable();
  }
});

$("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.getAttribute("data-theme") === "dark"
    || (!root.hasAttribute("data-theme")
        && window.matchMedia("(prefers-color-scheme: dark)").matches);
  root.setAttribute("data-theme", dark ? "light" : "dark");
  renderLegend();
  repaint();
  if (state.experiment.rows.length) finishExperiment();
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    repaint();
    if (state.experiment.rows.length) finishExperiment();
  }, 120);
});

boot().catch((error) => showError(String(error)));
