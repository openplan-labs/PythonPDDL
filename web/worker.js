/**
 * Playground worker: runs the real jupyddl inside Pyodide, off the UI thread.
 *
 * The library is stdlib-only, so there is no wheel to resolve and no micropip
 * round-trip — we write the package sources straight into Pyodide's filesystem
 * and import them. Search progress is streamed back to the page as batched
 * messages so the charts can animate while the planner is still working.
 *
 * The Python half lives in bootstrap.py and is fetched, not embedded, so it
 * stays a real source file.
 */

let pyodide = null;
let ready = false;

function post(type, payload) {
  self.postMessage({ type, payload });
}

async function init(pyodideBase, sources) {
  const bootstrap = await (await fetch("bootstrap.py")).text();
  // Resolve to an absolute URL: a dynamic import() of "vendor/..." would be
  // read as a bare module specifier and rejected.
  const base = new URL(pyodideBase, self.location.href).href;
  const { loadPyodide } = await import(`${base}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: base });

  // Write the package into the virtual filesystem, then make it importable.
  const FS = pyodide.FS;
  const made = new Set();
  for (const [path, text] of Object.entries(sources)) {
    const parts = path.split("/");
    parts.pop();
    let dir = "";
    for (const part of parts) {
      dir = dir ? `${dir}/${part}` : part;
      if (!made.has(dir)) {
        try { FS.mkdir(dir); } catch (e) { /* already there */ }
        made.add(dir);
      }
    }
    FS.writeFile(path, text, { encoding: "utf8" });
  }
  pyodide.runPython("import sys; sys.path.insert(0, '')");
  pyodide.runPython(bootstrap);
  ready = true;
  post("ready", JSON.parse(pyodide.runPython("describe()")));
}

/** Call a Python entry point, always releasing the proxy afterwards. */
function call(name, ...args) {
  const runner = pyodide.globals.get(name);
  try {
    return JSON.parse(runner(...args));
  } finally {
    runner.destroy();
  }
}

async function solve(request) {
  const emit = (json) => post("progress", JSON.parse(json));
  const started = performance.now();
  const parsed = call(
    "run_solve",
    request.domain, request.problem, request.planner,
    request.heuristic, request.weight ?? 2.0,
    JSON.stringify(request.limits || {}), emit,
  );
  parsed.wall = (performance.now() - started) / 1000;
  post("result", parsed);
}

async function experiment(request) {
  // Rows stream back as they finish so a long matrix fills the table live.
  const emit = (json) => post("experiment-row", JSON.parse(json).row);
  const parsed = call(
    "run_experiment",
    JSON.stringify(request.instances),
    JSON.stringify(request.configs),
    JSON.stringify(request.limits || {}),
    emit,
  );
  post("experiment-done", parsed);
}

async function inspect(request) {
  post("inspect", call("run_inspect", request.domain, request.problem));
}

async function generate(request) {
  post("generated", {
    ...call(
      "run_generate",
      request.kind, request.size, request.seed,
      JSON.stringify(request.extra || {}),
    ),
    kind: request.kind,
    size: request.size,
    seed: request.seed,
  });
}

self.onmessage = async (event) => {
  const { type } = event.data;
  try {
    if (type === "init") {
      await init(event.data.pyodideBase, event.data.sources);
    } else if (!ready) {
      post("error", { message: "the runtime is still loading" });
    } else if (type === "solve") {
      await solve(event.data);
    } else if (type === "experiment") {
      await experiment(event.data);
    } else if (type === "inspect") {
      await inspect(event.data);
    } else if (type === "generate") {
      await generate(event.data);
    }
  } catch (error) {
    // Pyodide surfaces Python exceptions with the traceback in .message.
    post("error", { message: String(error && error.message ? error.message : error) });
  }
};
