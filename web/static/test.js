(async function () {
  const el = (id) => document.getElementById(id);
  const status = (t) => (el("status").textContent = t);

  // Config
  const cfg = await fetch("/config").then((r) => r.json());
  el("dims").textContent = `${cfg.input[1]}×${cfg.input[0]} (h×w)`;
  el("gpu").textContent = cfg.gpu ? "GPU (WebGPU) ✓" : "CPU fallback";
  el("model").textContent = `${cfg.model}`;
  let W = cfg.input[0],
    H = cfg.input[1],
    NPIX = W * H;

  // Health poll
  setInterval(async () => {
    try {
      const h = await fetch("/health").then((r) => r.json());
      el("inflight").textContent = h.inflight;
    } catch {}
  }, 1000);

  // Session + run capture
  let sessionActive = false;
  let session = null; // persisted across multiple runs if you want
  let collected = []; // always filled for the *current* run (even if no session)

  // UI enable/disable
  function setDownloadsEnabled(on) {
    el("dlJSON").disabled = !on;
    el("dlCSV").disabled = !on;
    // Save-to-server only uses session; enable if session has data
    el("saveServer").disabled = !(
      session &&
      session.results &&
      session.results.length > 0
    );
  }
  function toggleSessionButtons(active) {
    el("startSess").disabled = active;
    el("stopSess").disabled = !active;
    setDownloadsEnabled(collected.length > 0);
  }

  function ensureSession() {
    if (!session) {
      session = {
        id: `sess_${new Date().toISOString().replace(/[:.]/g, "-")}`,
        started_at: new Date().toISOString(),
        model: cfg.model,
        modelPath: cfg.modelPath,
        gpu: cfg.gpu,
        input: cfg.input,
        results: [],
      };
    }
  }

  function startSession() {
    if (sessionActive) return;
    ensureSession();
    sessionActive = true;
    toggleSessionButtons(true);
    status(`session ${session.id} started`);
  }
  function stopSession() {
    if (!sessionActive) return;
    sessionActive = false;
    session.stopped_at = new Date().toISOString();
    toggleSessionButtons(false);
    status(
      `session ${session.id} stopped (results: ${session.results.length})`
    );
  }

  function downloadJSON() {
    const obj =
      sessionActive || (session && session.results.length > 0)
        ? session
        : {
            id: `run_${new Date().toISOString().replace(/[:.]/g, "-")}`,
            model: cfg.model,
            gpu: cfg.gpu,
            input: cfg.input,
            results: collected,
          };
    const blob = new Blob([JSON.stringify(obj, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${obj.id}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  }
  function toCSV(rows) {
    const out = [];
    out.push(
      [
        "req_id",
        "mode",
        "started_at",
        "finished_at",
        "client_ms",
        "server_ms",
        "queue_ms",
        "top_index",
        "top_score",
        "probs",
      ].join(",")
    );
    for (const r of rows) {
      const probs = Array.isArray(r.probs) ? JSON.stringify(r.probs) : "";
      out.push(
        [
          r.req_id,
          r.mode,
          r.started_at,
          r.finished_at,
          fmtNum(r.client_ms),
          fmtNum(r.server_ms),
          fmtNum(r.queue_ms),
          r.top_index,
          r.top_score,
          probs,
        ].join(",")
      );
    }
    return out.join("\n");
  }
  function downloadCSV() {
    const rows =
      sessionActive || (session && session.results.length > 0)
        ? session.results
        : collected;
    const id =
      sessionActive || (session && session.results.length > 0)
        ? session.id
        : `run_${new Date().toISOString().replace(/[:.]/g, "-")}`;
    const blob = new Blob([toCSV(rows)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${id}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  }
  async function saveToServer() {
    if (!(session && session.results.length > 0)) {
      status("no session data to save");
      return;
    }
    const res = await fetch("/save-session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(session),
    }).then((r) => r.json());
    el("log").textContent =
      `saved=${res.saved} path=${res.path} bytes=${res.bytes}\n` +
      el("log").textContent;
  }

  function makeInput(kind) {
    const x = new Array(NPIX);
    if (kind === "zeros") for (let i = 0; i < NPIX; i++) x[i] = 0.0;
    else for (let i = 0; i < NPIX; i++) x[i] = Math.random();
    return x;
  }
  function percentiles(arr) {
    const a = arr.slice().sort((x, y) => x - y);
    const pick = (p) =>
      a[Math.max(0, Math.min(a.length - 1, Math.floor(p * (a.length - 1))))];
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    return {
      p50: pick(0.5),
      p90: pick(0.9),
      p99: pick(0.99),
      mean,
      max: a[a.length - 1],
    };
  }
  function drawChart(latencies) {
    const canvas = el("chart");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!latencies.length) return;
    const a = latencies.slice().sort((x, y) => x - y);
    const Wc = canvas.width,
      Hc = canvas.height,
      pad = 20;
    const n = a.length;
    const bar = Math.max(1, Math.floor((Wc - pad * 2) / n));
    const max = Math.max(...a);
    ctx.strokeStyle = "#2b2f59";
    ctx.strokeRect(pad, pad, Wc - pad * 2, Hc - pad * 2);
    ctx.fillStyle = "#7aa2ff";
    for (let i = 0; i < n; i++) {
      const h = Math.round((a[i] / max) * (Hc - pad * 2));
      ctx.fillRect(pad + i * bar, Hc - pad - h, bar - 1, h);
    }
  }

  // table / panels
  function appendRow(mode, rix, res, clientMs) {
    const tb = el("tbl").querySelector("tbody");
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${rix}</td>
      <td>${mode}</td>
      <td>${res.top_index}</td>
      <td>${num(res.top_score, 4)}</td>
      <td>${num(res.latency_ms, 2)}</td>
      <td>${num(res.queued_ms, 2)}</td>
      <td>${clientMs != null ? num(clientMs, 2) : "–"}</td>
      <td>${res.used_gpu ? "GPU" : "CPU"}</td>
      <td>${fmtTime(res.when)}</td>
      <td><button class="btn ghost btn-view" data-ix="${rix}" data-mode="${mode}">View</button></td>`;
    tb.insertBefore(tr, tb.firstChild);
    // Last output panel
    if (Array.isArray(res.probs)) {
      renderProbsPanel(res.probs);
    }
    // keep last ~100 rows
    while (tb.children.length > 100) tb.removeChild(tb.lastChild);
  }
  function renderProbsPanel(probs) {
    const k = Math.max(
      1,
      Math.min(100, parseInt(el("topk").value || "10", 10))
    );
    const pairs = probs
      .map((p, i) => [i, p])
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
    const lines = pairs.map(
      ([i, p]) => `${i.toString().padStart(3)}  ${p.toFixed(6)}`
    );
    el("probsPanel").textContent = lines.join("\n");
  }

  // modal for full vector
  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".btn-view");
    if (!btn) return;
    const ix = parseInt(btn.dataset.ix, 10);
    const mode = btn.dataset.mode;
    const r = mode === "server" ? collected[ix] : collected[ix];
    if (!r || !Array.isArray(r.probs)) return;
    el("modalContent").textContent = JSON.stringify(r.probs, null, 2);
    el("modal").classList.remove("hidden");
  });
  el("closeModal").addEventListener("click", () =>
    el("modal").classList.add("hidden")
  );
  el("copyProbs").addEventListener("click", () => {
    navigator.clipboard.writeText(el("probsPanel").textContent || "");
  });
  el("topk").addEventListener("change", () => {
    const last = collected.length ? collected[collected.length - 1] : null;
    if (last && Array.isArray(last.probs)) renderProbsPanel(last.probs);
  });

  function recordResult(mode, req_id, res, clientMs) {
    const item = {
      req_id,
      mode,
      started_at: res.when || new Date().toISOString(),
      finished_at: new Date().toISOString(),
      client_ms: clientMs,
      server_ms: res.latency_ms,
      queue_ms: res.queued_ms,
      top_index: res.top_index,
      top_score: res.top_score,
      probs: res.probs,
    };
    collected.push(item);
    if (sessionActive) {
      ensureSession();
      session.results.push(item);
    }
    setDownloadsEnabled(collected.length > 0);
  }

  async function clientMode(N, parallel, kind) {
    const x = makeInput(kind);
    const body = JSON.stringify({ input: x });
    const headers = { "content-type": "application/json" };
    collected = [];
    const startAll = performance.now();
    const lat = [];
    status(`running (${N} requests @ parallel=${parallel})`);

    async function one(ix) {
      const t0 = performance.now();
      const r = await fetch("/infer", { method: "POST", headers, body });
      const js = await r.json();
      const dt = performance.now() - t0;
      lat.push(dt);
      recordResult("client", ix, js, dt);
      appendRow("client", ix, js, dt);
      return js;
    }

    const runPool = [];
    let next = 0;
    for (let i = 0; i < parallel; i++) {
      runPool.push(
        (async () => {
          while (true) {
            const ix = next++;
            if (ix >= N) break;
            try {
              await one(ix);
            } catch (e) {
              console.error(e);
            }
          }
        })()
      );
    }
    await Promise.all(runPool);

    const total = performance.now() - startAll;
    finalizeRun(lat, total);
  }

  async function serverMode(N, kind) {
    const x = makeInput(kind);
    collected = [];
    const res = await fetch("/blast", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ n: N, input: x }),
    }).then((r) => r.json());

    const lat = res.results.map((r) => r.latency_ms);
    res.results.forEach((r, i) => {
      recordResult("server", i, r, null);
      appendRow("server", i, r, null);
    });
    finalizeRun(lat, res.total_ms);
    el("log").textContent =
      `blast n=${res.count}, parallel=${res.parallel}\n` +
      res.results
        .slice(0, 10)
        .map((r, i) => `#${i} top=${r.top_index} lat=${num(r.latency_ms, 2)}ms`)
        .join("\n");
  }

  function finalizeRun(lat, totalMs) {
    const p = percentiles(lat);
    el("p50").textContent = num(p.p50, 1);
    el("p90").textContent = num(p.p90, 1);
    el("p99").textContent = num(p.p99, 1);
    el("mean").textContent = num(p.mean, 1);
    el("max").textContent = num(p.max, 1);
    el("total").textContent = num(totalMs, 1);
    drawChart(lat);
    status("done");
    setDownloadsEnabled(collected.length > 0);
  }

  // helpers
  function num(v, d = 2) {
    return v == null || isNaN(v) ? "–" : (+v).toFixed(d);
  }
  function fmtNum(v) {
    return v == null || isNaN(v) ? "" : (+v).toFixed(3);
  }
  function fmtTime(iso) {
    try {
      return new Date(iso).toLocaleTimeString();
    } catch {
      return "";
    }
  }

  // run button
  el("run").addEventListener("click", async () => {
    const mode = el("mode").value;
    const N = parseInt(el("count").value, 10);
    const P = parseInt(el("parallel").value, 10);
    const kind = el("inputkind").value;
    // clear table + metrics
    el("log").textContent = "";
    ["p50", "p90", "p99", "mean", "max", "total"].forEach(
      (id) => (el(id).textContent = "–")
    );
    const tb = el("tbl").querySelector("tbody");
    tb.innerHTML = "";
    el("probsPanel").textContent = "";
    status("starting…");
    try {
      if (mode === "client") await clientMode(N, P, kind);
      else await serverMode(N, kind);
    } catch (e) {
      status("error");
      el("log").textContent = String(e) + "\n" + el("log").textContent;
    }
  });

  // session buttons
  el("startSess").addEventListener("click", startSession);
  el("stopSess").addEventListener("click", stopSession);
  el("dlJSON").addEventListener("click", downloadJSON);
  el("dlCSV").addEventListener("click", downloadCSV);
  el("saveServer").addEventListener("click", saveToServer);

  // init
  toggleSessionButtons(false);
})();
