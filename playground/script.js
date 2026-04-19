/* ================================================================
   SAIR Playground – client-side logic
   ================================================================ */

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const elApiUrl       = $("#api-url");
const elDataset      = $("#dataset-select");
const elCustomDatasetSection = $("#custom-dataset-section");
const elCustomDatasetInput = $("#custom-dataset-input");
const elConcurrency  = $("#concurrency");
const elBtnRun       = $("#btn-run");
const elBtnStop      = $("#btn-stop");
const elBtnTest      = $("#btn-test");
const elEq1          = $("#eq1-input");
const elEq2          = $("#eq2-input");
const elCheatsheet   = $("#cheatsheet-area");
const elBtnLoadCS    = $("#btn-load-cs");
const elBadge        = $("#status-badge");
const elProgressWrap = $("#progress-wrapper");
const elProgressFill = $("#progress-fill");
const elProgressLbl  = $("#progress-label");
const elTbody        = $("#results-tbody");
const elSingleResult = $("#single-result");
const elSingleBody   = $("#single-result-body");
const elHistoryList  = $("#history-list");
const elBtnClearHist = $("#btn-clear-history");

// Metric displays
const mAccuracy = $("#m-accuracy");
const mTotal    = $("#m-total");
const mTP       = $("#m-tp");
const mTN       = $("#m-tn");
const mFP       = $("#m-fp");
const mFN       = $("#m-fn");
const cmTP      = $("#cm-tp");
const cmTN      = $("#cm-tn");
const cmFP      = $("#cm-fp");
const cmFN      = $("#cm-fn");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let abortController = null;
let metrics = { tp: 0, tn: 0, fp: 0, fn: 0, total: 0, done: 0 };
const CUSTOM_DATASET_KEY = "sair_custom_dataset";

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
$$(".nav-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".nav-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    $$(".tab-content").forEach((tc) => tc.classList.remove("active"));
    const target = tab.dataset.tab;
    $(`#tab-${target}-content`).classList.add("active");
    if (target === "history") renderHistory();
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function apiBase() {
  return elApiUrl.value.replace(/\/+$/, "");
}

function setBadge(text, cls) {
  elBadge.textContent = text;
  elBadge.className = `badge badge-${cls}`;
}

function resetMetrics() {
  metrics = { tp: 0, tn: 0, fp: 0, fn: 0, total: 0, done: 0 };
  updateMetricsUI();
  elTbody.innerHTML = "";
}

function updateMetricsUI() {
  const { tp, tn, fp, fn, done, total } = metrics;
  const correct = tp + tn;
  const answered = tp + tn + fp + fn;
  const acc = answered > 0 ? ((correct / answered) * 100).toFixed(1) + "%" : "--";

  mAccuracy.textContent = acc;
  mTotal.textContent = done;
  mTP.textContent = tp;
  mTN.textContent = tn;
  mFP.textContent = fp;
  mFN.textContent = fn;
  cmTP.textContent = tp;
  cmTN.textContent = tn;
  cmFP.textContent = fp;
  cmFN.textContent = fn;

  if (total > 0) {
    elProgressFill.style.width = `${(done / total) * 100}%`;
    elProgressLbl.textContent = `${done} / ${total}`;
  }
}

function verdictSpan(v) {
  if (v === true) return `<span class="verdict-true">TRUE</span>`;
  if (v === false) return `<span class="verdict-false">FALSE</span>`;
  return `<span class="verdict-null">NULL</span>`;
}

function escHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function parsePredictedVerdict(result) {
  if (result?.verdict === true || result?.verdict === false) {
    return result.verdict;
  }

  const responseText = String(result?.response || "").toUpperCase();
  if (responseText.includes("VERDICT: TRUE")) return true;
  if (responseText.includes("VERDICT: FALSE")) return false;
  return null;
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------
function normalizeProblems(rawProblems) {
  if (!Array.isArray(rawProblems)) {
    throw new Error("Dataset must be an array of objects");
  }

  return rawProblems
    .map((item, idx) => {
      const equation1 = typeof item.equation1 === "string" ? item.equation1.trim() : "";
      const equation2 = typeof item.equation2 === "string" ? item.equation2.trim() : "";
      if (!equation1 || !equation2) {
        return null;
      }

      let answer = null;
      if (typeof item.answer === "boolean") {
        answer = item.answer;
      } else if (typeof item.answer === "string") {
        if (item.answer.toLowerCase() === "true") answer = true;
        if (item.answer.toLowerCase() === "false") answer = false;
      }

      return {
        index: item.index ?? idx,
        equation1,
        equation2,
        answer,
      };
    })
    .filter(Boolean);
}

function parseCustomDataset(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("Custom dataset is empty");
  }

  try {
    const parsed = JSON.parse(trimmed);
    const payload = Array.isArray(parsed) ? parsed : parsed.data ?? parsed;
    return normalizeProblems(payload);
  } catch {
    // Try JSONL fallback
    const rows = trimmed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
    return normalizeProblems(rows);
  }
}

async function loadDatasetCatalog() {
  const res = await fetch(`${apiBase()}/datasets`);
  if (!res.ok) {
    throw new Error(`Failed to list datasets: ${res.status}`);
  }
  const json = await res.json();
  const datasets = Array.isArray(json.datasets) ? json.datasets : [];
  return datasets;
}

async function refreshDatasetOptions() {
  const currentValue = elDataset.value;
  try {
    const datasets = await loadDatasetCatalog();
    const options = datasets
      .map((ds) => {
        const name = typeof ds === "string" ? ds : ds.name;
        return `<option value="${name}">${escHtml(name)}</option>`;
      })
      .join("");

    elDataset.innerHTML = `${options}<option value="custom">custom</option>`;
    const firstDataset = datasets[0] ? (typeof datasets[0] === "string" ? datasets[0] : datasets[0].name) : "custom";
    elDataset.value = datasets.some((ds) => (typeof ds === "string" ? ds : ds.name) === currentValue)
      ? currentValue
      : firstDataset;
  } catch {
    // Keep static options if the endpoint is unavailable.
  }
  toggleCustomDatasetInput();
}

function toggleCustomDatasetInput() {
  const isCustom = elDataset.value === "custom";
  elCustomDatasetSection.classList.toggle("hidden", !isCustom);
}

async function loadDataset(name) {
  if (name === "custom") {
    const problems = parseCustomDataset(elCustomDatasetInput.value);
    if (!problems.length) {
      throw new Error("Custom dataset has no valid rows");
    }
    return problems;
  }

  const url = `${apiBase()}/datasets/${name}`;
  const res = await fetch(url, { signal: abortController?.signal });
  if (!res.ok) throw new Error(`Failed to load dataset: ${res.status}`);
  const json = await res.json();
  // Support both {ok, data:[...]} wrapper and plain arrays
  const payload = Array.isArray(json) ? json : json.data ?? json;
  return normalizeProblems(payload);
}

function filterProblems(problems) {
  const filter = document.querySelector('input[name="gt-filter"]:checked').value;
  if (filter === "true") return problems.filter((p) => p.answer === true);
  if (filter === "false") return problems.filter((p) => p.answer === false);
  return problems;
}

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------
async function callSair(eq1, eq2, signal) {
  const res = await fetch(`${apiBase()}/sair`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ equation1: eq1, equation2: eq2 }),
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Add a result row to the table
// ---------------------------------------------------------------------------
function addResultRow(idx, eq1, eq2, truth, predicted, response, trace) {
  const match = truth != null && predicted != null ? truth === predicted : null;

  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${idx}</td>
    <td title="${escHtml(eq1)}">${escHtml(eq1)}</td>
    <td title="${escHtml(eq2)}">${escHtml(eq2)}</td>
    <td>${verdictSpan(truth)}</td>
    <td>${verdictSpan(predicted)}</td>
    <td class="${match === true ? "match-yes" : match === false ? "match-no" : ""}">${
      match === true ? "yes" : match === false ? "NO" : "-"
    }</td>
  `;

  // Build detail text from trace
  let detail = response || "";
  if (trace && trace.length) {
    detail += "\n\n--- Trace ---\n";
    for (const step of trace) {
      detail += `\n[${step.agent}] ${step.content}\n`;
    }
  }

  tr.addEventListener("click", () => {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("trace-row")) {
      next.remove();
    } else {
      const detailRow = document.createElement("tr");
      detailRow.className = "trace-row";
      detailRow.innerHTML = `<td colspan="6">${escHtml(detail)}</td>`;
      tr.after(detailRow);
    }
  });

  elTbody.appendChild(tr);
}

// ---------------------------------------------------------------------------
// Batch evaluation with concurrency control
// ---------------------------------------------------------------------------
async function runBatch(problems) {
  const concurrency = Math.max(1, Math.min(10, parseInt(elConcurrency.value) || 1));
  const signal = abortController.signal;

  let nextIdx = 0;

  async function worker() {
    while (nextIdx < problems.length) {
      if (signal.aborted) return;
      const i = nextIdx++;
      const p = problems[i];

      try {
        const result = await callSair(p.equation1, p.equation2, signal);
        const predicted = parsePredictedVerdict(result);

        // Classify
        if (p.answer != null && predicted != null) {
          if (p.answer === true && predicted === true) metrics.tp++;
          else if (p.answer === false && predicted === false) metrics.tn++;
          else if (p.answer === false && predicted === true) metrics.fp++;
          else if (p.answer === true && predicted === false) metrics.fn++;
        }

        metrics.done++;
        updateMetricsUI();
        addResultRow(
          p.index ?? i,
          p.equation1,
          p.equation2,
          p.answer,
          predicted,
          result.response,
          result.trace
        );
      } catch (err) {
        if (signal.aborted) return;
        metrics.done++;
        updateMetricsUI();
        addResultRow(p.index ?? i, p.equation1, p.equation2, p.answer, null, `ERROR: ${err.message}`, []);
      }
    }
  }

  const workers = [];
  for (let w = 0; w < concurrency; w++) {
    workers.push(worker());
  }
  await Promise.all(workers);
}

// ---------------------------------------------------------------------------
// Run button handler
// ---------------------------------------------------------------------------
elBtnRun.addEventListener("click", async () => {
  abortController = new AbortController();
  resetMetrics();
  setBadge("Running", "running");
  elBtnRun.disabled = true;
  elBtnStop.disabled = false;
  elProgressWrap.classList.remove("hidden");
  elSingleResult.classList.add("hidden");

  try {
    const dsName = elDataset.value;
    const problems = filterProblems(await loadDataset(dsName));
    if (problems.length === 0) {
      setBadge("No problems", "error");
      return;
    }

    metrics.total = problems.length;
    updateMetricsUI();

    await runBatch(problems);

    if (!abortController.signal.aborted) {
      setBadge("Done", "done");
      saveHistoryEntry(dsName, problems.length);
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      setBadge("Error", "error");
      console.error(err);
      elSingleResult.classList.remove("hidden");
      elSingleBody.innerHTML = `<span class="verdict-null">ERROR: ${escHtml(err.message)}</span>`;
    }
  } finally {
    elBtnRun.disabled = false;
    elBtnStop.disabled = true;
  }
});

// ---------------------------------------------------------------------------
// Stop button handler
// ---------------------------------------------------------------------------
elBtnStop.addEventListener("click", () => {
  if (abortController) {
    abortController.abort();
    setBadge("Stopped", "error");
    elBtnRun.disabled = false;
    elBtnStop.disabled = true;
  }
});

// ---------------------------------------------------------------------------
// Single test
// ---------------------------------------------------------------------------
elBtnTest.addEventListener("click", async () => {
  const eq1 = elEq1.value.trim();
  const eq2 = elEq2.value.trim();
  if (!eq1 || !eq2) return;

  elBtnTest.disabled = true;
  elSingleResult.classList.remove("hidden");
  elSingleBody.textContent = "Running...";
  setBadge("Running", "running");

  try {
    const result = await callSair(eq1, eq2);
    const predicted = parsePredictedVerdict(result);
    let html = `<span class="${
      predicted === true ? "verdict-true" : predicted === false ? "verdict-false" : "verdict-null"
    }">VERDICT: ${predicted === true ? "TRUE" : predicted === false ? "FALSE" : "NULL"}</span>\n\n`;

    html += escHtml(result.response || "");

    if (result.trace && result.trace.length) {
      html += "\n\n--- Trace ---\n";
      for (const step of result.trace) {
        html += `\n[${escHtml(step.agent)}] ${escHtml(step.content)}\n`;
      }
    }

    elSingleBody.innerHTML = html;
    setBadge("Done", "done");
  } catch (err) {
    elSingleBody.innerHTML = `<span class="verdict-null">ERROR: ${escHtml(err.message)}</span>`;
    setBadge("Error", "error");
  } finally {
    elBtnTest.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Cheat sheet
// ---------------------------------------------------------------------------
async function loadCheatsheet() {
  try {
    const res = await fetch(`${apiBase()}/cheatsheet`);
    if (res.ok) {
      const data = await res.json();
      elCheatsheet.value = data.content || "";
    }
  } catch {
    elCheatsheet.value = "(Could not load cheat sheet from server)";
  }
}

elBtnLoadCS.addEventListener("click", loadCheatsheet);

// Load on startup (deferred to avoid blocking if server is down)
setTimeout(loadCheatsheet, 500);

// ---------------------------------------------------------------------------
// History (localStorage)
// ---------------------------------------------------------------------------
const HISTORY_KEY = "sair_playground_history";

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistoryEntry(dataset, totalProblems) {
  const { tp, tn, fp, fn } = metrics;
  const correct = tp + tn;
  const answered = tp + tn + fp + fn;
  const entry = {
    date: new Date().toISOString(),
    dataset,
    total: totalProblems,
    answered,
    tp, tn, fp, fn,
    accuracy: answered > 0 ? ((correct / answered) * 100).toFixed(1) : "0.0",
  };
  const hist = getHistory();
  hist.unshift(entry);
  if (hist.length > 50) hist.length = 50;
  localStorage.setItem(HISTORY_KEY, JSON.stringify(hist));
}

function renderHistory() {
  const hist = getHistory();
  if (hist.length === 0) {
    elHistoryList.innerHTML = `<p style="color:var(--text-dim)">No evaluation runs yet.</p>`;
    return;
  }
  elHistoryList.innerHTML = hist
    .map(
      (h) => `
      <div class="history-card">
        <div class="history-meta">
          <strong>${escHtml(h.dataset)}</strong> &mdash; ${h.total} problems<br/>
          ${new Date(h.date).toLocaleString()}<br/>
          TP ${h.tp} &middot; TN ${h.tn} &middot; FP ${h.fp} &middot; FN ${h.fn}
        </div>
        <div class="history-accuracy">${h.accuracy}%</div>
      </div>
    `
    )
    .join("");
}

elBtnClearHist.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
// Restore API URL from localStorage if previously set
const savedUrl = localStorage.getItem("sair_api_url");
if (savedUrl) elApiUrl.value = savedUrl;

const savedCustomDataset = localStorage.getItem(CUSTOM_DATASET_KEY);
if (savedCustomDataset) elCustomDatasetInput.value = savedCustomDataset;

elApiUrl.addEventListener("change", () => {
  localStorage.setItem("sair_api_url", elApiUrl.value);
  refreshDatasetOptions();
});

elDataset.addEventListener("change", toggleCustomDatasetInput);

elCustomDatasetInput.addEventListener("input", () => {
  localStorage.setItem(CUSTOM_DATASET_KEY, elCustomDatasetInput.value);
});

refreshDatasetOptions();
