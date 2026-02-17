const state = {
  currentJobId: null,
  pollTimer: null,
  jobsTimer: null,
};

const els = {
  apiHealth: document.getElementById("apiHealth"),
  runForm: document.getElementById("runForm"),
  cancelBtn: document.getElementById("cancelBtn"),
  logConsole: document.getElementById("logConsole"),
  commandPreview: document.getElementById("commandPreview"),
  metricStatus: document.getElementById("metricStatus"),
  metricCode: document.getElementById("metricCode"),
  metricStart: document.getElementById("metricStart"),
  metricEnd: document.getElementById("metricEnd"),
  summaryGrid: document.getElementById("summaryGrid"),
  resultBody: document.getElementById("resultBody"),
  outputExcerptWrap: document.getElementById("outputExcerptWrap"),
  outputExcerpt: document.getElementById("outputExcerpt"),
  jobHistory: document.getElementById("jobHistory"),
  inputPath: document.getElementById("inputPath"),
  outputDir: document.getElementById("outputDir"),
  browseInputFile: document.getElementById("browseInputFile"),
  browseInputDir: document.getElementById("browseInputDir"),
  browseOutputDir: document.getElementById("browseOutputDir"),
};

function text(v) {
  if (v === null || v === undefined || v === "") return "-";
  return String(v);
}

function fixed(v) {
  if (v === null || v === undefined || v === "") return "";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  return n.toFixed(4);
}

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload.error || `Request failed: ${res.status}`);
  }
  return payload;
}

function payloadFromForm() {
  return {
    input_path: document.getElementById("inputPath").value.trim(),
    output_dir: document.getElementById("outputDir").value.trim() || null,
    processing: document.getElementById("processing").value,
    workers: document.getElementById("workers").value.trim() || null,
    charge: Number(document.getElementById("charge").value || 0),
    spin: Number(document.getElementById("spin").value || 1),
    ram_per_job: Number(document.getElementById("ramPerJob").value || 50),
    force: document.getElementById("force").checked,
    clean: document.getElementById("clean").checked,
    debug: document.getElementById("debug").checked,
    storage_efficient: document.getElementById("storageEfficient").checked,
    refresh_autoconfig: document.getElementById("refreshAutoconfig").checked,
    quiet_config: document.getElementById("quietConfig").checked,
  };
}

function renderJobMetrics(job) {
  els.metricStatus.textContent = text(job?.status);
  els.metricCode.textContent = text(job?.returncode);
  els.metricStart.textContent = text(job?.started_at);
  els.metricEnd.textContent = text(job?.finished_at);
  els.commandPreview.textContent = (job?.command || []).join(" ") || "No command yet.";
}

function renderLogs(job) {
  const logs = job?.logs || [];
  els.logConsole.textContent = logs.join("\n");
  els.logConsole.scrollTop = els.logConsole.scrollHeight;
}

function renderSummary(summary) {
  const cards = [
    ["Total", summary?.total_files],
    ["Success", summary?.successful_files],
    ["Failed", summary?.failed_files],
    ["Total Time (s)", summary?.total_time_seconds],
  ];
  els.summaryGrid.innerHTML = cards
    .map(
      ([k, v]) =>
        `<article><p>${k}</p><strong>${text(v)}</strong></article>`
    )
    .join("");
}

function renderRows(rows) {
  if (!rows || rows.length === 0) {
    els.resultBody.innerHTML = `<tr><td colspan="12">No results yet.</td></tr>`;
    return;
  }
  els.resultBody.innerHTML = rows
    .map((r) => {
      return `<tr>
        <td>${text(r.File)}</td>
        <td>${fixed(r.f1)}</td>
        <td>${fixed(r.f2)}</td>
        <td>${fixed(r.f3)}</td>
        <td>${fixed(r.f4)}</td>
        <td>${fixed(r.f5)}</td>
        <td>${fixed(r.f6)}</td>
        <td>${fixed(r.f7)}</td>
        <td>${fixed(r.f8)}</td>
        <td>${fixed(r.f9)}</td>
        <td>${fixed(r.SNCI)}</td>
        <td>${fixed(r.SCDI)}</td>
      </tr>`;
    })
    .join("");
}

function renderPreview(preview) {
  renderSummary(preview?.summary || {});
  renderRows(preview?.rows || []);
  if (preview?.output_excerpt) {
    els.outputExcerptWrap.open = true;
    els.outputExcerpt.textContent = preview.output_excerpt;
  } else {
    els.outputExcerptWrap.open = false;
    els.outputExcerpt.textContent = "";
  }
}

function statusPill(status) {
  const cls = status || "queued";
  return `<span class="pill ${cls}">${cls}</span>`;
}

function renderHistory(jobs) {
  if (!jobs || jobs.length === 0) {
    els.jobHistory.innerHTML = "<li><span>-</span><span>No jobs yet.</span><span>-</span></li>";
    return;
  }

  els.jobHistory.innerHTML = jobs
    .slice(0, 14)
    .map((job) => {
      const id = job.id || "";
      const short = id ? id.slice(0, 8) : "-";
      const input = job.payload?.input_path || "-";
      const end = job.finished_at || job.started_at || job.created_at || "-";
      return `<li data-job-id="${id}">
        ${statusPill(job.status)}
        <span title="${input}">${input}</span>
        <span>${short} @ ${end}</span>
      </li>`;
    })
    .join("");

  els.jobHistory.querySelectorAll("li[data-job-id]").forEach((li) => {
    li.addEventListener("click", () => {
      state.currentJobId = li.getAttribute("data-job-id");
      pullCurrentJob();
    });
  });
}

async function pullCurrentJob() {
  if (!state.currentJobId) return;
  try {
    const job = await jsonFetch(`/api/jobs/${state.currentJobId}`);
    renderJobMetrics(job);
    renderLogs(job);
    renderPreview(job.result_preview || {});
  } catch (err) {
    els.logConsole.textContent += `\n[GUI] ${err.message}`;
  }
}

async function pullJobs() {
  try {
    const payload = await jsonFetch("/api/jobs");
    renderHistory(payload.jobs || []);
  } catch (err) {
    console.error(err);
  }
}

async function checkHealth() {
  try {
    const payload = await jsonFetch("/api/health");
    els.apiHealth.textContent = `API: online (${payload.time})`;
  } catch (err) {
    els.apiHealth.textContent = "API: offline";
  }
}

async function startRun(event) {
  event.preventDefault();
  const payload = payloadFromForm();
  try {
    const out = await jsonFetch("/api/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJobId = out.job_id;
    await pullCurrentJob();
    await pullJobs();
  } catch (err) {
    alert(err.message);
  }
}

async function cancelRun() {
  if (!state.currentJobId) return;
  try {
    await jsonFetch(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
    await pullCurrentJob();
  } catch (err) {
    alert(err.message);
  }
}

async function browsePath(mode, targetEl) {
  try {
    const payload = await jsonFetch("/api/dialog", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    if (payload.path) {
      targetEl.value = payload.path;
    }
  } catch (err) {
    alert(err.message);
  }
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  if (state.jobsTimer) clearInterval(state.jobsTimer);
  state.pollTimer = setInterval(pullCurrentJob, 1200);
  state.jobsTimer = setInterval(pullJobs, 2200);
}

function init() {
  els.runForm.addEventListener("submit", startRun);
  els.cancelBtn.addEventListener("click", cancelRun);
  els.browseInputFile.addEventListener("click", () => browsePath("file", els.inputPath));
  els.browseInputDir.addEventListener("click", () => browsePath("directory", els.inputPath));
  els.browseOutputDir.addEventListener("click", () => browsePath("directory", els.outputDir));
  checkHealth();
  pullJobs();
  startPolling();
}

init();
