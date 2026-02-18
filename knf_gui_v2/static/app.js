/* ================================================================
   KNF GUI V3 - Application Logic
   ================================================================ */

const state = {
  currentJobId: null,
  pollTimer: null,
  jobsTimer: null,
  currentJobInFlight: false,
  jobsInFlight: false,
  currentJobStatus: null,
  viewer3d: null,
  multiwfnDetected: false,
  multiwfnPath: "",
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
  presetTorchCpu: document.getElementById("presetTorchCpu"),
  presetTorchGpu: document.getElementById("presetTorchGpu"),
  presetMultiwfn: document.getElementById("presetMultiwfn"),
  processingAutoBtn: document.getElementById("processingAutoBtn"),
  processingSingleBtn: document.getElementById("processingSingleBtn"),
  processingMultiBtn: document.getElementById("processingMultiBtn"),
  nciBackend: document.getElementById("nciBackend"),
  nciDevice: document.getElementById("nciDevice"),
  nciDtype: document.getElementById("nciDtype"),
  nciGridSpacing: document.getElementById("nciGridSpacing"),
  nciGridPadding: document.getElementById("nciGridPadding"),
  nciBatchSize: document.getElementById("nciBatchSize"),
  nciEigBatchSize: document.getElementById("nciEigBatchSize"),
  nciRhoFloor: document.getElementById("nciRhoFloor"),
  refreshFirstRun: document.getElementById("refreshFirstRun"),
  nciPrimitiveNorm: document.getElementById("nciPrimitiveNorm"),
  browseInputFile: document.getElementById("browseInputFile"),
  browseInputDir: document.getElementById("browseInputDir"),
  browseOutputDir: document.getElementById("browseOutputDir"),
  dropZone: document.getElementById("dropZone"),
  dropFilename: document.getElementById("dropFilename"),
  fileInput: document.getElementById("fileInput"),
  moleculeViewer: document.getElementById("moleculeViewer"),
  scatterPlot: document.getElementById("scatterPlot"),
  multiwfnModal: document.getElementById("multiwfnModal"),
  multiwfnPath: document.getElementById("multiwfnPath"),
  browseMultiwfnFile: document.getElementById("browseMultiwfnFile"),
  browseMultiwfnDir: document.getElementById("browseMultiwfnDir"),
  saveMultiwfnPath: document.getElementById("saveMultiwfnPath"),
  closeMultiwfnModal: document.getElementById("closeMultiwfnModal"),
  multiwfnModalNote: document.getElementById("multiwfnModalNote"),
};

/* ---- Helpers ---- */

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

/* ---- File Upload ---- */

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || "Upload failed");
  return payload;
}

function setupDropZone() {
  const zone = els.dropZone;
  const fileInput = els.fileInput;

  zone.addEventListener("click", () => fileInput.click());

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("dragover");
  });

  zone.addEventListener("drop", async (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) await handleFileUpload(file);
  });

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (file) await handleFileUpload(file);
  });
}

async function handleFileUpload(file) {
  try {
    els.dropFilename.textContent = `Uploading ${file.name}...`;
    const result = await uploadFile(file);
    els.inputPath.value = result.path;
    els.dropFilename.textContent = `✓ ${result.filename}`;
    fetchMolecule3D(result.path);
  } catch (err) {
    els.dropFilename.textContent = `✗ ${err.message}`;
  }
}

/* ---- 3D Molecule Viewer (3Dmol.js) ---- */

async function fetchMolecule3D(path) {
  if (!path) return;

  const container = els.moleculeViewer;
  container.innerHTML = "Loading 3D structure...";
  container.style.color = "#6e6e73";

  try {
    const res = await fetch(`/api/molecule-data?path=${encodeURIComponent(path)}`);
    if (!res.ok) {
      const err = await res.json();
      container.innerHTML = `<span style="color:#c0392b">${err.error || "Render failed"}</span>`;
      return;
    }

    const molData = await res.text();

    // Clear container and remove placeholder text
    container.innerHTML = "";
    container.style.color = "";

    // Create 3Dmol viewer
    if (state.viewer3d) {
      state.viewer3d.clear();
      state.viewer3d = null;
    }

    const viewer = $3Dmol.createViewer(container, {
      backgroundColor: "0xf2f4f9",
      antialias: true,
    });

    viewer.addModel(molData, "sdf");

    // Stick + ball style for a tactile, physical look
    viewer.setStyle({}, {
      stick: {
        radius: 0.14,
        colorscheme: "Jmol",
      },
      sphere: {
        scale: 0.28,
        colorscheme: "Jmol",
      },
    });

    viewer.zoomTo();
    viewer.spin("y", 0.6);
    viewer.render();

    state.viewer3d = viewer;

    // Stop spin on user interaction, resume on double-click
    container.addEventListener("mousedown", () => viewer.spin(false), { once: false });
    container.addEventListener("dblclick", () => viewer.spin("y", 0.6));

  } catch (err) {
    container.innerHTML = `<span style="color:#c0392b">${err.message}</span>`;
  }
}

/* ---- Scatter Plot (Batch) ---- */

function renderScatterPlot(rows) {
  if (!rows || rows.length < 2) {
    els.scatterPlot.style.display = "none";
    return;
  }

  const snciVals = rows.map((r) => Number(r.SNCI)).filter((v) => !isNaN(v));
  const scdiVals = rows.map((r) => Number(r.SCDI)).filter((v) => !isNaN(v));

  if (snciVals.length < 2 || scdiVals.length < 2) {
    els.scatterPlot.style.display = "none";
    return;
  }

  // Min-max normalize SNCI
  const snciMin = Math.min(...snciVals);
  const snciMax = Math.max(...snciVals);
  const snciRange = snciMax - snciMin || 1;

  const points = rows
    .filter((r) => !isNaN(Number(r.SNCI)) && !isNaN(Number(r.SCDI)))
    .map((r) => ({
      x: (Number(r.SNCI) - snciMin) / snciRange,
      y: Number(r.SCDI),
      rawSNCI: Number(r.SNCI),
      label: r.File || "unknown",
    }));

  const trace = {
    x: points.map((p) => p.x),
    y: points.map((p) => p.y),
    text: points.map((p) => `${p.label}<br>SNCI (raw): ${p.rawSNCI.toFixed(4)}<br>SCDI: ${p.y.toFixed(4)}`),
    mode: "markers+text",
    type: "scatter",
    textposition: "top center",
    textfont: { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif", size: 9, color: "#6e6e73" },
    marker: {
      size: 12,
      color: points.map((p) => p.y),
      colorscale: [
        [0, "#8fc3ff"],
        [0.5, "#2a8cff"],
        [1, "#005fc1"],
      ],
      line: { width: 1.5, color: "#6e6e73" },
    },
    hovertemplate: "%{text}<extra></extra>",
  };

  const layout = {
    title: {
      text: "SCDI vs Min-Max Normalised SNCI",
      font: { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif", size: 15, color: "#1d1d1f" },
    },
    xaxis: {
      title: { text: "Normalised SNCI (min-max)", font: { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif", size: 12 } },
      gridcolor: "#e6eaf1",
      linecolor: "#cad2df",
      linewidth: 1,
      zeroline: false,
    },
    yaxis: {
      title: { text: "SCDI", font: { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif", size: 12 } },
      gridcolor: "#e6eaf1",
      linecolor: "#cad2df",
      linewidth: 1,
      zeroline: false,
    },
    plot_bgcolor: "#f7f8fb",
    paper_bgcolor: "#f7f8fb",
    margin: { t: 50, r: 30, b: 60, l: 70 },
    font: { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif" },
  };

  els.scatterPlot.style.display = "block";
  Plotly.newPlot("scatterPlot", [trace], layout, { responsive: true });
}

/* ---- Form Payload ---- */

function payloadFromForm() {
  return {
    input_path: document.getElementById("inputPath").value.trim(),
    output_dir: document.getElementById("outputDir").value.trim() || null,
    processing: document.getElementById("processing").value,
    nci_backend: document.getElementById("nciBackend").value,
    nci_device: document.getElementById("nciDevice").value,
    nci_dtype: document.getElementById("nciDtype").value,
    nci_grid_spacing: Number(document.getElementById("nciGridSpacing").value || 0.2),
    nci_grid_padding: Number(document.getElementById("nciGridPadding").value || 3.0),
    nci_batch_size: Number(document.getElementById("nciBatchSize").value || 250000),
    nci_eig_batch_size: Number(document.getElementById("nciEigBatchSize").value || 200000),
    nci_rho_floor: Number(document.getElementById("nciRhoFloor").value || 1e-12),
    workers: document.getElementById("workers").value.trim() || null,
    charge: Number(document.getElementById("charge").value || 0),
    spin: Number(document.getElementById("spin").value || 1),
    ram_per_job: Number(document.getElementById("ramPerJob").value || 50),
    force: document.getElementById("force").checked,
    clean: document.getElementById("clean").checked,
    debug: document.getElementById("debug").checked,
    storage_efficient: document.getElementById("storageEfficient").checked,
    refresh_autoconfig: document.getElementById("refreshAutoconfig").checked,
    refresh_first_run: document.getElementById("refreshFirstRun").checked,
    quiet_config: document.getElementById("quietConfig").checked,
    nci_apply_primitive_norm: document.getElementById("nciPrimitiveNorm").checked,
    multiwfn_path: state.multiwfnPath || null,
  };
}

function isMultiwfnBackendSelected() {
  return (els.nciBackend?.value || "torch") === "multiwfn";
}

function showMultiwfnModal(message = "") {
  els.multiwfnModal.classList.remove("hidden");
  if (state.multiwfnPath && !els.multiwfnPath.value) {
    els.multiwfnPath.value = state.multiwfnPath;
  }
  els.multiwfnModalNote.textContent = message || "";
}

function hideMultiwfnModal() {
  els.multiwfnModal.classList.add("hidden");
}

async function refreshDependencyStatus(requireMultiwfn = false, showPromptIfMissing = false) {
  const payload = await jsonFetch("/api/dependencies");
  const dep = payload?.multiwfn || {};
  state.multiwfnDetected = !!dep.detected;
  state.multiwfnPath = dep.registered_path || state.multiwfnPath || "";
  if (state.multiwfnPath) {
    els.multiwfnPath.value = state.multiwfnPath;
  }
  if (requireMultiwfn && !state.multiwfnDetected && showPromptIfMissing) {
    showMultiwfnModal("Multiwfn is required before launching jobs.");
  } else if (!requireMultiwfn || state.multiwfnDetected) {
    hideMultiwfnModal();
  }
}

function syncBackendControls() {
  const useMultiwfn = isMultiwfnBackendSelected();
  if (useMultiwfn) {
    els.nciDevice.value = "auto";
    els.nciDevice.disabled = true;
    els.nciDtype.disabled = true;
    els.nciGridSpacing.disabled = true;
    els.nciGridPadding.disabled = true;
    els.nciBatchSize.disabled = true;
    els.nciEigBatchSize.disabled = true;
    els.nciRhoFloor.disabled = true;
    els.nciPrimitiveNorm.disabled = true;
  } else {
    els.nciDevice.disabled = false;
    els.nciDtype.disabled = false;
    els.nciGridSpacing.disabled = false;
    els.nciGridPadding.disabled = false;
    els.nciBatchSize.disabled = false;
    els.nciEigBatchSize.disabled = false;
    els.nciRhoFloor.disabled = false;
    els.nciPrimitiveNorm.disabled = false;
  }
}

function applyPreset(preset) {
  if (preset === "torch-cpu") {
    els.nciBackend.value = "torch";
    els.nciDevice.value = "cpu";
    els.nciDtype.value = "float32";
    els.nciGridSpacing.value = "0.2";
    els.nciGridPadding.value = "3.0";
    els.nciBatchSize.value = "250000";
    els.nciEigBatchSize.value = "200000";
    els.nciRhoFloor.value = "1e-12";
  } else if (preset === "torch-gpu") {
    els.nciBackend.value = "torch";
    els.nciDevice.value = "cuda";
    els.nciDtype.value = "float64";
    els.nciGridSpacing.value = "0.2";
    els.nciGridPadding.value = "3.0";
    els.nciBatchSize.value = "250000";
    els.nciEigBatchSize.value = "200000";
    els.nciRhoFloor.value = "1e-12";
  } else if (preset === "multiwfn") {
    els.nciBackend.value = "multiwfn";
    els.nciDevice.value = "auto";
  }
  syncBackendControls();
  refreshDependencyStatus(isMultiwfnBackendSelected(), false).catch(() => {});
}

function applyProcessingMode(mode) {
  if (!["auto", "single", "multi"].includes(mode)) return;
  document.getElementById("processing").value = mode;
}

function sanitizeNumericPayload(payload) {
  // Keep the command clean by not sending NaN/invalid values.
  const fallback = {
    nci_grid_spacing: 0.2,
    nci_grid_padding: 3.0,
    nci_batch_size: 250000,
    nci_eig_batch_size: 200000,
    nci_rho_floor: 1e-12,
  };
  for (const [key, value] of Object.entries(fallback)) {
    if (!Number.isFinite(payload[key])) {
      payload[key] = value;
    }
  }
}

async function saveMultiwfnPath() {
  const rawPath = (els.multiwfnPath.value || "").trim();
  if (!rawPath) {
    showMultiwfnModal("Please enter a valid path first.");
    return;
  }
  try {
    const out = await jsonFetch("/api/multiwfn-path", {
      method: "POST",
      body: JSON.stringify({ path: rawPath }),
    });
    state.multiwfnPath = out.path || rawPath;
    await refreshDependencyStatus(false);
    if (!state.multiwfnDetected) {
      showMultiwfnModal("Path saved, but Multiwfn is still not detected.");
      return;
    }
    hideMultiwfnModal();
  } catch (err) {
    showMultiwfnModal(err.message);
  }
}

/* ---- Rendering ---- */

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

  // Scatter plot for batch results
  if (preview?.type === "batch" && preview?.rows?.length >= 2) {
    renderScatterPlot(preview.rows);
  } else {
    els.scatterPlot.style.display = "none";
  }

  // 3D Molecule viewer for single results
  if (preview?.type === "single" && preview?.artifacts?.knf_json) {
    const inputPath = els.inputPath.value.trim();
    if (inputPath) fetchMolecule3D(inputPath);
  }

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
      state.currentJobStatus = null;
      pullCurrentJob();
    });
  });
}

/* ---- API Polling ---- */

async function pullCurrentJob() {
  if (!state.currentJobId) return;
  if (state.currentJobInFlight) return;
  if (["completed", "succeeded", "failed", "cancelled"].includes(state.currentJobStatus)) return;
  state.currentJobInFlight = true;
  try {
    const job = await jsonFetch(`/api/jobs/${state.currentJobId}`);
    state.currentJobStatus = job?.status || null;
    renderJobMetrics(job);
    renderLogs(job);
    renderPreview(job.result_preview || {});
  } catch (err) {
    els.logConsole.textContent += `\n[GUI] ${err.message}`;
  } finally {
    state.currentJobInFlight = false;
  }
}

async function pullJobs() {
  if (state.jobsInFlight) return;
  state.jobsInFlight = true;
  try {
    const payload = await jsonFetch("/api/jobs");
    renderHistory(payload.jobs || []);
  } catch (err) {
    console.error(err);
  } finally {
    state.jobsInFlight = false;
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

/* ---- Actions ---- */

async function startRun(event) {
  event.preventDefault();
  const payload = payloadFromForm();
  sanitizeNumericPayload(payload);
  const requireMultiwfn = payload.nci_backend === "multiwfn";
  try {
    await refreshDependencyStatus(requireMultiwfn, true);
  } catch (err) {
    alert(err.message);
    return;
  }
  if (requireMultiwfn && !state.multiwfnDetected) {
    showMultiwfnModal("Set Multiwfn path to continue.");
    return;
  }
  try {
    const out = await jsonFetch("/api/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJobId = out.job_id;
    state.currentJobStatus = "running";
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

async function browseMultiwfnPath(mode) {
  try {
    const payload = await jsonFetch("/api/dialog", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    if (payload.path) {
      els.multiwfnPath.value = payload.path;
      els.multiwfnModalNote.textContent = "";
    }
  } catch (err) {
    showMultiwfnModal(err.message);
  }
}

/* ---- Init ---- */

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
  els.presetTorchCpu.addEventListener("click", () => applyPreset("torch-cpu"));
  els.presetTorchGpu.addEventListener("click", () => applyPreset("torch-gpu"));
  els.presetMultiwfn.addEventListener("click", () => applyPreset("multiwfn"));
  els.processingAutoBtn.addEventListener("click", () => applyProcessingMode("auto"));
  els.processingSingleBtn.addEventListener("click", () => applyProcessingMode("single"));
  els.processingMultiBtn.addEventListener("click", () => applyProcessingMode("multi"));
  els.nciBackend.addEventListener("change", () => {
    syncBackendControls();
    refreshDependencyStatus(isMultiwfnBackendSelected(), false).catch(() => {});
  });
  els.browseMultiwfnFile.addEventListener("click", () => browseMultiwfnPath("multiwfn_file"));
  els.browseMultiwfnDir.addEventListener("click", () => browseMultiwfnPath("multiwfn_directory"));
  els.saveMultiwfnPath.addEventListener("click", saveMultiwfnPath);
  els.closeMultiwfnModal.addEventListener("click", hideMultiwfnModal);
  setupDropZone();
  syncBackendControls();
  checkHealth();
  refreshDependencyStatus(isMultiwfnBackendSelected(), true).catch((err) => {
    if (isMultiwfnBackendSelected()) {
      showMultiwfnModal(err.message);
    }
  });
  pullJobs();
  startPolling();
}

init();
