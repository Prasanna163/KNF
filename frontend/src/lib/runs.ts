/**
 * runs.ts — Client-side "Run" registry.
 *
 * knf_core/api.py has no run/batch concept — every uploaded file becomes an
 * independent job in an in-memory store. A "Run" here is just a named group
 * of job IDs submitted together in one UI action, persisted in localStorage
 * so the Dashboard/Run Manager/Run Details pages have something to group by.
 * It does not survive a backend restart (the job IDs it references will
 * 404) — callers should treat a missing job as orphaned, not crash.
 */
import type { Run, RunConfig, RunStatus } from '@/types';
import { ApiError, getJob, mapJobStatus, type JobSummary, type KnfRunOptions } from './api';

/** Maps the frontend's RunConfig (Run Manager form state) to knf_core's RunOptions payload. */
export function toKnfRunOptions(config: RunConfig): KnfRunOptions {
  const options: KnfRunOptions = {
    charge: config.charge,
    spin: config.spin,
    force: config.forceRecomputation,
    clean: config.cleanOutputs,
    debug: config.debugMode,
    nci_backend: config.nciBackend,
    nci_device: (config.nciDevice as 'cpu' | 'cuda') || (config.gpuEnabled ? 'cuda' : 'cpu'),
    compute_scdi: true,
  };
  if (config.outputDirectory && config.outputDirectory !== './output/' && config.outputDirectory !== 'output') {
    options.output_dir = config.outputDirectory;
  }
  if (config.gridSpacing != null) options.nci_grid_spacing = config.gridSpacing;
  if (config.gridPadding != null) options.nci_grid_padding = config.gridPadding;
  if (config.batchSize != null) options.nci_batch_size = config.batchSize;
  if (config.eigBatchSize != null) options.nci_eig_batch_size = config.eigBatchSize;
  if (config.rhoFloor != null) options.nci_rho_floor = config.rhoFloor;
  if (config.scdiVarMin != null) options.scdi_var_min = config.scdiVarMin;
  if (config.scdiVarMax != null) options.scdi_var_max = config.scdiVarMax;
  return options;
}

export interface RunRecord {
  id: string;
  name: string;
  createdAt: string;
  jobIds: string[];
  config: RunConfig;
}

const STORAGE_KEY = 'knf-run-registry';

function loadRegistry(): RunRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveRegistry(records: RunRecord[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
}

export function listRunRecords(): RunRecord[] {
  return loadRegistry().sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
}

export function getRunRecord(id: string): RunRecord | undefined {
  return loadRegistry().find(r => r.id === id);
}

export function createRunId(): string {
  return `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createRunRecord(
  name: string,
  jobIds: string[],
  config: RunConfig,
  id = createRunId(),
): RunRecord {
  const record: RunRecord = {
    id,
    name,
    createdAt: new Date().toISOString(),
    jobIds,
    config,
  };
  const registry = loadRegistry();
  registry.push(record);
  saveRegistry(registry);
  return record;
}

export function deleteRunRecord(id: string): void {
  const registry = loadRegistry().filter(r => r.id !== id);
  saveRegistry(registry);
}

function aggregateStatus(jobs: JobSummary[], missingJobs = 0, unavailableJobs = 0): RunStatus {
  if (unavailableJobs > 0) return 'processing';
  if (jobs.length === 0) return missingJobs > 0 ? 'failed' : 'queued';
  const statuses = jobs.map(j => mapJobStatus(j.status));
  if (statuses.some(s => s === 'running')) return 'processing';
  if (statuses.every(s => s === 'queued')) return 'queued';
  if (statuses.some(s => s === 'queued')) return 'processing';
  const successCount = statuses.filter(s => s === 'success').length;
  const failedOrStopped = statuses.filter(s => s === 'failed' || s === 'stopped').length + missingJobs;
  if (successCount === 0 && failedOrStopped > 0) return 'failed';
  return 'completed';
}

/** Fetches every job in a run record and builds the aggregate `Run` the UI expects. */
export async function hydrateRun(record: RunRecord): Promise<Run> {
  const results = await Promise.all(
    record.jobIds.map(id =>
      getJob(id).catch(error =>
        error instanceof ApiError && error.status === 404
          ? null as JobSummary | null
          : undefined,
      ),
    ),
  );
  const jobs = results.filter((j): j is JobSummary => j !== null && j !== undefined);
  // The backend job store is intentionally in-memory. After an API restart,
  // persisted client-side run IDs return 404. Count those orphaned IDs as
  // terminal failures so the UI does not poll them forever as queued jobs.
  const missingFiles = results.filter(j => j === null).length;
  const unavailableFiles = results.filter(j => j === undefined).length;

  const statuses = jobs.map(j => mapJobStatus(j.status));
  const successFiles = statuses.filter(s => s === 'success').length;
  const failedFiles = statuses.filter(s => s === 'failed').length + missingFiles;
  const stoppedFiles = statuses.filter(s => s === 'stopped').length;
  const completedFiles = successFiles + failedFiles + stoppedFiles;

  const elapsedMs = jobs.reduce((acc, j) => acc + (j.elapsed_seconds || 0) * 1000, 0);

  const startedTimes = jobs.map(j => j.started_at).filter((t): t is string => !!t).sort();
  const finishedTimes = jobs.map(j => j.finished_at).filter((t): t is string => !!t).sort();

  return {
    id: record.id,
    name: record.name,
    status: aggregateStatus(jobs, missingFiles, unavailableFiles),
    config: record.config,
    files: [],
    createdAt: record.createdAt,
    startedAt: startedTimes[0],
    completedAt: completedFiles === record.jobIds.length ? finishedTimes[finishedTimes.length - 1] : undefined,
    totalFiles: record.jobIds.length,
    completedFiles,
    successFiles,
    failedFiles,
    stoppedFiles,
    elapsedMs,
  };
}

export async function hydrateAllRuns(): Promise<Run[]> {
  const records = listRunRecords();
  return Promise.all(records.map(hydrateRun));
}

/** Reverse lookup: which run (if any) a job belongs to, for pages that list jobs globally. */
export function getRunIdForJob(jobId: string): string {
  const record = loadRegistry().find(r => r.jobIds.includes(jobId));
  return record?.id ?? jobId;
}
