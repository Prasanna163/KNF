/**
 * api.ts — Central client for the knf_core FastAPI backend (knf_core/api.py).
 *
 * This backend is job-oriented (one molecular file per job) with an in-memory
 * job store — there is no run/batch grouping, no WebSocket, and no server-side
 * persistence across restarts. See src/lib/runs.ts for the client-side "Run"
 * concept built on top of these jobs, and src/lib/descriptors.ts for the
 * normalization/quadrant math ported from knf_core/engine/quadrants.py.
 */
import type { JobStatus } from '@/types';

export function apiBaseUrl(): string {
  try {
    const raw = localStorage.getItem('knf-settings');
    if (!raw) return 'http://127.0.0.1:8000';
    const parsed = JSON.parse(raw);
    let url = (parsed.apiBaseUrl || 'http://127.0.0.1:8000').trim().replace(/\/+$/, '');
    if (url.includes(':8766')) {
      url = url.replace(':8766', ':8000');
    }
    return url;
  } catch {
    return 'http://127.0.0.1:8000';
  }
}

// ── Server-side job status → frontend JobStatus ────────────────────────────
export type ServerJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

const JOB_STATUS_MAP: Record<ServerJobStatus, JobStatus> = {
  queued: 'queued',
  running: 'running',
  succeeded: 'success',
  failed: 'failed',
  cancelled: 'stopped',
};

export function mapJobStatus(status: string | undefined | null): JobStatus {
  return JOB_STATUS_MAP[(status || '') as ServerJobStatus] || 'failed';
}

// ── RunOptions accepted by /jobs/upload and /jobs/path ─────────────────────
export interface KnfRunOptions {
  batch_id?: string | null;
  charge?: number;
  spin?: number;
  water?: boolean;
  hydration_fragment_mode?: boolean;
  force?: boolean;
  clean?: boolean;
  debug?: boolean;
  full_files?: boolean;
  nci_backend?: 'torch' | 'multiwfn';
  nci_grid_spacing?: number;
  nci_grid_padding?: number;
  nci_device?: 'cpu' | 'cuda';
  nci_dtype?: string;
  nci_batch_size?: number;
  nci_eig_batch_size?: number;
  nci_rho_floor?: number;
  nci_apply_primitive_norm?: boolean;
  scdi_var_min?: number | null;
  scdi_var_max?: number | null;
  compute_scdi?: boolean;
  wbo_mode?: 'native' | 'xtb';
  preopt?: 'uff' | 'geoinit';
  xtb_engine?: 'xtb' | 'xtbx' | 'auto';
  xtb_gpu_atoms?: number;
  sp?: boolean;
  output_dir?: string | null;
  multiwfn_path?: string | null;
}

// ── knf.json shape (as parsed server-side and embedded in job summaries) ───
export interface KnfJson {
  SNCI?: number;
  SCDI?: number;
  SCDI_variance?: number;
  SNCI_Norm?: number | null;
  SCDI_Norm?: number | null;
  KNF_vector?: (number | null)[];
  metadata?: Record<string, unknown>;
  error?: string;
}

export interface JobArtifact {
  name: string;
  size_bytes: number;
  download_url: string;
}

export interface JobSummary {
  job_id: string;
  status: ServerJobStatus | string;
  kind: 'path' | 'upload';
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number | null;
  input_path: string | null;
  output_root: string | null;
  result_dir: string | null;
  managed_workspace: boolean;
  error: string | null;
  options: KnfRunOptions;
  artifacts: JobArtifact[];
  knf_json_path: string;
  knf_json: KnfJson | null;
  output_txt_path: string;
  batch_normalization?: {
    batch_id: string;
    state: 'provisional' | 'final';
    SNCI_method: string;
    SCDI_method: string;
    SNCI_min: number | null;
    SNCI_max: number | null;
    SCDI_variance_min: number | null;
    SCDI_variance_max: number | null;
    valid_SNCI_count: number;
    valid_SCDI_count: number;
  };
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return body.detail || res.statusText;
  } catch {
    return res.statusText;
  }
}

export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function health(): Promise<{ status: string; job_counts: Record<string, number> }> {
  const res = await fetch(`${apiBaseUrl()}/health`, { signal: AbortSignal.timeout(3000) });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function listJobs(): Promise<JobSummary[]> {
  const res = await fetch(`${apiBaseUrl()}/jobs`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  const data = await res.json();
  return data.jobs || [];
}

export async function getJob(jobId: string): Promise<JobSummary> {
  const res = await fetch(`${apiBaseUrl()}/jobs/${jobId}`);
  if (!res.ok) throw new ApiError(await parseErrorDetail(res), res.status);
  return res.json();
}

export async function submitUploadJob(file: File, options: KnfRunOptions): Promise<JobSummary> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('options_json', JSON.stringify(options));
  const res = await fetch(`${apiBaseUrl()}/jobs/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export function downloadArtifactUrl(jobId: string, artifactName: string): string {
  return `${apiBaseUrl()}/jobs/${jobId}/download/${encodeURIComponent(artifactName)}`;
}

export async function getInputContent(jobId: string): Promise<string> {
  const res = await fetch(`${apiBaseUrl()}/jobs/${jobId}/input`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.text();
}

export async function getArtifactText(jobId: string, artifactName: string): Promise<string> {
  const res = await fetch(downloadArtifactUrl(jobId, artifactName));
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.text();
}

export async function deleteJob(jobId: string): Promise<{ job_id: string; deleted: boolean }> {
  const res = await fetch(`${apiBaseUrl()}/jobs/${jobId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}
