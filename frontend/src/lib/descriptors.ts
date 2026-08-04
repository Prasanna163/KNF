/**
 * descriptors.ts — Client-side port of knf_core/engine/quadrants.py's
 * `_compute_norm_and_quadrants` (pure math only, no plotting).
 *
 * knf_core/api.py never calls this — it's a CLI-batch-only step — so the API
 * only ever hands us raw SNCI/SCDI/SCDI_variance per job. This reproduces the
 * min-max normalization + median-split quadrant classification over whatever
 * job set the caller passes in, so Results/Explorer/RunDetails keep working.
 */
import type { Quadrant, ResultRecord } from '@/types';
import { mapJobStatus, type JobSummary } from './api';

export interface DescriptorInput {
  id: string;
  snci: number | null;
  scdi: number | null;
  scdiVariance: number | null;
}

export interface DescriptorNorm {
  id: string;
  snciNorm: number;
  scdiNorm: number;
  quadrant: Quadrant;
}

function normalizeMinMax(values: (number | null)[], invert = false): (number | null)[] {
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length === 0) return values.map(() => null);

  const vmin = Math.min(...finite);
  const vmax = Math.max(...finite);
  if (Math.abs(vmax - vmin) <= 1e-12) {
    return values.map(v => (v !== null ? 0.5 : null));
  }

  return values.map(v => {
    if (v === null) return null;
    let normalized = (v - vmin) / (vmax - vmin);
    if (invert) normalized = 1.0 - normalized;
    return Math.max(0, Math.min(1, normalized));
  });
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function classifyQuadrant(x: number, y: number, mx: number, my: number): Quadrant {
  if (x >= mx && y >= my) return 'Q1';
  if (x < mx && y >= my) return 'Q2';
  if (x < mx && y < my) return 'Q3';
  return 'Q4';
}

/** Computes SNCI_Norm/SCDI_Norm/quadrant over a set of successful job results. */
export function computeNormAndQuadrants(rows: DescriptorInput[]): Map<string, DescriptorNorm> {
  const result = new Map<string, DescriptorNorm>();
  if (rows.length === 0) return result;

  const snciNorm = normalizeMinMax(rows.map(r => r.snci), false);

  // SCDI is represented by the inverse min-max normalization of its raw
  // COSMO variance within this batch. Absolute variance scale can differ
  // substantially between batches, so raw values remain available separately.
  const scdiNorm = normalizeMinMax(rows.map(r => r.scdiVariance), true);

  const validIdx = rows
    .map((_, i) => i)
    .filter(i => snciNorm[i] !== null && scdiNorm[i] !== null);
  if (validIdx.length === 0) return result;

  const medianX = median(validIdx.map(i => snciNorm[i] as number));
  const medianY = median(validIdx.map(i => scdiNorm[i] as number));

  for (const i of validIdx) {
    const x = snciNorm[i] as number;
    const y = scdiNorm[i] as number;
    result.set(rows[i].id, {
      id: rows[i].id,
      snciNorm: x,
      scdiNorm: y,
      quadrant: classifyQuadrant(x, y, medianX, medianY),
    });
  }
  return result;
}

function fileNameFromInputPath(inputPath: string | null, fallback: string): string {
  if (!inputPath) return fallback;
  const parts = inputPath.split(/[\\/]/);
  return parts[parts.length - 1] || fallback;
}

/**
 * Builds ResultRecord[] from a set of job summaries, normalizing/classifying
 * quadrants over exactly this set (so a "run" and the global Results library
 * each get their own consistent normalization, matching how the CLI's batch
 * mode normalizes per-batch). KUID/KUID_Cluster have no API equivalent — they
 * are surfaced as 'N/A' rather than fabricated.
 */
export function buildResultRecords(
  jobs: JobSummary[],
  resolveRunId: (jobId: string) => string,
): ResultRecord[] {
  const norms = new Map<string, DescriptorNorm>();
  const successfulByBatch = new Map<string, JobSummary[]>();
  jobs
    .filter(j => mapJobStatus(j.status) === 'success' && j.knf_json)
    .forEach(job => {
      const batchId = job.options?.batch_id || resolveRunId(job.job_id);
      successfulByBatch.set(batchId, [...(successfulByBatch.get(batchId) || []), job]);
    });

  successfulByBatch.forEach(batchJobs => {
    const batchNorms = computeNormAndQuadrants(
      batchJobs.map(j => ({
        id: j.job_id,
        snci: j.knf_json?.SNCI ?? null,
        scdi: j.knf_json?.SCDI ?? null,
        scdiVariance: j.knf_json?.SCDI_variance ?? null,
      })),
    );
    batchNorms.forEach((value, key) => norms.set(key, value));
  });

  return jobs.map(job => {
    const vector = job.knf_json?.KNF_vector ?? [];
    const norm = norms.get(job.job_id);
    return {
      id: job.job_id,
      runId: resolveRunId(job.job_id),
      fileName: fileNameFromInputPath(job.input_path, job.job_id),
      f1: vector[0] ?? null,
      f2: vector[1] ?? null,
      f3: vector[2] ?? null,
      f4: vector[3] ?? null,
      f5: vector[4] ?? null,
      f6: vector[5] ?? null,
      f7: vector[6] ?? null,
      f8: vector[7] ?? null,
      f9: vector[8] ?? null,
      f2_defined: vector[1] != null,
      KUID_raw: 'N/A',
      KUID: 'N/A',
      KUID_Cluster: '',
      KUID_Intensive_raw: 'N/A',
      KUID_Intensive: 'N/A',
      KUID_Intensive_Cluster: '',
      KUID_prefix2: '',
      KUID_prefix4: '',
      KUID_prefix6: '',
      SNCI: job.knf_json?.SNCI ?? null,
      SCDI: job.knf_json?.SCDI ?? null,
      SCDI_variance: job.knf_json?.SCDI_variance ?? null,
      SNCI_Norm: norm?.snciNorm ?? null,
      SCDI_Norm: norm?.scdiNorm ?? null,
      quadrant: norm?.quadrant ?? 'Q4',
      status: mapJobStatus(job.status),
    };
  });
}
