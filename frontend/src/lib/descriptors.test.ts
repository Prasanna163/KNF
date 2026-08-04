import { describe, it, expect } from 'vitest';
import { computeNormAndQuadrants, buildResultRecords } from './descriptors';
import type { JobSummary } from './api';

function makeJob(id: string, snci: number, scdiVariance: number, overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    job_id: id,
    status: 'succeeded',
    kind: 'upload',
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    finished_at: '2026-01-01T00:00:05Z',
    elapsed_seconds: 5,
    input_path: `C:\\tmp\\${id}.mol`,
    output_root: 'C:\\tmp\\Results',
    result_dir: 'C:\\tmp\\Results\\x',
    managed_workspace: true,
    error: null,
    options: {},
    artifacts: [],
    knf_json_path: 'C:\\tmp\\knf.json',
    knf_json: {
      SNCI: snci,
      SCDI_variance: scdiVariance,
      KNF_vector: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    output_txt_path: 'C:\\tmp\\output.txt',
    ...overrides,
  };
}

describe('computeNormAndQuadrants', () => {
  it('matches the reference min-max + median-split algorithm from quadrants.py', () => {
    // Three rows spanning a range, no SCDI present -> falls back to inverted SCDI_variance normalization.
    const rows = [
      { id: 'a', snci: 42.588, scdi: null, scdiVariance: 0.999997 },
      { id: 'b', snci: 42.604, scdi: null, scdiVariance: 0.9999975 },
      { id: 'c', snci: 42.5955, scdi: null, scdiVariance: 0.9999975 },
    ];
    const norms = computeNormAndQuadrants(rows);
    expect(norms.size).toBe(3);

    // SNCI min-max: a=0 (min), b=1 (max), c in between.
    expect(norms.get('a')!.snciNorm).toBeCloseTo(0, 5);
    expect(norms.get('b')!.snciNorm).toBeCloseTo(1, 5);

    // Every quadrant assigned must be one of Q1-Q4 and consistent with median split.
    for (const v of norms.values()) {
      expect(['Q1', 'Q2', 'Q3', 'Q4']).toContain(v.quadrant);
      expect(v.snciNorm).toBeGreaterThanOrEqual(0);
      expect(v.snciNorm).toBeLessThanOrEqual(1);
    }
  });

  it('collapses to 0.5 when every value is identical (matches the Python abs(vmax-vmin)<=1e-12 branch)', () => {
    const rows = [
      { id: 'a', snci: 10, scdi: 0.5, scdiVariance: 1 },
      { id: 'b', snci: 10, scdi: 0.5, scdiVariance: 1 },
    ];
    const norms = computeNormAndQuadrants(rows);
    expect(norms.get('a')!.snciNorm).toBeCloseTo(0.5, 5);
    expect(norms.get('b')!.snciNorm).toBeCloseTo(0.5, 5);
  });

  it('always normalizes raw SCDI variance within the batch', () => {
    const rows = [
      { id: 'a', snci: 1, scdi: 0.2, scdiVariance: 0.1 },
      { id: 'b', snci: 2, scdi: 1.4, scdiVariance: 0.9 }, // out-of-range SCDI gets clamped to 1
    ];
    const norms = computeNormAndQuadrants(rows);
    expect(norms.get('a')!.scdiNorm).toBeCloseTo(1, 5);
    expect(norms.get('b')!.scdiNorm).toBeCloseTo(0, 5);
  });

  it('returns an empty map when nothing is successful', () => {
    expect(computeNormAndQuadrants([]).size).toBe(0);
  });
});

describe('buildResultRecords', () => {
  it('builds ResultRecord[] from job summaries, only normalizing across successful jobs', () => {
    const jobs = [
      makeJob('job-1', 42.588, 0.999997),
      makeJob('job-2', 42.604, 0.9999975),
      makeJob('job-3', 0, 0, { status: 'failed', knf_json: null }),
    ];
    const records = buildResultRecords(jobs, () => 'run-x');
    expect(records).toHaveLength(3);

    const success = records.filter(r => r.status === 'success');
    expect(success).toHaveLength(2);
    expect(success.every(r => r.f1 === 1 && r.f9 === 9)).toBe(true);
    expect(success.every(r => r.KUID === 'N/A')).toBe(true);

    const failed = records.find(r => r.status === 'failed')!;
    expect(failed.SNCI).toBeNull();
    expect(failed.fileName).toBe('job-3.mol');
  });

  it('normalizes each run independently instead of mixing global results', () => {
    const jobs = [
      makeJob('a-1', 1, 2),
      makeJob('a-2', 2, 6),
      makeJob('b-1', 100, 200),
      makeJob('b-2', 200, 600),
    ];
    const records = buildResultRecords(jobs, id => id.startsWith('a-') ? 'run-a' : 'run-b');
    expect(records.find(r => r.id === 'a-1')!.SCDI_Norm).toBeCloseTo(1, 6);
    expect(records.find(r => r.id === 'a-2')!.SCDI_Norm).toBeCloseTo(0, 6);
    expect(records.find(r => r.id === 'b-1')!.SCDI_Norm).toBeCloseTo(1, 6);
    expect(records.find(r => r.id === 'b-2')!.SCDI_Norm).toBeCloseTo(0, 6);
  });
});
