import { Run, Job, ResultRecord, QuadrantData, Quadrant, JobStatus } from '@/types';

// ─── Seed RNG for reproducibility ───
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

const rng = seededRandom(42);
const rand = (min: number, max: number) => min + rng() * (max - min);
const randInt = (min: number, max: number) => Math.floor(rand(min, max + 1));

// ─── Generate 40 molecular files ───
const extensions = ['.xyz', '.sdf', '.mol', '.pdb', '.mol2'];
const moleculeNames = [
  'benzene', 'toluene', 'naphthalene', 'ethanol', 'methane', 'propane',
  'acetone', 'formaldehyde', 'glycine', 'alanine', 'caffeine', 'aspirin',
  'ibuprofen', 'dopamine', 'serotonin', 'adenine', 'thymine', 'cytosine',
  'guanine', 'uracil', 'glucose', 'fructose', 'sucrose', 'cholesterol',
  'retinol', 'ribose', 'lysine', 'proline', 'valine', 'leucine',
  'histidine', 'arginine', 'tryptophan', 'tyrosine', 'cysteine', 'serine',
  'threonine', 'glutamine', 'aspartate', 'phenylalanine',
];

const statuses: JobStatus[] = ['success', 'success', 'success', 'success', 'success',
  'success', 'success', 'failed', 'stopped', 'success'];

const kuidClusters = ['A1', 'B2', 'C3', 'D4', 'E5'];

function generateResults(runId: string): ResultRecord[] {
  return moleculeNames.map((name, i) => {
    const ext = extensions[i % extensions.length];
    const status = statuses[i % statuses.length];
    const f1 = +rand(0.5, 5.0).toFixed(6);
    const f2 = +rand(0.1, 3.0).toFixed(6);
    const f3 = +rand(-2.0, 2.0).toFixed(6);
    const f4 = +rand(0.01, 1.0).toFixed(6);
    const f5 = +rand(10, 500).toFixed(2);
    const f6 = +rand(0.1, 10.0).toFixed(6);
    const f7 = +rand(-5.0, 5.0).toFixed(6);
    const f8 = +rand(0.001, 0.5).toFixed(6);
    const f9 = +rand(1.0, 100.0).toFixed(2);
    const SNCI = +rand(-8, 8).toFixed(6);
    const SCDI = +rand(0, 18).toFixed(6);
    const SCDI_variance = +rand(0.01, 4.5).toFixed(6);
    const SNCI_Norm = +rand(0.02, 0.98).toFixed(6);
    const SCDI_Norm = +rand(0.02, 0.98).toFixed(6);
    const f2_defined = f2 > 1.0;
    const kuidBase = `KUID-${String(i + 1).padStart(4, '0')}-${name.substring(0, 4).toUpperCase()}`;
    const KUID_raw = `${kuidBase}-raw`;
    const KUID = kuidBase;
    const KUID_Cluster = kuidClusters[i % kuidClusters.length];
    const KUID_Intensive_raw = `${KUID_raw}-int`;
    const KUID_Intensive = `${KUID}-int`;
    const KUID_Intensive_Cluster = kuidClusters[(i + 2) % kuidClusters.length];
    const KUID_prefix2 = KUID.substring(0, 2);
    const KUID_prefix4 = KUID.substring(0, 4);
    const KUID_prefix6 = KUID.substring(0, 6);
    return {
      id: `res-${runId}-${i}`,
      runId,
      fileName: `${name}${ext}`,
      f1, f2, f3, f4, f5, f6, f7, f8, f9,
      f2_defined,
      KUID_raw, KUID, KUID_Cluster,
      KUID_Intensive_raw, KUID_Intensive, KUID_Intensive_Cluster,
      KUID_prefix2, KUID_prefix4, KUID_prefix6,
      SNCI, SCDI, SCDI_variance,
      SNCI_Norm, SCDI_Norm,
      quadrant: 'Q1' as Quadrant, // assigned below
      status,
    };
  });
}

function assignQuadrants(records: ResultRecord[]): { records: ResultRecord[]; quadrantData: QuadrantData } {
  const snciVals = records.map(r => r.SNCI_Norm).sort((a, b) => a - b);
  const scdiVals = records.map(r => r.SCDI_Norm).sort((a, b) => a - b);
  const median = (arr: number[]) => {
    const m = Math.floor(arr.length / 2);
    return arr.length % 2 ? arr[m] : (arr[m - 1] + arr[m]) / 2;
  };
  const medianSNCI = +median(snciVals).toFixed(6);
  const medianSCDI = +median(scdiVals).toFixed(6);
  const counts: Record<Quadrant, number> = { Q1: 0, Q2: 0, Q3: 0, Q4: 0 };
  const files: Record<Quadrant, string[]> = { Q1: [], Q2: [], Q3: [], Q4: [] };
  
  const updated = records.map(r => {
    let q: Quadrant;
    if (r.SNCI_Norm >= medianSNCI && r.SCDI_Norm >= medianSCDI) q = 'Q1';
    else if (r.SNCI_Norm < medianSNCI && r.SCDI_Norm >= medianSCDI) q = 'Q2';
    else if (r.SNCI_Norm < medianSNCI && r.SCDI_Norm < medianSCDI) q = 'Q3';
    else q = 'Q4';
    counts[q]++;
    files[q].push(r.fileName);
    return { ...r, quadrant: q };
  });
  
  return { records: updated, quadrantData: { medianSNCI, medianSCDI, counts, files } };
}

// ─── Build mock data ───
const rawResults = generateResults('run-001');
const { records: mockResults, quadrantData: mockQuadrantData } = assignQuadrants(rawResults);

export const MOCK_RESULTS: ResultRecord[] = mockResults;
export const MOCK_QUADRANT_DATA: QuadrantData = mockQuadrantData;

export const MOCK_JOBS: Job[] = mockResults.map((r, i) => ({
  id: `job-${i}`,
  runId: 'run-001',
  fileName: r.fileName,
  status: r.status,
  elapsedMs: randInt(800, 45000),
  errorMessage: r.status === 'failed' ? 'Convergence error: SCF did not converge within 200 cycles' : undefined,
}));

export const MOCK_RUNS: Run[] = [
  {
    id: 'run-001',
    name: 'Amino Acid Batch Analysis',
    status: 'completed',
    config: {
      charge: 0, spin: 1, processingMode: 'multi', workers: 4,
      forceRecomputation: false, cleanOutputs: true, debugMode: false,
      outputDirectory: './output/run-001', nciBackend: 'torch', gpuEnabled: true,
      enableStopKey: true, interactiveQuadrant: true,
    },
    files: [],
    createdAt: '2025-02-20T10:30:00Z',
    startedAt: '2025-02-20T10:31:00Z',
    completedAt: '2025-02-20T11:15:00Z',
    totalFiles: 40, completedFiles: 40, successFiles: 32, failedFiles: 4, stoppedFiles: 4,
    elapsedMs: 2640000, throughput: 0.91, cpuPercent: 45, ramPercent: 62, activeWorkers: 0,
  },
  {
    id: 'run-002',
    name: 'Drug Candidate Screen',
    status: 'processing',
    config: {
      charge: 0, spin: 1, processingMode: 'auto',
      forceRecomputation: true, cleanOutputs: false, debugMode: true,
      outputDirectory: './output/run-002', nciBackend: 'torch', gpuEnabled: true,
      enableStopKey: true, interactiveQuadrant: false,
    },
    files: [],
    createdAt: '2025-02-21T14:00:00Z',
    startedAt: '2025-02-21T14:02:00Z',
    totalFiles: 25, completedFiles: 12, successFiles: 11, failedFiles: 1, stoppedFiles: 0,
    elapsedMs: 980000, etaMs: 1200000, throughput: 0.73, cpuPercent: 78, ramPercent: 55, activeWorkers: 3,
  },
  {
    id: 'run-003',
    name: 'Solvent Interaction Study',
    status: 'failed',
    config: {
      charge: -1, spin: 2, processingMode: 'single',
      forceRecomputation: false, cleanOutputs: true, debugMode: false,
      outputDirectory: './output/run-003', nciBackend: 'multiwfn', gpuEnabled: false,
      enableStopKey: false, interactiveQuadrant: true,
    },
    files: [],
    createdAt: '2025-02-19T08:00:00Z',
    startedAt: '2025-02-19T08:01:00Z',
    completedAt: '2025-02-19T08:15:00Z',
    totalFiles: 10, completedFiles: 3, successFiles: 2, failedFiles: 8, stoppedFiles: 0,
    elapsedMs: 840000, cpuPercent: 12, ramPercent: 30, activeWorkers: 0,
  },
  {
    id: 'run-004',
    name: 'Polymer Fragment Analysis',
    status: 'queued',
    config: {
      charge: 0, spin: 1, processingMode: 'multi', workers: 8,
      forceRecomputation: false, cleanOutputs: false, debugMode: false,
      outputDirectory: './output/run-004', nciBackend: 'torch', gpuEnabled: true,
      enableStopKey: true, interactiveQuadrant: true,
    },
    files: [],
    createdAt: '2025-02-22T09:00:00Z',
    totalFiles: 15, completedFiles: 0, successFiles: 0, failedFiles: 0, stoppedFiles: 0,
    elapsedMs: 0, activeWorkers: 0,
  },
];

export const MOCK_LOG_LINES: string[] = [
  '[10:31:00] Initializing KNF pipeline v2.4.1...',
  '[10:31:01] Loading molecular files: 40 detected',
  '[10:31:01] Validation complete: 40/40 valid',
  '[10:31:02] Backend: torch | Device: cuda:0 | Workers: 4',
  '[10:31:02] Starting batch processing...',
  '[10:31:05] [1/40] benzene.xyz → computing descriptors...',
  '[10:31:08] [1/40] benzene.xyz → f1=2.3401 f2=1.2034 ... SNCI=-3.2100 SCDI=8.4521 ✓',
  '[10:31:09] [2/40] toluene.sdf → computing descriptors...',
  '[10:31:13] [2/40] toluene.sdf → f1=3.1023 f2=0.8901 ... SNCI=1.4520 SCDI=12.3400 ✓',
  '[10:31:14] [3/40] naphthalene.mol → computing descriptors...',
  '[10:31:19] [3/40] naphthalene.mol → f1=4.2100 f2=2.1034 ... SNCI=5.1200 SCDI=3.2100 ✓',
  '[10:31:20] [4/40] ethanol.pdb → computing descriptors...',
  '[10:31:22] [4/40] ethanol.pdb → FAILED: SCF convergence error after 200 cycles',
  '[10:31:23] [5/40] methane.mol2 → computing descriptors...',
  '[10:31:25] [5/40] methane.mol2 → f1=0.8901 f2=0.2100 ... SNCI=-7.1200 SCDI=1.2300 ✓',
  '[10:31:26] Throughput: 0.91 files/min | ETA: 38m remaining',
  '[10:31:26] CPU: 78% | RAM: 55% | Workers: 4/4 active',
  '[10:45:00] [20/40] glucose.xyz → computing descriptors...',
  '[10:45:05] [20/40] glucose.xyz → f1=1.5600 f2=2.8900 ... SNCI=2.3400 SCDI=9.1200 ✓',
  '[11:14:50] [40/40] phenylalanine.mol2 → computing descriptors...',
  '[11:14:58] [40/40] phenylalanine.mol2 → f1=3.8900 f2=1.7800 ... SNCI=0.4500 SCDI=14.5600 ✓',
  '[11:15:00] Batch processing complete.',
  '[11:15:00] Summary: 32 success | 4 failed | 4 stopped',
  '[11:15:01] Computing SNCI_Norm and SCDI_Norm...',
  '[11:15:01] Normalization complete. Median SNCI_Norm=0.4821, Median SCDI_Norm=0.5134',
  '[11:15:02] Quadrant analysis: Q1=12 Q2=8 Q3=10 Q4=10',
  '[11:15:02] Writing batch_knf.csv...',
  '[11:15:02] Writing batch_knf.json...',
  '[11:15:03] Writing snci_scdi_quadrants.json...',
  '[11:15:03] Pipeline complete. Total elapsed: 44m 3s',
];
