// ─── Run & Job Status ───
export type RunStatus = 'idle' | 'validating' | 'queued' | 'processing' | 'stop_requested' | 'finalizing' | 'completed' | 'failed';
export type JobStatus = 'queued' | 'running' | 'success' | 'failed' | 'stopped';
export type Quadrant = 'Q1' | 'Q2' | 'Q3' | 'Q4';

// ─── Molecular File ───
export interface MolecularFile {
  id: string;
  name: string;
  extension: string;
  size: number; // bytes
  valid: boolean;
  error?: string;
}

// ─── Run Configuration ───
export interface RunConfig {
  charge: number;
  spin: number;
  processingMode: 'auto' | 'single' | 'multi';
  workers?: number;
  forceRecomputation: boolean;
  cleanOutputs: boolean;
  debugMode: boolean;
  outputDirectory: string;
  nciBackend: 'torch' | 'multiwfn';
  gpuEnabled: boolean;
  scdiVarMin?: number;
  scdiVarMax?: number;
  enableStopKey: boolean;
  interactiveQuadrant: boolean;
  // Advanced NCI
  gridSpacing?: number;
  gridPadding?: number;
  nciDevice?: string;
  dtype?: string;
  batchSize?: number;
  eigBatchSize?: number;
  rhoFloor?: number;
}

// ─── Run ───
export interface Run {
  id: string;
  name: string;
  status: RunStatus;
  config: RunConfig;
  files: MolecularFile[];
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  totalFiles: number;
  completedFiles: number;
  successFiles: number;
  failedFiles: number;
  stoppedFiles: number;
  elapsedMs: number;
  etaMs?: number;
  throughput?: number; // files/min
  cpuPercent?: number;
  ramPercent?: number;
  activeWorkers?: number;
}

// ─── Job ───
export interface Job {
  id: string;
  runId: string;
  fileName: string;
  status: JobStatus;
  elapsedMs: number;
  errorMessage?: string;
}

// ─── Result Record ───
export interface ResultRecord {
  id: string;
  runId: string;
  fileName: string;
  f1: number | null;
  f2: number | null;
  f3: number | null;
  f4: number | null;
  f5: number | null;
  f6: number | null;
  f7: number | null;
  f8: number | null;
  f9: number | null;
  f2_defined: boolean;
  KUID_raw: string;
  KUID: string;
  KUID_Cluster: string;
  KUID_Intensive_raw: string;
  KUID_Intensive: string;
  KUID_Intensive_Cluster: string;
  KUID_prefix2: string;
  KUID_prefix4: string;
  KUID_prefix6: string;
  SNCI: number | null;
  SCDI: number | null;
  SCDI_variance: number | null;
  SNCI_Norm: number | null;
  SCDI_Norm: number | null;
  quadrant: Quadrant;
  status: JobStatus;
}

// ─── Quadrant Data ───
export interface QuadrantData {
  medianSNCI: number;
  medianSCDI: number;
  counts: Record<Quadrant, number>;
  files: Record<Quadrant, string[]>;
}

// ─── Settings ───
export interface AppSettings {
  apiBaseUrl: string;
  defaultProcessingMode: 'auto' | 'single' | 'multi';
  defaultBackend: 'torch' | 'multiwfn';
  defaultWorkers: number;
  notificationsEnabled: boolean;
}
