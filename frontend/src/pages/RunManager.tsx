import { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, X, FileText, ChevronDown, ChevronUp, Play, Loader2, Plus, Search, Filter, Clock, CheckCircle2, XCircle, Square, StopCircle, ChevronRight, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { MolecularFile, Run, RunConfig } from '@/types';
import { useToast } from '@/hooks/use-toast';
import { submitUploadJob } from '@/lib/api';
import { createRunId, createRunRecord, deleteRunRecord, hydrateAllRuns, toKnfRunOptions } from '@/lib/runs';

const ACCEPTED_EXTENSIONS = ['.xyz', '.sdf', '.mol', '.pdb', '.mol2'];

const extensionBadges: Record<string, string> = {
  '.xyz': 'bg-amber-500/10 text-amber-400 border border-amber-500/20',
  '.sdf': 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
  '.mol': 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20',
  '.pdb': 'bg-purple-500/10 text-purple-400 border border-purple-500/20',
  '.mol2': 'bg-pink-500/10 text-pink-400 border border-pink-500/20',
};

const statusConfig: Record<string, { icon: React.ReactNode; color: string; bg: string; label: string }> = {
  processing: { icon: <Loader2 className="w-3 h-3 animate-spin" />, color: 'text-primary', bg: 'bg-primary/10 border-primary/20', label: 'Processing' },
  queued: { icon: <Clock className="w-3 h-3" />, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/20', label: 'Queued' },
  completed: { icon: <CheckCircle2 className="w-3 h-3" />, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20', label: 'Completed' },
  failed: { icon: <XCircle className="w-3 h-3" />, color: 'text-rose-400', bg: 'bg-rose-500/10 border-rose-500/20', label: 'Failed' },
  stopped: { icon: <Square className="w-3 h-3" />, color: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/20', label: 'Stopped' },
  stop_requested: { icon: <StopCircle className="w-3 h-3" />, color: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/20', label: 'Stopping' },
};

const defaultRunConfig: RunConfig = {
  charge: 0, spin: 1, processingMode: 'auto',
  forceRecomputation: false, cleanOutputs: true, debugMode: false,
  outputDirectory: './output/', nciBackend: 'torch', gpuEnabled: false,
  enableStopKey: true, interactiveQuadrant: true,
};

const STORAGE_KEY = 'knf-run-config';

function loadInitialConfig(): RunConfig {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return defaultRunConfig;
    return { ...defaultRunConfig, ...JSON.parse(saved) };
  } catch {
    return defaultRunConfig;
  }
}

function formatElapsed(ms: number): string {
  if (!ms || ms <= 0) return '-';
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function formatDate(iso: string): string {
  if (!iso) return '-';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso;
  }
}

const RunManager = () => {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const { toast } = useToast();
  const navigate = useNavigate();

  // ── Create dialog state ──
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [files, setFiles] = useState<MolecularFile[]>([]);
  const [pendingFiles, setPendingFiles] = useState<Map<string, File>>(new Map());
  const [isStarting, setIsStarting] = useState(false);
  const [submitProgress, setSubmitProgress] = useState<{ done: number; total: number } | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [config, setConfig] = useState<RunConfig>(loadInitialConfig);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [runName, setRunName] = useState('');

  const fetchRuns = useCallback(async () => {
    try {
      setRuns(await hydrateAllRuns());
    } catch {
      setRuns([]);
    }
  }, []);

  useEffect(() => {
    const mounted = true;
    (async () => {
      await fetchRuns();
      if (mounted) setLoading(false);
    })();
  }, [fetchRuns]);

  useEffect(() => {
    const hasActive = runs.some(r => r.status === 'processing' || r.status === 'queued');
    if (!hasActive) return;
    const id = setInterval(fetchRuns, 3000);
    return () => clearInterval(id);
  }, [runs, fetchRuns]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  }, [config]);

  // knf_core/api.py has no job-cancellation support — a submitted job runs to
  // completion. Rather than pretend a Stop button works, it's disabled below.
  const notifyStopUnsupported = useCallback(() => {
    toast({
      title: 'Cancellation not supported',
      description: 'This backend runs each job to completion — there is no way to stop it mid-run.',
      variant: 'destructive',
    });
  }, [toast]);

  const addFiles = useCallback((incoming: File[]) => {
    const newFiles: MolecularFile[] = incoming.map((f, i) => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase();
      const valid = ACCEPTED_EXTENSIONS.includes(ext);
      const duplicate = files.some(ef => ef.name === f.name);
      return {
        id: `file-${Date.now()}-${i}`,
        name: f.name,
        extension: ext,
        size: f.size,
        valid: valid && !duplicate,
        error: !valid ? `Unsupported: ${ext}` : duplicate ? 'Duplicate' : undefined,
      };
    });
    setFiles(prev => [...prev, ...newFiles]);
    setPendingFiles(prev => {
      const next = new Map(prev);
      incoming.forEach((f, i) => {
        if (newFiles[i].valid) next.set(newFiles[i].id, f);
      });
      return next;
    });
  }, [files]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);
    addFiles(Array.from(e.dataTransfer.files));
  }, [addFiles]);

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
    setPendingFiles(prev => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  };

  const handleStart = async () => {
    const validFiles = files.filter(f => f.valid);
    if (validFiles.length === 0) {
      toast({
        title: 'No Valid Files',
        description: files.length === 0
          ? 'Please select or drag at least one molecular file first.'
          : 'None of the uploaded files have a supported format (.xyz, .sdf, .mol, .pdb, .mol2).',
        variant: 'destructive',
      });
      return;
    }
    setIsStarting(true);
    setSubmitProgress({ done: 0, total: validFiles.length });

    const batchId = createRunId();
    const options = { ...toKnfRunOptions(config), batch_id: batchId };
    const jobIds: string[] = [];
    let failures = 0;

    for (const meta of validFiles) {
      const file = pendingFiles.get(meta.id);
      if (!file) continue;
      try {
        const job = await submitUploadJob(file, options);
        jobIds.push(job.job_id);
      } catch (err: unknown) {
        failures++;
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error('File upload submission failed:', meta.name, errMsg);
        toast({ title: `Upload Failed: ${meta.name}`, description: errMsg, variant: 'destructive' });
      }
      setSubmitProgress(prev => (prev ? { ...prev, done: prev.done + 1 } : prev));
    }

    setIsStarting(false);
    setSubmitProgress(null);

    if (jobIds.length === 0) {
      return;
    }

    const name = runName.trim() || `Run ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
    const record = createRunRecord(name, jobIds, config, batchId);
    toast({
      title: 'Run Started',
      description: failures > 0
        ? `${jobIds.length} file(s) submitted, ${failures} failed to submit.`
        : `${jobIds.length} file(s) submitted for processing.`,
    });
    setShowCreateDialog(false);
    setFiles([]);
    setPendingFiles(new Map());
    fetchRuns();
    navigate(`/runs/${record.id}`);
  };

  const filteredRuns = runs.filter(r =>
    !searchQuery || r.name.toLowerCase().includes(searchQuery.toLowerCase()) || r.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const validCount = files.filter(f => f.valid).length;

  return (
    <div className="p-6 lg:p-8 space-y-0 max-w-7xl animate-fade-in-up">

      {/* ── Header ── */}
      <header className="flex items-center justify-between gap-4 pb-6 border-b border-glass">
        <div>
          <div className="flex items-center gap-2 text-primary font-mono tracking-widest text-[10px] uppercase mb-1">
            <Play className="w-3.5 h-3.5" />
            Computation Runs
          </div>
          <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">
            Run <span className="text-gradient">Manager</span>
          </h1>
        </div>
        <button
              onClick={() => { setShowCreateDialog(true); setFiles([]); setPendingFiles(new Map()); setRunName(''); }}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 transition-all shadow"
        >
          <Plus className="w-4 h-4" /> Create
        </button>
      </header>

      {/* ── Search bar ── */}
      <div className="flex items-center gap-3 py-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search across your runs"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg bg-white/5 border border-glass text-sm text-foreground placeholder:text-muted-foreground/50 outline-none focus:border-primary/50 transition-colors"
          />
        </div>
        <button className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-glass bg-white/5 hover:bg-white/10 text-xs font-medium text-muted-foreground transition-colors">
          <Filter className="w-3.5 h-3.5" /> Filter
        </button>
      </div>

      {/* ── Runs table ── */}
      <div className="rounded-lg border border-glass glass-panel overflow-hidden">
        {/* Table header */}
        <div className="grid grid-cols-12 gap-4 px-6 py-3 border-b border-glass text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
          <div className="col-span-1">Status</div>
          <div className="col-span-4">Run</div>
          <div className="col-span-2">Files</div>
          <div className="col-span-2">Date</div>
          <div className="col-span-2">Duration</div>
          <div className="col-span-1 text-right">Actions</div>
        </div>

        {loading ? (
          <div className="p-6 space-y-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-14 rounded-lg animate-shimmer" style={{ animationDelay: `${i * 0.1}s` }} />
            ))}
          </div>
        ) : filteredRuns.length === 0 ? (
          <div className="p-16 text-center">
            <div className="w-16 h-16 rounded-full bg-white/5 border border-glass flex items-center justify-center mx-auto mb-4">
              <Play className="w-7 h-7 text-muted-foreground/40" />
            </div>
            <p className="text-sm font-medium text-foreground mb-1">No computation runs yet</p>
            <p className="text-xs text-muted-foreground mb-4">Click <strong>Create</strong> to upload molecular files and start a new run.</p>
            <button
          onClick={() => { setShowCreateDialog(true); setFiles([]); setPendingFiles(new Map()); setRunName(''); }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 text-primary text-sm font-medium border border-primary/20 hover:bg-primary/20 transition-colors"
            >
              <Plus className="w-4 h-4" /> Create your first run
            </button>
          </div>
        ) : (
          <div className="divide-y divide-glass/40">
            {filteredRuns.map((run) => {
              const sc = statusConfig[run.status] || statusConfig.failed;
              const progressPct = run.totalFiles > 0 ? Math.round((run.completedFiles / run.totalFiles) * 100) : 0;
              const isActive = run.status === 'processing' || run.status === 'queued';

              return (
                <div
                  key={run.id}
                  className="grid grid-cols-12 gap-4 items-center px-6 py-3.5 hover:bg-white/[0.02] transition-colors group cursor-pointer"
                  onClick={() => navigate(`/runs/${run.id}`)}
                >
                  {/* Status */}
                  <div className="col-span-1">
                    <span className={cn('inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-semibold border', sc.bg, sc.color)}>
                      {sc.icon} {sc.label}
                    </span>
                  </div>

                  {/* Run name + ID */}
                  <div className="col-span-4 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{run.name}</p>
                    <p className="text-[10px] text-muted-foreground font-mono mt-0.5 truncate">{run.id}</p>
                  </div>

                  {/* Files + progress */}
                  <div className="col-span-2">
                    <p className="text-xs font-medium text-foreground">{run.totalFiles} file{run.totalFiles !== 1 ? 's' : ''}</p>
                    {isActive && run.totalFiles > 0 && (
                      <div className="flex items-center gap-2 mt-1">
                        <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                          <div className="h-full bg-primary rounded-full transition-all duration-500" style={{ width: `${progressPct}%` }} />
                        </div>
                        <span className="text-[9px] font-mono text-muted-foreground">{progressPct}%</span>
                      </div>
                    )}
                  </div>

                  {/* Date */}
                  <div className="col-span-2">
                    <p className="text-xs text-muted-foreground">{formatDate(run.createdAt)}</p>
                  </div>

                  {/* Duration */}
                  <div className="col-span-2">
                    <p className="text-xs text-muted-foreground font-mono">{formatElapsed(run.elapsedMs)}</p>
                  </div>

                  {/* Actions */}
                  <div className="col-span-1 flex justify-end items-center gap-2" onClick={e => e.stopPropagation()}>
                    <button
                      onClick={() => {
                        deleteRunRecord(run.id);
                        fetchRuns();
                        toast({ title: 'Run Removed', description: 'Run record deleted from workspace history.' });
                      }}
                      title="Delete run record"
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-white/10 text-muted-foreground hover:text-rose-400 transition-all"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    <ChevronRight className="w-4 h-4 text-muted-foreground/30 group-hover:text-muted-foreground transition-colors" />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ════════════════════════════════════════════════════════════════════════
          CREATE DIALOG — YouTube-style upload modal
          ════════════════════════════════════════════════════════════════════════ */}
      {showCreateDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => { setShowCreateDialog(false); setFiles([]); setPendingFiles(new Map()); }} />

          {/* Dialog */}
          <div className="relative bg-zinc-950 border border-zinc-800 rounded-xl w-full max-w-2xl max-h-[90vh] overflow-hidden shadow-2xl animate-fade-in-up">
            {/* Dialog header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
              <h2 className="text-lg font-semibold text-foreground">New Computation Run</h2>
              <button onClick={() => { setShowCreateDialog(false); setFiles([]); setPendingFiles(new Map()); }} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                <X className="w-5 h-5 text-muted-foreground" />
              </button>
            </div>

            {/* Scrollable body */}
            <div className="overflow-y-auto max-h-[calc(90vh-140px)] p-6 space-y-6">

              {/* ── Upload Zone ── */}
              <div
                onDrop={handleDrop}
                onDragOver={e => { e.preventDefault(); setIsDraggingOver(true); }}
                onDragLeave={() => setIsDraggingOver(false)}
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  'relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all',
                  isDraggingOver
                    ? 'border-primary bg-primary/5'
                    : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/30 hover:bg-zinc-900/50'
                )}
              >
                <div className={cn(
                  'w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-all',
                  isDraggingOver ? 'bg-primary/10' : 'bg-white/5'
                )}>
                  <Upload className={cn('w-8 h-8 transition-colors', isDraggingOver ? 'text-primary' : 'text-muted-foreground/50')} />
                </div>
                <p className="text-sm font-semibold text-foreground mb-1">Drag and drop molecular files to upload</p>
                <p className="text-xs text-muted-foreground mb-3">or click to browse from your computer</p>
                <div className="flex flex-wrap items-center justify-center gap-1.5">
                  {ACCEPTED_EXTENSIONS.map(ext => (
                    <span key={ext} className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 border border-zinc-800 text-muted-foreground">
                      {ext}
                    </span>
                  ))}
                </div>
                <input ref={fileInputRef} type="file" multiple accept={ACCEPTED_EXTENSIONS.join(',')} className="hidden" onChange={e => e.target.files && addFiles(Array.from(e.target.files))} />
              </div>

              {/* ── Uploaded file list ── */}
              {files.length > 0 && (
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 divide-y divide-zinc-800/50 max-h-[200px] overflow-y-auto">
                  <div className="px-4 py-2.5 flex items-center justify-between border-b border-zinc-800">
                    <span className="text-xs font-semibold text-foreground">{files.length} file{files.length !== 1 ? 's' : ''}</span>
                    <span className="text-[10px] font-mono text-muted-foreground">{validCount} valid</span>
                  </div>
                  {files.map(f => (
                    <div key={f.id} className={cn('flex items-center gap-3 px-4 py-2 text-xs', !f.valid && 'opacity-50')}>
                      <FileText className={cn('w-3.5 h-3.5 shrink-0', f.valid ? 'text-primary' : 'text-destructive')} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <p className="text-foreground font-medium truncate">{f.name}</p>
                          {f.valid && (
                            <span className={cn('text-[9px] font-mono font-semibold px-1 rounded shrink-0', extensionBadges[f.extension] || 'bg-white/5 text-muted-foreground')}>
                              {f.extension.slice(1).toUpperCase()}
                            </span>
                          )}
                        </div>
                        {f.error ? (
                          <p className="text-[9px] text-destructive mt-0.5">{f.error}</p>
                        ) : (
                          <p className="text-[9px] text-muted-foreground font-mono mt-0.5">{(f.size / 1024).toFixed(1)} KB</p>
                        )}
                      </div>
                      <button onClick={() => removeFile(f.id)} className="text-muted-foreground hover:text-destructive transition-colors">
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* ── Run Name ── */}
              <div>
                <label className="block text-[10px] font-semibold text-muted-foreground mb-1.5 uppercase tracking-wider">Run Name</label>
                <input
                  type="text"
                  value={runName}
                  onChange={e => setRunName(e.target.value)}
                  placeholder={`Run ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`}
                  className="w-full rounded-lg bg-input/40 border border-glass px-3 py-2 text-sm text-foreground font-medium placeholder:text-muted-foreground/40 outline-none focus:border-primary/50 transition-colors"
                />
                <p className="text-[9px] text-muted-foreground mt-1">Optional — leave empty for auto-generated name</p>
              </div>

              {/* ── Config section ── */}
              <div className="space-y-4">
                <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">Engine Parameters</h3>
                <div className="grid grid-cols-2 gap-4">
                  <ConfigField label="Charge">
                    <input type="number" value={config.charge} onChange={e => setConfig({...config, charge: parseInt(e.target.value) || 0})} className="w-full rounded-lg bg-input/40 border border-glass px-3 py-1.5 text-xs text-foreground font-mono outline-none" />
                  </ConfigField>
                  <ConfigField label="Spin">
                    <input type="number" value={config.spin} onChange={e => setConfig({...config, spin: parseInt(e.target.value) || 1})} className="w-full rounded-lg bg-input/40 border border-glass px-3 py-1.5 text-xs text-foreground font-mono outline-none" />
                  </ConfigField>
                  <ConfigField label="Processing Mode">
                    <select value={config.processingMode} onChange={e => setConfig({...config, processingMode: e.target.value as any})} className="w-full rounded-lg bg-input/40 border border-glass px-3 py-1.5 text-xs text-foreground outline-none">
                      <option value="auto">Auto</option>
                      <option value="single">Single</option>
                      <option value="multi">Parallel</option>
                    </select>
                  </ConfigField>
                  <ConfigField label="NCI Backend">
                    <select value={config.nciBackend} onChange={e => setConfig({...config, nciBackend: e.target.value as any})} className="w-full rounded-lg bg-input/40 border border-glass px-3 py-1.5 text-xs text-foreground outline-none">
                      <option value="torch">PyTorch</option>
                      <option value="multiwfn">Multiwfn</option>
                    </select>
                  </ConfigField>
                </div>

                <ConfigField label="Output Directory">
                  <div className="flex gap-2">
                    <input type="text" value={config.outputDirectory} onChange={e => setConfig({...config, outputDirectory: e.target.value})} className="w-full rounded-lg bg-input/40 border border-glass px-3 py-1.5 text-xs text-foreground font-mono outline-none" />
                    <button type="button" onClick={async () => {
                      if (!window.electronAPI?.selectOutputDirectory) return;
                      const chosen = await window.electronAPI.selectOutputDirectory();
                      if (chosen) setConfig({...config, outputDirectory: chosen});
                    }} className="shrink-0 px-3 py-1.5 rounded-lg border border-glass bg-white/5 hover:bg-white/10 text-xs font-medium transition-colors">Browse</button>
                  </div>
                </ConfigField>

                {/* Toggles */}
                <div className="grid grid-cols-2 gap-2 pt-2">
                  {([
                    ['gpuEnabled', 'GPU Acceleration'],
                    ['forceRecomputation', 'Force Recompute'],
                    ['cleanOutputs', 'Clean Outputs'],
                    ['debugMode', 'Debug Mode'],
                    ['enableStopKey', 'Enable Stop Key'],
                    ['interactiveQuadrant', 'Interactive Quadrant'],
                  ] as const).map(([key, label]) => (
                    <label key={key} className="flex items-center gap-2 text-xs cursor-pointer py-1">
                      <input type="checkbox" checked={config[key] as boolean} onChange={e => setConfig({...config, [key]: e.target.checked})} className="rounded border-glass bg-background/50 focus:ring-primary text-primary" />
                      <span className="text-muted-foreground">{label}</span>
                    </label>
                  ))}
                </div>

                {/* Advanced toggle */}
                <button type="button" onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground hover:text-foreground transition-colors pt-2 border-t border-zinc-800">
                  {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  ADVANCED SETTINGS
                </button>
                {showAdvanced && (
                  <div className="grid grid-cols-2 gap-3">
                    {(['gridSpacing', 'gridPadding', 'batchSize', 'eigBatchSize', 'rhoFloor'] as const).map(key => (
                      <ConfigField key={key} label={key.replace(/([A-Z])/g, ' $1').trim()}>
                        <input type="number" step="any" value={(config as any)[key] || ''} onChange={e => setConfig({...config, [key]: e.target.value ? parseFloat(e.target.value) : null})} placeholder="Auto" className="w-full rounded-lg bg-input/40 border border-glass px-2.5 py-1.5 text-xs text-foreground font-mono outline-none" />
                      </ConfigField>
                    ))}
                    <ConfigField label="NCI Device">
                      <input type="text" value={config.nciDevice || ''} onChange={e => setConfig({...config, nciDevice: e.target.value || null})} placeholder="Auto" className="w-full rounded-lg bg-input/40 border border-glass px-2.5 py-1.5 text-xs text-foreground font-mono outline-none" />
                    </ConfigField>
                  </div>
                )}
              </div>

              {/* ── Submission progress (when submitting jobs) ── */}
              {isStarting && submitProgress && (
                <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
                  <div className="flex items-center justify-between border-b border-zinc-800/60 pb-2 mb-2">
                    <span className="text-[10px] font-mono text-muted-foreground uppercase">Submitting Jobs</span>
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500 animate-pulse" />
                  </div>
                  <p className="text-xs font-mono text-muted-foreground">
                    {submitProgress.done} / {submitProgress.total} files submitted to the backend...
                  </p>
                  <p className="text-[9px] text-muted-foreground/70 mt-1">
                    Each file becomes its own job. Progress after this is tracked in the Run Manager table.
                  </p>
                </div>
              )}
            </div>

            {/* Dialog footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-zinc-800 bg-zinc-950/80">
              <span className="text-[10px] font-mono text-muted-foreground">
                {validCount} file{validCount !== 1 ? 's' : ''} ready
              </span>
              <div className="flex items-center gap-3">
                <button onClick={() => { setShowCreateDialog(false); setFiles([]); setPendingFiles(new Map()); }} className="px-4 py-2 rounded-lg border border-zinc-800 bg-white/5 hover:bg-white/10 text-xs font-medium transition-colors">
                  Cancel
                </button>
                <button
                  onClick={handleStart}
                  disabled={validCount === 0 || isStarting}
                  className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
                >
                  {isStarting ? (
                    <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Submitting...</>
                  ) : (
                    <><Play className="w-3.5 h-3.5" /> Run ({validCount} files)</>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

function ConfigField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] font-semibold text-muted-foreground mb-1.5 uppercase tracking-wider">{label}</label>
      {children}
    </div>
  );
}

export default RunManager;
