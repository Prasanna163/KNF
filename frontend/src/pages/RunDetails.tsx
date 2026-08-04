import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Copy, RefreshCw, Square, Terminal, CheckCircle2, XCircle, FileText, Check, Table, List, Search, ZoomIn, ZoomOut, RotateCcw, CircleDot, Sparkles, BarChart2 } from 'lucide-react';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { EmptyState } from '@/components/shared/EmptyState';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import type { Run, ResultRecord, Quadrant } from '@/types';
import { getJob, getArtifactText, mapJobStatus, type JobSummary } from '@/lib/api';
import { getRunRecord, hydrateRun } from '@/lib/runs';
import { buildResultRecords } from '@/lib/descriptors';
import {
  CartesianGrid,
  Label,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const RunDetails = () => {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<Run | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<ResultRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<'logs' | 'results' | 'explorer'>('logs');
  const { toast } = useToast();

  const fetchData = useCallback(async () => {
    if (!runId) return;
    const record = getRunRecord(runId);
    if (!record) {
      setRun(null);
      setLogs([]);
      setResults([]);
      return;
    }
    try {
      const [runData, jobs] = await Promise.all([
        hydrateRun(record),
        Promise.all(record.jobIds.map(id => getJob(id).catch(() => null))),
      ]);
      const liveJobs = jobs.filter((j): j is JobSummary => j !== null);
      setRun(runData);
      setResults(buildResultRecords(liveJobs, () => record.id));

      // No log-streaming endpoint on this backend — fall back to each
      // succeeded job's output.txt artifact as a static log substitute.
      const logChunks = await Promise.all(
        liveJobs.map(async j => {
          if (j.error) {
            return `[ERROR Job ${j.job_id}]: ${j.error}`;
          }
          if (mapJobStatus(j.status) === 'success') {
            return getArtifactText(j.job_id, (j.output_txt_path || 'output.txt').split(/[\\/]/).pop() || 'output.txt').catch(() => `[Job ${j.job_id} Succeeded]`);
          }
          return `[Job ${j.job_id} Status: ${j.status}]`;
        }),
      );
      setLogs(logChunks.filter(Boolean).flatMap(c => c.split('\n')));
    } catch {
      setRun(null);
      setLogs([]);
      setResults([]);
    }
  }, [runId]);

  useEffect(() => {
    const mounted = true;
    (async () => {
      await fetchData();
      if (mounted) setLoading(false);
    })();
  }, [fetchData]);

  // Poll every 3s while the run is still processing
  useEffect(() => {
    if (!run || (run.status !== 'processing' && run.status !== 'queued')) return;
    const id = setInterval(fetchData, 3000);
    return () => clearInterval(id);
  }, [run, fetchData]);

  // ── Scatter Explorer (per-run) ──────────────────────────────────────
  type Domain = [number, number];
  const fullDomain: Domain = [0, 1];

  const quadrantColors: Record<Quadrant, string> = {
    Q1: '#0ea5e9', // Sky Blue
    Q2: '#6366f1', // Indigo
    Q3: '#f59e0b', // Amber
    Q4: '#10b981', // Emerald
  };

  const quadrantBgs: Record<Quadrant, string> = {
    Q1: 'bg-[#0ea5e9]/10 text-[#0ea5e9] border-[#0ea5e9]/20',
    Q2: 'bg-[#6366f1]/10 text-[#6366f1] border-[#6366f1]/20',
    Q3: 'bg-[#f59e0b]/10 text-[#f59e0b] border-[#f59e0b]/20',
    Q4: 'bg-[#10b981]/10 text-[#10b981] border-[#10b981]/20',
  };

  const descriptorKeys: ('f1' | 'f2' | 'f3' | 'f4' | 'f5' | 'f6' | 'f7' | 'f8' | 'f9')[] = [
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
  ];

  const [xDomain, setXDomain] = useState<Domain>(fullDomain);
  const [yDomain, setYDomain] = useState<Domain>(fullDomain);
  const [refAreaLeft, setRefAreaLeft] = useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<number | null>(null);
  const [refAreaTop, setRefAreaTop] = useState<number | null>(null);
  const [refAreaBottom, setRefAreaBottom] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const handleMouseDown = useCallback((e: any) => {
    if (e && e.xValue != null && e.yValue != null) {
      setRefAreaLeft(e.xValue);
      setRefAreaTop(e.yValue);
    }
  }, []);

  const handleMouseMove = useCallback((e: any) => {
    if (refAreaLeft != null && e && e.xValue != null && e.yValue != null) {
      setRefAreaRight(e.xValue);
      setRefAreaBottom(e.yValue);
    }
  }, [refAreaLeft]);

  const handleMouseUp = useCallback(() => {
    if (refAreaLeft != null && refAreaRight != null && refAreaLeft !== refAreaRight) {
      const [left, right] = [refAreaLeft, refAreaRight].sort((a, b) => a - b);
      const [bottom, top] = [refAreaTop!, refAreaBottom!].sort((a, b) => a - b);
      setXDomain([left, right]);
      setYDomain([bottom, top]);
    }
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setRefAreaTop(null);
    setRefAreaBottom(null);
  }, [refAreaLeft, refAreaRight, refAreaTop, refAreaBottom]);

  const data = useMemo(() =>
    results
      .filter(r => r.status === 'success' && r.SNCI_Norm !== null && r.SCDI_Norm !== null)
      .map(r => ({ ...r, x: r.SNCI_Norm as number, y: r.SCDI_Norm as number })),
  [results]);

  const bounds = useMemo(() => {
    if (!data.length) return { x: fullDomain, y: fullDomain };
    const xs = data.map(d => d.x);
    const ys = data.map(d => d.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin || 1) * 0.08;
    const yPad = (yMax - yMin || 1) * 0.08;
    return { x: [xMin - xPad, xMax + xPad] as Domain, y: [yMin - yPad, yMax + yPad] as Domain };
  }, [data]);

  const selected = useMemo(() =>
    data.find(d => d.id === selectedId) ?? data[0] ?? null,
  [data, selectedId]);

  useEffect(() => {
    if (!data.length) {
      setSelectedId(null);
      setXDomain(fullDomain);
      setYDomain(fullDomain);
      return;
    }
    setXDomain(bounds.x);
    setYDomain(bounds.y);
    setSelectedId(prev => (prev && data.some(d => d.id === prev) ? prev : data[0].id));
  }, [data, bounds]);

  const stats = useMemo(() => {
    const counts: Record<Quadrant, number> = { Q1: 0, Q2: 0, Q3: 0, Q4: 0 };
    data.forEach(d => { counts[d.quadrant] += 1; });
    return counts;
  }, [data]);

  const medianX = useMemo(() => {
    const xs = data.map(d => d.x).sort((a, b) => a - b);
    const mid = Math.floor(xs.length / 2);
    return xs.length ? (xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2) : 0.5;
  }, [data]);

  const medianY = useMemo(() => {
    const ys = data.map(d => d.y).sort((a, b) => a - b);
    const mid = Math.floor(ys.length / 2);
    return ys.length ? (ys.length % 2 ? ys[mid] : (ys[mid - 1] + ys[mid]) / 2) : 0.5;
  }, [data]);

  const maxDescriptorVal = useMemo(() => {
    let maxVal = 0.1;
    data.forEach(r => {
      descriptorKeys.forEach(k => {
        const val = r[k];
        if (typeof val === 'number' && val > maxVal) maxVal = val;
      });
    });
    return maxVal;
  }, [data]);

  const zoom = (dir: 'in' | 'out') => {
    const factor = dir === 'in' ? 0.8 : 1.25;
    const expand = (domain: Domain, bounds: Domain) => {
      const center = (domain[0] + domain[1]) / 2;
      const half = ((domain[1] - domain[0]) * factor) / 2;
      const [newMin, newMax] = [center - half, center + half];
      return [Math.max(bounds[0], newMin), Math.min(bounds[1], newMax)] as Domain;
    };
    setXDomain(d => expand(d, bounds.x));
    setYDomain(d => expand(d, bounds.y));
  };

  const resetZoom = () => {
    if (!data.length) return;
    setXDomain(bounds.x);
    setYDomain(bounds.y);
  };

  const renderDot = (props: any) => {
    const { cx, cy, payload } = props;
    if (cx == null || cy == null) return null;
    const isSel = selected?.id === payload.id;
    return (
      <circle
        cx={cx}
        cy={cy}
        r={isSel ? 7 : 4.5}
        fill={quadrantColors[payload.quadrant as Quadrant]}
        stroke={isSel ? '#ffffff' : 'rgba(255,255,255,0.15)'}
        strokeWidth={isSel ? 2.5 : 1}
        className="transition-all duration-150 cursor-pointer"
        onClick={(e) => { e.stopPropagation(); setSelectedId(payload.id); }}
      />
    );
  };

  // knf_core/api.py has no job-cancellation support — a submitted job runs to completion.
  const notifyStopUnsupported = useCallback(() => {
    toast({
      title: 'Cancellation not supported',
      description: 'This backend runs each job to completion — there is no way to stop it mid-run.',
      variant: 'destructive',
    });
  }, [toast]);

  const copyLogs = () => {
    navigator.clipboard.writeText(logs.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return (
      <div className="p-6 lg:p-8 space-y-4 animate-fade-in-up">
        <div className="h-8 w-48 animate-shimmer rounded-lg" />
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => <div key={i} className="h-24 rounded-xl animate-shimmer" />)}
        </div>
        <div className="h-64 rounded-xl animate-shimmer" />
      </div>
    );
  }

  if (!run) {
    return (
      <div className="p-6 lg:p-8">
        <EmptyState
          title="Run not found"
          description={`No computation record found with ID "${runId}".`}
          action={
            <Link to="/" className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 text-primary text-sm font-medium border border-primary/20 hover:bg-primary/20 transition-colors">
              <ArrowLeft className="w-3.5 h-3.5" /> Back to Dashboard
            </Link>
          }
        />
      </div>
    );
  }

  const isActive = run.status === 'processing' || run.status === 'queued';
  const progressPct = run.totalFiles > 0 ? Math.round((run.completedFiles / run.totalFiles) * 100) : 0;

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-7xl animate-fade-in-up">
      {/* Top Navigation & Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
        <div>
          <Link
            to="/runs"
            className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1 mb-3 transition-colors"
          >
            <ArrowLeft className="w-3 h-3" /> Runs
          </Link>
          <h1 className="text-2xl font-display font-bold text-foreground leading-tight">{run.name}</h1>
          <p className="text-xs text-muted-foreground font-mono mt-0.5 select-all">{run.id}</p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {isActive && (
            <>
              <button
                onClick={notifyStopUnsupported}
                disabled
                title="Cancellation isn't supported by this backend"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 text-rose-400/50 text-xs font-semibold border border-rose-500/20 cursor-not-allowed transition-all"
              >
                <Square className="w-3 h-3" />
                Terminate Run
              </button>
              <span className="inline-flex items-center gap-1.5 text-xs text-primary font-medium">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Live
              </span>
            </>
          )}
          <StatusBadge status={run.status} />
        </div>
      </div>

      {/* Progress Bar (Active Runs Only) */}
      {isActive && (
        <div className="rounded-lg border border-glass bg-card/45 p-4 space-y-3">
          <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
            <span>Computation Progress</span>
            <span>{run.completedFiles} / {run.totalFiles} files processed ({progressPct}%)</span>
          </div>
          <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-700"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Files"
          value={run.totalFiles}
          icon={<FileText className="w-4 h-4" />}
          color="#6366f1"
        />
        <MetricCard
          label="Completed"
          value={run.completedFiles}
          icon={<CheckCircle2 className="w-4 h-4" />}
          color="#0ea5e9"
        />
        <MetricCard
          label="Successful"
          value={run.successFiles}
          icon={<CheckCircle2 className="w-4 h-4" />}
          color="#10b981"
        />
        <MetricCard
          label="Failed"
          value={run.failedFiles}
          icon={<XCircle className="w-4 h-4" />}
          color={run.failedFiles > 0 ? '#ef4444' : '#71717a'}
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-glass pb-1">
        <button
          onClick={() => setActiveTab('logs')}
          className={cn(
            'px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-colors',
            activeTab === 'logs'
              ? 'border-primary text-primary bg-primary/5'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          )}
        >
          <Terminal className="w-3.5 h-3.5 inline mr-1.5" /> Logs
        </button>
        <button
          onClick={() => setActiveTab('results')}
          className={cn(
            'px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-colors',
            activeTab === 'results'
              ? 'border-primary text-primary bg-primary/5'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          )}
        >
          <Table className="w-3.5 h-3.5 inline mr-1.5" /> Results ({results.length})
        </button>
        <button
          onClick={() => setActiveTab('explorer')}
          className={cn(
            'px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-colors',
            activeTab === 'explorer'
              ? 'border-primary text-primary bg-primary/5'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          )}
        >
          <BarChart2 className="w-3.5 h-3.5 inline mr-1.5" /> Explorer
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'logs' && (
        <div className="rounded-lg border border-glass overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-glass bg-zinc-950">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold text-foreground">Execution Log Stream</h2>
              {logs.length > 0 && (
                <span className="text-[10px] text-muted-foreground font-mono">({logs.length} lines)</span>
              )}
            </div>
            <button
              onClick={copyLogs}
              disabled={logs.length === 0}
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied!' : 'Copy All'}
            </button>
          </div>
          <div className="terminal-console p-4 h-80 overflow-y-auto space-y-1">
            {logs.length === 0 ? (
              <div className="flex items-center justify-center h-full text-xs text-muted-foreground/50 font-mono">
                No log output captured for this run.
              </div>
            ) : (
              logs.map((line, idx) => {
                let cls = 'terminal-row';
                if (line.startsWith('> ')) cls = 'terminal-row command text-primary/80';
                else if (/error|fail/i.test(line)) cls = 'terminal-row error text-rose-400';
                else if (/success|completed|done|finished/i.test(line)) cls = 'terminal-row success text-emerald-400';

                return (
                  <div key={idx} className={cn(cls, 'text-[10px] leading-relaxed text-muted-foreground')}>
                    {line}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}

{activeTab === 'results' && (
        <div className="rounded-lg border border-glass overflow-hidden">
          {results.length === 0 ? (
            <EmptyState
              title="No results yet"
              description={isActive ? "Results will appear here as files complete processing." : "This run has no completed results."}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-glass bg-white/[0.02] text-[9px] font-mono uppercase tracking-wider text-muted-foreground">
                    <th className="px-4 py-3 text-left">File</th>
                    <th className="px-4 py-3 text-right">SNCI</th>
                    <th className="px-4 py-3 text-right">SCDI Variance</th>
                    <th className="px-4 py-3 text-right">SNCI Norm</th>
                    <th className="px-4 py-3 text-right">SCDI Batch Norm</th>
                    <th className="px-4 py-3 text-center">Quadrant</th>
                    <th className="px-4 py-3 text-center">Status</th>
                    <th className="px-4 py-3 text-center">KUID</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-glass/30">
                  {results.map((r) => (
                    <tr key={r.id} className="hover:bg-white/[0.02] transition-colors">
                      <td className="px-4 py-3 font-mono text-foreground/90 truncate max-w-xs" title={r.fileName}>{r.fileName}</td>
                      <td className="px-4 py-3 text-right font-mono text-foreground/80">{r.SNCI?.toFixed(6) ?? '—'}</td>
                      <td className="px-4 py-3 text-right font-mono text-foreground/80">{r.SCDI_variance?.toFixed(6) ?? '—'}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{r.SNCI_Norm?.toFixed(6) ?? '—'}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{r.SCDI_Norm?.toFixed(6) ?? '—'}</td>
                      <td className="px-4 py-3 text-center">
                        <span className={cn(
                          'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-semibold border',
                          r.quadrant === 'Q1' && 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
                          r.quadrant === 'Q2' && 'bg-amber-500/10 text-amber-400 border-amber-500/20',
                          r.quadrant === 'Q3' && 'bg-rose-500/10 text-rose-400 border-rose-500/20',
                          r.quadrant === 'Q4' && 'bg-sky-500/10 text-sky-400 border-sky-500/20'
                        )}>
                          {r.quadrant}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <StatusBadge status={r.status} />
                      </td>
                      <td className="px-4 py-3 text-center font-mono text-[10px] text-muted-foreground">{r.KUID?.slice(0, 12) || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === 'explorer' && (
        <div className="space-y-6">
          {/* Controls */}
          <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-lg border border-glass bg-card/40">
            <div className="flex items-center gap-2">
              <button onClick={() => zoom('in')} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-xs text-foreground font-semibold hover:bg-zinc-900/50 transition-colors">
                <ZoomIn className="w-3.5 h-3.5" /> Zoom In
              </button>
              <button onClick={() => zoom('out')} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-xs text-foreground font-semibold hover:bg-zinc-900/50 transition-colors">
                <ZoomOut className="w-3.5 h-3.5" /> Zoom Out
              </button>
              <button onClick={resetZoom} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-xs text-foreground font-semibold hover:bg-zinc-900/50 transition-colors">
                <RotateCcw className="w-3.5 h-3.5" /> Reset
              </button>
            </div>
            <div className="text-[10px] text-muted-foreground inline-flex items-center gap-2 font-mono">
              <CircleDot className="w-3 h-3 text-primary animate-pulse" /> Click points to inspect
            </div>
          </div>

          {/* Scatter Plot + Inspector */}
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-6 items-start">
            {/* Scatter Plot */}
            <div className="rounded-lg border border-glass bg-card/45 p-5 min-h-[700px] flex flex-col justify-between">
              {data.length === 0 ? (
                <EmptyState title="No plot data" description="Run a descriptor analysis on molecular structures first." />
              ) : (
                <>
                  {/* Quadrant Legend */}
                  <div className="flex flex-wrap gap-4 text-xs text-muted-foreground border-b border-glass pb-4 mb-4">
                    {(['Q1', 'Q2', 'Q3', 'Q4'] as const).map(q => (
                      <span key={q} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.03] border border-glass/40">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: quadrantColors[q] }} />
                        <span className="font-semibold text-foreground/80">{q}</span>
                        <span className="font-mono text-white pl-1">{stats[q]}</span>
                      </span>
                    ))}
                    <span className="ml-auto font-mono text-foreground/75 inline-flex items-center">
                      Total: {data.length}
                    </span>
                  </div>

                  {/* Chart */}
                  <div className="h-[580px] w-full relative z-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart
                        margin={{ top: 20, right: 20, bottom: 20, left: 8 }}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onDoubleClick={resetZoom}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={xDomain}
                          stroke="rgba(255, 255, 255, 0.3)"
                          fontSize={10}
                          fontFamily="JetBrains Mono"
                          tickLine={false}
                        >
                          <Label value="Normalized Non-Covalent Density (SNCI_Norm)" position="bottom" offset={10} style={{ fill: 'rgba(255, 255, 255, 0.5)', fontSize: 11, fontFamily: 'Space Grotesk' }} />
                        </XAxis>
                        <YAxis
                          type="number"
                          dataKey="y"
                          domain={yDomain}
                          stroke="rgba(255, 255, 255, 0.3)"
                          fontSize={10}
                          fontFamily="JetBrains Mono"
                          tickLine={false}
                        >
                          <Label value="Normalized Spatial Intensity (SCDI_Norm)" angle={-90} position="insideLeft" offset={10} style={{ fill: 'rgba(255, 255, 255, 0.5)', fontSize: 11, fontFamily: 'Space Grotesk' }} />
                        </YAxis>
                        <ReferenceLine x={medianX} stroke="rgba(255, 255, 255, 0.15)" strokeDasharray="6 4" />
                        <ReferenceLine y={medianY} stroke="rgba(255, 255, 255, 0.15)" strokeDasharray="6 4" />
                        <Tooltip
                          content={({ payload }) => {
                            if (!payload?.[0]) return null;
                            const d = payload[0].payload as any;
                            return (
                              <div className="rounded-lg border border-glass bg-zinc-950/95 p-3 text-xs shadow-xl max-w-xs backdrop-blur-md">
                                <p className="font-semibold text-white truncate border-b border-glass/40 pb-1.5 mb-1.5">{d.fileName}</p>
                                <div className="space-y-1 font-mono text-muted-foreground text-[10px]">
                                  <p>SNCI_Norm: <span className="text-white">{Number(d.x).toFixed(6)}</span></p>
                                  <p>SCDI_Norm: <span className="text-white">{Number(d.y).toFixed(6)}</span></p>
                                  <p className="flex items-center gap-1 mt-1 pt-1 border-t border-white/5">
                                    Quadrant: <span className="px-1.5 py-0.5 rounded-full text-[9px]" style={{ backgroundColor: `${quadrantColors[d.quadrant as Quadrant]}22`, color: quadrantColors[d.quadrant as Quadrant] }}>{d.quadrant}</span>
                                  </p>
                                </div>
                              </div>
                            );
                          }}
                        />
                        <Scatter data={data} shape={renderDot} />
                        {refAreaLeft != null && refAreaRight != null ? (
                          <ReferenceArea
                            x1={refAreaLeft}
                            x2={refAreaRight}
                            y1={refAreaTop!}
                            y2={refAreaBottom!}
                            strokeOpacity={0.3}
                            fill="rgba(255, 255, 255, 0.08)"
                            stroke="#ffffff"
                          />
                        ) : null}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </div>

            {/* Inspector Sidebar */}
            <aside className="rounded-lg border border-glass bg-card/45 p-5 space-y-6 max-h-[80vh] overflow-y-auto sticky top-8">
              {!selected ? (
                <EmptyState title="Select a Point" description="Select any coordinate point in the scatter cloud to load its 3D model and descriptor details." />
              ) : (
                <div className="space-y-5">
                  <div className="border-b border-glass pb-4">
                    <span className="text-[9px] text-primary font-mono tracking-widest uppercase block">Selected Descriptor</span>
                    <h2 className="text-xl font-display font-bold text-foreground truncate mt-0.5" title={selected.fileName}>
                      {selected.fileName}
                    </h2>
                    <div className="flex items-center gap-2 mt-2">
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold border ${quadrantBgs[selected.quadrant]}`}>
                        Quadrant {selected.quadrant}
                      </span>
                      <StatusBadge status={selected.status} />
                    </div>
                  </div>

                  {/* Normalized Metrics */}
                  <div className="grid grid-cols-2 gap-3 border-t border-glass pt-4">
                    <div className="rounded-lg border border-zinc-800 bg-zinc-900/10 p-3">
                      <span className="text-[9px] text-muted-foreground uppercase font-semibold block">SNCI Raw</span>
                      <span className="text-sm font-mono font-bold text-foreground block mt-0.5">{selected.SNCI?.toFixed(6) ?? '—'}</span>
                    </div>
                    <div className="rounded-lg border border-zinc-800 bg-zinc-900/10 p-3">
                      <span className="text-[9px] text-muted-foreground uppercase font-semibold block">SCDI Variance (Raw)</span>
                      <span className="text-sm font-mono font-bold text-foreground block mt-0.5">{selected.SCDI_variance?.toFixed(6) ?? '—'}</span>
                    </div>
                    <div className="rounded-lg border border-zinc-800 bg-zinc-900/10 p-3">
                      <span className="text-[9px] text-muted-foreground uppercase font-semibold block">SNCI Norm</span>
                      <span className="text-sm font-mono font-bold text-foreground block mt-0.5">{selected.SNCI_Norm?.toFixed(6) ?? '—'}</span>
                    </div>
                    <div className="rounded-lg border border-zinc-800 bg-zinc-900/10 p-3">
                      <span className="text-[9px] text-muted-foreground uppercase font-semibold block">SCDI Batch Norm</span>
                      <span className="text-sm font-mono font-bold text-foreground block mt-0.5">{selected.SCDI_Norm?.toFixed(6) ?? '—'}</span>
                    </div>
                    <div className="col-span-2 rounded-lg border border-zinc-800 bg-zinc-900/10 p-3 flex items-center justify-between font-mono">
                      <div>
                        <span className="text-[9px] text-muted-foreground uppercase font-semibold block">KUID Hash</span>
                        <span className="text-xs text-white tracking-wide mt-0.5 block">{selected.KUID || 'N/A'}</span>
                      </div>
                      {selected.KUID_Cluster && (
                        <span className="text-[9px] font-semibold px-2 py-0.5 rounded bg-primary/15 text-primary border border-primary/25 shrink-0">
                          Cluster {selected.KUID_Cluster}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* 9-D Descriptor Gauges */}
                  <div className="space-y-3 border-t border-glass pt-4">
                    <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">Fingerprint Feature Grid</h3>
                    <div className="grid grid-cols-3 gap-2">
                      {descriptorKeys.map(k => {
                        const val = selected[k];
                        const ratio = typeof val === 'number' ? val / maxDescriptorVal : 0;
                        return (
                          <div key={k} className="rounded-lg border border-zinc-800 bg-zinc-900/10 p-2 flex flex-col justify-between">
                            <span className="text-[9px] font-mono text-muted-foreground font-semibold">{k}</span>
                            <span className="text-xs font-mono font-bold text-foreground break-all mt-1">
                              {typeof val === 'number' ? val.toFixed(6) : '-'}
                            </span>
                            <div className="w-full bg-zinc-850 h-[3px] rounded-full overflow-hidden mt-1.5">
                              <div
                                className="h-full"
                                style={{
                                  width: `${ratio * 100}%`,
                                  backgroundColor: quadrantColors[selected.quadrant],
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </aside>
          </div>
        </div>
      )}

      {/* Config Summary */}
      {run.config && Object.keys(run.config).length > 0 && (
        <div className="rounded-lg border border-glass bg-card/45 overflow-hidden">
          <div className="px-5 py-3 border-b border-glass">
            <h2 className="text-sm font-semibold text-foreground">Execution Configuration</h2>
          </div>
          <div className="p-5 grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(run.config)
              .filter(([, v]) => v !== null && v !== undefined && v !== '')
              .map(([key, value]) => (
                <div key={key} className="rounded-lg border border-zinc-800 bg-zinc-900/10 px-3 py-2">
                  <span className="text-[9px] text-muted-foreground font-mono uppercase tracking-wider block">{key}</span>
                  <span className="text-xs font-mono text-foreground/90 block mt-0.5 truncate">
                    {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

function MetricCard({ label, value, icon, color }: { label: string; value: number; icon: React.ReactNode; color: string }) {
  return (
    <div className="rounded-lg border border-glass bg-card/45 p-4 card-hover-glow">
      <div className="flex items-center gap-2 mb-2">
        <div style={{ color }}>{icon}</div>
        <span className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-2xl font-display font-bold text-foreground">{value}</p>
    </div>
  );
}

export default RunDetails;
