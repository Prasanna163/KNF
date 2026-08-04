import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Activity, CheckCircle2, XCircle, Clock, BarChart2,
  FileText, Loader2, Play, ArrowRight, ChevronDown, ChevronUp,
  Zap, Cpu, RefreshCw
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { subscribeWsMessages, getBufferedMessages } from '@/lib/wsStore';
import type { ResultRecord } from '@/types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function quadrantColor(q: string) {
  return q === 'Q1' ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
       : q === 'Q2' ? 'text-amber-400 bg-amber-500/10 border-amber-500/20'
       : q === 'Q3' ? 'text-rose-400 bg-rose-500/10 border-rose-500/20'
       :               'text-sky-400 bg-sky-500/10 border-sky-500/20';
}

function Pill({ value, className }: { value: string; className?: string }) {
  return (
    <span className={cn('inline-flex px-1.5 py-0.5 rounded border text-[10px] font-mono font-semibold', className)}>
      {value}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] font-mono uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className="text-sm font-bold font-mono text-foreground">{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LiveFileResult extends ResultRecord {
  arrivedAt: number;
}

interface PhaseLog {
  type: 'log' | 'command' | 'error' | 'status' | 'info';
  message: string;
  ts: number;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const Analysis = () => {
  const navigate = useNavigate();

  const [liveResults, setLiveResults] = useState<LiveFileResult[]>([]);
  const [logs, setLogs] = useState<PhaseLog[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [isFailed, setIsFailed] = useState(false);
  const [totalFiles, setTotalFiles] = useState(0);
  const [normUpdated, setNormUpdated] = useState(false);
  const [normCount, setNormCount] = useState(0);
  const [showLogs, setShowLogs] = useState(false);

  const logBottomRef = useRef<HTMLDivElement>(null);
  const seenIds = useRef(new Set<string>());

  // ── Handle a single WS message (inline — no type complexity from params) ──
  const handleMsg = useRef((msg: Record<string, unknown>) => { void msg; });

  // Keep handleMsg.current up-to-date with latest state setters via stable ref
  useEffect(() => {
    handleMsg.current = (msg: Record<string, unknown>) => {
      const type = msg.type as string;
      const message = ((msg.message ?? '') as string);

      const addLog = (logType: PhaseLog['type'], text: string) =>
        setLogs(p => [...p, { type: logType, message: text, ts: Date.now() }].slice(-200) as PhaseLog[]);

      if (type === 'status') {
        setIsRunning(true);
        setIsComplete(false);
        setIsFailed(false);
        const match = message.match(/(\d+) file/);
        if (match) setTotalFiles(parseInt(match[1], 10));
        addLog('status', message);

      } else if (type === 'log') {
        addLog('log', message);

      } else if (type === 'command') {
        addLog('command', message);

      } else if (type === 'file_result' && msg.result) {
        const r = msg.result as ResultRecord;
        if (!seenIds.current.has(r.id)) {
          seenIds.current.add(r.id);
          setLiveResults(p => [...p, { ...r, arrivedAt: Date.now() }]);
          addLog('info', `✓ ${r.fileName} — SNCI ${r.SNCI?.toFixed(6) ?? 'unavailable'} · SCDI variance ${r.SCDI_variance?.toFixed(6) ?? 'unavailable'}`);
        }

      } else if (type === 'completed') {
        setIsRunning(false);
        setIsComplete(true);
        addLog('info', message);

      } else if (type === 'error') {
        setIsRunning(false);
        setIsFailed(true);
        addLog('error', message);

      } else if (type === 'normalized_update') {
        setNormUpdated(true);
        setNormCount(typeof msg.count === 'number' ? msg.count : 0);
        addLog('info', message);
      }
    };
  });

  // ── Hydrate from buffered messages (sent before this page mounted) ────────
  useEffect(() => {
    getBufferedMessages().forEach(m => handleMsg.current(m));
     
  }, []);

  // ── Subscribe to live messages published by RunManager ───────────────────
  useEffect(() => {
    return subscribeWsMessages(msg => handleMsg.current(msg));
  }, []);

  // Scroll logs to bottom
  useEffect(() => {
    logBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // ── Computed ─────────────────────────────────────────────────────────────
  const done = liveResults.length;
  const pct = totalFiles > 0 ? Math.round((done / totalFiles) * 100) : 0;

  const validSNCI = liveResults.map(r => r.SNCI).filter((v): v is number => v !== null);
  const validSCDIVariance = liveResults.map(r => r.SCDI_variance).filter((v): v is number => v !== null);
  const avgSNCI = validSNCI.length > 0
    ? validSNCI.reduce((sum, value) => sum + value, 0) / validSNCI.length
    : null;
  const avgSCDI = validSCDIVariance.length > 0
    ? validSCDIVariance.reduce((sum, value) => sum + value, 0) / validSCDIVariance.length
    : null;

  const quadrantCounts = liveResults.reduce<Record<string, number>>((acc, r) => {
    acc[r.quadrant] = (acc[r.quadrant] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="p-6 lg:p-8 max-w-7xl space-y-6 animate-fade-in-up">
      {/* Header */}
      <header className="space-y-1">
        <div className="flex items-center gap-2 text-primary font-mono tracking-widest text-[9px] uppercase">
          <Activity className="w-3.5 h-3.5" />
          Live Analysis Window
        </div>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">
              Run <span className="text-gradient">Analysis</span>
            </h1>
            <p className="text-xs text-muted-foreground mt-1">
              Results stream here as each file completes — separate from Run Manager
            </p>
          </div>
          <div className="flex items-center gap-3">
            {normUpdated && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-mono">
                <RefreshCw className="w-3 h-3" />
                Global norms updated · {normCount} molecules
              </div>
            )}
            <button
              onClick={() => navigate('/results')}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary/10 border border-primary/20 text-primary text-xs font-semibold hover:bg-primary/20 transition-colors"
            >
              View All Results <ArrowRight className="w-3 h-3" />
            </button>
            <button
              onClick={() => navigate('/runs')}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-glass bg-white/5 hover:bg-white/10 text-xs font-semibold transition-colors"
            >
              <Play className="w-3 h-3" /> New Run
            </button>
          </div>
        </div>
      </header>

      {/* Status Bar */}
      <div className="rounded-xl border border-glass bg-card/50 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isRunning && (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-primary">
                <Loader2 className="w-4 h-4 animate-spin" /> Processing...
              </span>
            )}
            {isComplete && !isRunning && (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
                <CheckCircle2 className="w-4 h-4" /> Completed
              </span>
            )}
            {isFailed && (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-rose-400">
                <XCircle className="w-4 h-4" /> Failed
              </span>
            )}
            {!isRunning && !isComplete && !isFailed && (
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono">
                <Clock className="w-3.5 h-3.5" /> Waiting for a run — go to Run Manager to start
              </span>
            )}
          </div>
          <span className="text-xs font-mono text-muted-foreground">
            {done}{totalFiles > 0 ? ` / ${totalFiles}` : ''} files complete
          </span>
        </div>

        {/* Progress Bar */}
        {(isRunning || isComplete) && totalFiles > 0 && (
          <div className="space-y-1">
            <div className="h-2 rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary to-violet-500 rounded-full transition-all duration-700 ease-out"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
              <span>{pct}% complete</span>
              {done > 0 && avgSNCI !== null && (
                <span>avg SNCI {avgSNCI?.toFixed(6) ?? '—'} · avg SCDI variance {avgSCDI?.toFixed(6) ?? '—'}</span>
              )}
            </div>
          </div>
        )}

        {/* Quadrant mini-summary */}
        {done > 0 && (
          <div className="flex flex-wrap gap-2 pt-1 border-t border-glass/30">
            {(['Q1', 'Q2', 'Q3', 'Q4'] as const).map(q => (
              <div
                key={q}
                className={cn(
                  'flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[10px] font-mono font-semibold',
                  quadrantColor(q)
                )}
              >
                <span>{q}</span>
                <span className="opacity-80">{quadrantCounts[q] ?? 0}</span>
              </div>
            ))}
            <div className="ml-auto flex items-center gap-1 text-[10px] text-muted-foreground font-mono">
              <Cpu className="w-3 h-3" />
              quadrant distribution (current run)
            </div>
          </div>
        )}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live Results Table */}
        <div className="lg:col-span-2 space-y-3">
          <div className="flex items-center justify-between border-b border-glass pb-2">
            <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider flex items-center gap-2">
              <BarChart2 className="w-3.5 h-3.5 text-primary" />
              Live Results
            </h2>
            <span className="text-[10px] font-mono text-muted-foreground">{done} records</span>
          </div>

          {liveResults.length === 0 ? (
            <div className="rounded-xl border border-dashed border-zinc-800 bg-zinc-900/10 py-20 flex flex-col items-center gap-3 text-muted-foreground/50">
              <Zap className="w-8 h-8" />
              <p className="text-xs font-mono">Results will stream here as each file completes</p>
              <p className="text-[10px]">Go to Run Manager to start a new computation</p>
            </div>
          ) : (
            <div className="rounded-xl border border-glass overflow-hidden">
              {/* Column Headers */}
              <div className="grid grid-cols-[1fr_100px_100px_80px_80px_72px] border-b border-glass/40 bg-white/[0.02]">
                {['File', 'SNCI', 'SCDI Variance', 'SNCI Norm', 'SCDI Batch Norm', 'Quad'].map(h => (
                  <div key={h} className="px-3 py-2 text-[9px] font-mono uppercase tracking-wider text-muted-foreground/70">
                    {h}
                  </div>
                ))}
              </div>
              {/* Result Rows */}
              <div className="divide-y divide-glass/20 max-h-[520px] overflow-y-auto">
                {liveResults.map((r, i) => (
                  <div
                    key={r.id}
                    className="grid grid-cols-[1fr_100px_100px_80px_80px_72px] items-center hover:bg-white/[0.03] transition-colors animate-slide-in-row"
                    style={{ animationDelay: `${Math.min(i, 12) * 0.04}s` }}
                  >
                    <div className="px-3 py-2.5 flex items-center gap-2 min-w-0">
                      <FileText className="w-3 h-3 text-primary/70 shrink-0" />
                      <span className="text-xs font-mono text-foreground/90 truncate" title={r.fileName}>
                        {r.fileName}
                      </span>
                    </div>
                    <div className="px-3 py-2.5 text-xs font-mono text-foreground/80">
                      {r.SNCI?.toFixed(6) ?? '—'}
                    </div>
                    <div className="px-3 py-2.5 text-xs font-mono text-foreground/80">
                      {r.SCDI_variance?.toFixed(6) ?? '—'}
                    </div>
                    <div className="px-3 py-2.5 text-xs font-mono text-muted-foreground">
                      {r.SNCI_Norm?.toFixed(6) ?? '—'}
                    </div>
                    <div className="px-3 py-2.5 text-xs font-mono text-muted-foreground">
                      {r.SCDI_Norm?.toFixed(6) ?? '—'}
                    </div>
                    <div className="px-3 py-2.5">
                      <Pill value={r.quadrant} className={quadrantColor(r.quadrant)} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Panel — Terminal + Stats */}
        <div className="space-y-3">
          <div className="flex items-center justify-between border-b border-glass pb-2">
            <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider">
              Terminal
            </h2>
            <button
              onClick={() => setShowLogs(!showLogs)}
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label={showLogs ? 'Collapse terminal' : 'Expand terminal'}
            >
              {showLogs ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>

          <div
            className={cn(
              'rounded-xl border border-zinc-800 bg-zinc-950 p-4 overflow-hidden flex flex-col transition-all duration-300',
              showLogs ? 'h-[560px]' : 'h-[380px]'
            )}
          >
            <div className="flex items-center justify-between border-b border-glass/30 pb-2 mb-2 shrink-0">
              <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider">
                Execution Stream
              </span>
              {isRunning && (
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              )}
            </div>
            <div className="flex-1 overflow-y-auto space-y-1 text-[10px] font-mono pr-1">
              {logs.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground/40 text-center">
                  Console idle — start a run to see output
                </div>
              ) : (
                logs.map((l, i) => (
                  <div
                    key={i}
                    className={cn(
                      'leading-relaxed break-all',
                      l.type === 'command' ? 'text-primary/90' :
                      l.type === 'error'   ? 'text-rose-400' :
                      l.type === 'info'    ? 'text-emerald-400' :
                      l.type === 'status'  ? 'text-amber-400' :
                                             'text-muted-foreground'
                    )}
                  >
                    {l.type === 'command' ? `> ${l.message}` : l.message}
                  </div>
                ))
              )}
              <div ref={logBottomRef} />
            </div>
          </div>

          {/* Stats Panel */}
          {done > 0 && (
            <div className="rounded-xl border border-glass bg-card/40 p-4 grid grid-cols-2 gap-4">
              <Stat label="Files processed" value={done} />
              <Stat label="Avg SNCI" value={avgSNCI?.toFixed(6) ?? '—'} />
              <Stat label="Avg SCDI Variance" value={avgSCDI?.toFixed(6) ?? '—'} />
              <Stat label="Top quadrant" value={
                Object.entries(quadrantCounts).sort(([, a], [, b]) => b - a)[0]?.[0] ?? '—'
              } />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;
