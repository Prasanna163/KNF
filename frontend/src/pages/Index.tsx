import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Play, BarChart3, Activity, CheckCircle2, Zap, Clock,
  RefreshCw, Square, ChevronRight, Atom, FlaskConical, Database
} from 'lucide-react';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { EmptyState } from '@/components/shared/EmptyState';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import type { Run } from '@/types';
import { hydrateAllRuns } from '@/lib/runs';

interface StatTileProps {
  label: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  accentColor: string;
  borderColor: string;
  delay?: number;
}

function StatTile({ label, value, subtitle, icon, accentColor, delay = 0 }: Omit<StatTileProps, 'borderColor'>) {
  return (
    <div
      className="rounded-lg border bg-card p-5 card-hover-glow animate-fade-in-up"
      style={{
        animationDelay: `${delay}s`,
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1 min-w-0">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{label}</p>
          <p className="text-3xl font-display font-bold text-foreground leading-none">{value}</p>
          {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
        </div>
        <div
          className="flex items-center justify-center w-10 h-10 rounded-lg shrink-0"
          style={{ backgroundColor: `${accentColor}12`, border: `1px solid ${accentColor}25` }}
        >
          <div style={{ color: accentColor }}>{icon}</div>
        </div>
      </div>
    </div>
  );
}

const Dashboard = () => {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  const fetchRuns = useCallback(async () => {
    try {
      setRuns(await hydrateAllRuns());
    } catch {
      setRuns([]);
    }
  }, []);

  // knf_core/api.py has no job-cancellation support — see RunManager.tsx.
  const notifyStopUnsupported = useCallback(() => {
    toast({
      title: 'Cancellation not supported',
      description: 'This backend runs each job to completion — there is no way to stop it mid-run.',
      variant: 'destructive',
    });
  }, [toast]);

  useEffect(() => {
    const mounted = true;
    (async () => {
      await fetchRuns();
      if (mounted) setLoading(false);
    })();
  }, [fetchRuns]);

  // Poll every 3s while processing
  useEffect(() => {
    const hasActive = runs.some(r => r.status === 'processing' || r.status === 'queued');
    if (!hasActive) return;
    const id = setInterval(fetchRuns, 3000);
    return () => clearInterval(id);
  }, [runs, fetchRuns]);

  const stats = useMemo(() => {
    const totalRuns = runs.length;
    const activeRuns = runs.filter(r => r.status === 'processing' || r.status === 'queued').length;
    const success = runs.filter(r => r.status === 'completed').length;
    const successRate = totalRuns > 0 ? Math.round((success / totalRuns) * 100) : 0;
    const totalFiles = runs.reduce((acc, r) => acc + (r.totalFiles || 0), 0);
    const lastThroughput = runs[0]?.throughput ?? 0;
    return { totalRuns, activeRuns, successRate, lastThroughput, totalFiles };
  }, [runs]);

  return (
    <div className="p-6 lg:p-8 space-y-8 max-w-7xl">

      {/* Header */}
      <header className="animate-fade-in-up flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-primary font-mono tracking-widest text-[10px] uppercase mb-2">
            <FlaskConical className="w-3.5 h-3.5" />
            KNF Descriptor Engine
          </div>
          <h1 className="text-4xl font-display font-bold text-foreground tracking-tight">
            <span className="text-gradient">KNF Studio</span>
          </h1>
          <p className="text-muted-foreground mt-1 text-sm max-w-lg">
            Quantum–chemical molecular descriptor computation platform for non–covalent interaction analysis.
          </p>
        </div>
        {stats.activeRuns > 0 && (
          <div className="shrink-0 flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/20">
            <RefreshCw className="w-3.5 h-3.5 text-primary animate-spin" />
            <span className="text-xs font-medium text-primary">{stats.activeRuns} active</span>
          </div>
        )}
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 animate-fade-in-up" style={{ animationDelay: '0.05s' }}>
        <StatTile
          label="Total Runs"
          value={stats.totalRuns}
          subtitle="Lifetime computations"
          icon={<Activity className="w-4 h-4" />}
          accentColor="#3b82f6"
          delay={0.0}
        />
        <StatTile
          label="Active Runs"
          value={stats.activeRuns}
          subtitle="Currently processing"
          icon={<Zap className="w-4 h-4" />}
          accentColor="#f59e0b"
          delay={0.05}
        />
        <StatTile
          label="Success Rate"
          value={`${stats.successRate}%`}
          subtitle={`${runs.filter(r => r.status === 'completed').length} completed`}
          icon={<CheckCircle2 className="w-4 h-4" />}
          accentColor="#10b981"
          delay={0.1}
        />
        <StatTile
          label="Molecules Analyzed"
          value={stats.totalFiles}
          subtitle="Across all runs"
          icon={<Atom className="w-4 h-4" />}
          accentColor="#6366f1"
          delay={0.15}
        />
      </div>

      {/* Quick Actions Row */}
      <div className="flex flex-wrap gap-3 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
        <Link
          to="/runs"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 transition-all shadow"
        >
          <Play className="w-3.5 h-3.5" /> Start New Run
        </Link>
        <Link
          to="/results"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg border border-glass bg-white/5 hover:bg-white/10 text-xs font-semibold transition-all"
        >
          <BarChart3 className="w-3.5 h-3.5" /> Browse Results Library
        </Link>
        <Link
          to="/explorer"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg border border-glass bg-white/5 hover:bg-white/10 text-xs font-semibold transition-all"
        >
          <Database className="w-3.5 h-3.5" /> Open Scatter Explorer
        </Link>
      </div>

      {/* Run History Table */}
      <div className="rounded-lg border border-glass glass-panel overflow-hidden animate-fade-in-up" style={{ animationDelay: '0.25s' }}>
        <div className="px-6 py-4 border-b border-glass flex items-center justify-between">
          <h2 className="text-sm font-semibold text-foreground">Recent Computation Runs</h2>
          <Link
            to="/runs"
            className="text-xs font-medium text-primary hover:text-primary/80 transition-colors inline-flex items-center gap-1"
          >
            View all <ChevronRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {loading ? (
          <div className="p-6 space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-12 rounded-lg animate-shimmer" style={{ animationDelay: `${i * 0.15}s` }} />
            ))}
          </div>
        ) : runs.length === 0 ? (
          <div className="p-8">
            <EmptyState
              title="No computation runs yet"
              description="Upload molecular structure files in the Run Manager to begin descriptor calculations."
              action={
                <Link to="/runs" className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 text-primary text-sm font-medium border border-primary/20 hover:bg-primary/20 transition-colors">
                  <Play className="w-3.5 h-3.5" /> Launch Run Manager
                </Link>
              }
            />
          </div>
        ) : (
          <div className="divide-y divide-glass/40">
            {runs.slice(0, 10).map((run, idx) => {
              const isActive = run.status === 'processing' || run.status === 'queued';
              const progressPct = run.totalFiles > 0
                ? Math.round((run.completedFiles / run.totalFiles) * 100)
                : 0;

              return (
                <div
                  key={run.id}
                  className="flex items-center gap-4 px-6 py-3.5 hover:bg-white/[0.03] transition-colors group"
                  style={{ animationDelay: `${idx * 0.03}s` }}
                >
                  {/* Run Info */}
                  <Link to={`/runs/${run.id}`} className="min-w-0 flex-1 flex items-center gap-4">
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-foreground truncate">{run.name}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <p className="text-[10px] text-muted-foreground font-mono truncate">{run.id}</p>
                        <span className="text-muted-foreground/40">·</span>
                        <p className="text-[10px] text-muted-foreground font-mono shrink-0">
                          {run.totalFiles} file{run.totalFiles !== 1 ? 's' : ''}
                        </p>
                      </div>
                    </div>

                    {/* Progress bar for active runs */}
                    {isActive && run.totalFiles > 0 && (
                      <div className="hidden sm:flex items-center gap-2 shrink-0">
                        <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary rounded-full transition-all duration-500"
                            style={{ width: `${progressPct}%` }}
                          />
                        </div>
                        <span className="text-[10px] font-mono text-muted-foreground">{progressPct}%</span>
                      </div>
                    )}
                  </Link>

                  {/* Actions */}
                  <div className="flex items-center gap-2 shrink-0">
                    {isActive && (
                      <button
                        onClick={e => { e.preventDefault(); notifyStopUnsupported(); }}
                        disabled
                        title="Cancellation isn't supported by this backend"
                        className="opacity-0 group-hover:opacity-100 inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-rose-500/10 text-rose-400/50 text-xs font-medium border border-rose-500/20 cursor-not-allowed transition-all"
                      >
                        <Square className="w-3 h-3" />
                        Stop
                      </button>
                    )}
                    <StatusBadge status={run.status} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
