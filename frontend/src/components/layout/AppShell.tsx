import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, Play, BarChart3, Settings, BookOpen,
  ChevronLeft, ChevronRight, Atom, CircleDot, Zap, Activity
} from 'lucide-react';
import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, description: 'Overview & history' },
  { to: '/runs', label: 'Run Manager', icon: Play, description: 'Execute computations' },
  { to: '/analysis', label: 'Analysis', icon: Activity, description: 'Live run results' },
  { to: '/results', label: 'Results', icon: BarChart3, description: 'All historical results' },
  { to: '/explorer', label: 'Explorer', icon: CircleDot, description: 'Scatter analysis' },
  { to: '/settings', label: 'Settings', icon: Settings, description: 'System settings' },
  { to: '/docs', label: 'Documentation', icon: BookOpen, description: 'API reference' },
];

function BackendStatus() {
  const [status, setStatus] = useState<'connected' | 'offline' | 'checking'>('checking');

  useEffect(() => {
    let retries = 0;
    const maxRetries = 5;
    const check = async () => {
      try {
        const raw = localStorage.getItem('knf-settings');
        let base = raw ? JSON.parse(raw).apiBaseUrl : 'http://127.0.0.1:8000';
        if (base && base.includes(':8766')) base = base.replace(':8766', ':8000');
        const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
          setStatus('connected');
          retries = 0;
          return;
        }
      } catch { /* ignore */ }
      retries++;
      if (retries >= maxRetries) {
        setStatus('offline');
      }
    };
    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={cn(
      'inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-mono font-semibold',
      status === 'connected' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
      status === 'offline' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' :
      'bg-amber-500/10 text-amber-400 border border-amber-500/20'
    )}>
      <span className={cn(
        'w-1.5 h-1.5 rounded-full',
        status === 'connected' ? 'bg-emerald-400 animate-pulse' :
        status === 'offline' ? 'bg-rose-400' :
        'bg-amber-400 animate-pulse'
      )} />
      {status === 'connected' ? 'API Online' : status === 'offline' ? 'API Offline' : 'Connecting...'}
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <div className="flex min-h-screen w-full bg-background bg-grid">
      {/* Sidebar */}
      <aside
        className={cn(
          'sticky top-0 h-screen flex flex-col border-r border-glass glass-panel-heavy transition-all duration-300 z-30 shrink-0',
          collapsed ? 'w-[68px]' : 'w-64'
        )}
      >
        {/* Logo + Branding */}
        <div className="flex items-center gap-3 px-4 h-[70px] border-b border-glass shrink-0">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 border border-primary/20 shrink-0">
            <Atom className="w-5 h-5 text-primary" />
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <span className="font-display font-bold text-foreground text-sm tracking-tight block leading-tight">
                NCIForge
              </span>
              <span className="text-[9px] text-muted-foreground font-mono leading-tight block mt-0.5">
                Descriptor Engine v1.0
              </span>
            </div>
          )}
        </div>

        {/* Backend Status Badge (only when expanded) */}
        {!collapsed && (
          <div className="px-4 pt-3">
            <BackendStatus />
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
          {navItems.map((item, i) => {
            const active = item.to === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(item.to);

            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  'flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 group relative overflow-hidden',
                  active
                    ? 'bg-primary/10 text-primary border border-primary/20'
                    : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground border border-transparent'
                )}
                aria-current={active ? 'page' : undefined}
                style={{ animationDelay: `${i * 0.05}s` }}
                title={collapsed ? `${item.label} — ${item.description}` : undefined}
              >
                {/* Active glow line */}
                {active && (
                  <span className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-primary" />
                )}
                <item.icon className={cn(
                  'w-[16px] h-[16px] shrink-0 transition-transform duration-200',
                  !active && 'group-hover:scale-105'
                )} />
                {!collapsed && (
                  <div className="min-w-0 flex-1">
                    <span className="block leading-tight truncate">{item.label}</span>
                    {!active && (
                      <span className="text-[9px] text-muted-foreground/50 block truncate leading-tight mt-0.5">
                        {item.description}
                      </span>
                    )}
                  </div>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Version & Collapse Toggle */}
        <div className="border-t border-glass shrink-0">
          {!collapsed && (
            <div className="px-4 py-2 flex items-center gap-2">
              <Zap className="w-3 h-3 text-primary/60" />
              <span className="text-[10px] font-mono text-muted-foreground/60">NCIForge · xTB · PyTorch</span>
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex items-center justify-center w-full h-10 border-t border-glass text-muted-foreground hover:text-foreground hover:bg-white/5 transition-colors"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 min-w-0 overflow-auto">
        {children}
      </main>
    </div>
  );
}
