import { useCallback, useEffect, useMemo, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Sparkles, CircleDot, HelpCircle, ChevronRight } from 'lucide-react';
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
import { EmptyState } from '@/components/shared/EmptyState';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { MoleculeViewer } from '@/components/MoleculeViewer';
import type { ResultRecord, Quadrant } from '@/types';
import { listJobs, getInputContent } from '@/lib/api';
import { getRunIdForJob } from '@/lib/runs';
import { buildResultRecords } from '@/lib/descriptors';

type Domain = [number, number];

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

const fullDomain: Domain = [0, 1];

const descriptorKeys: ('f1' | 'f2' | 'f3' | 'f4' | 'f5' | 'f6' | 'f7' | 'f8' | 'f9')[] = [
  'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'
];

function clampDomain([min, max]: Domain, bounds: Domain): Domain {
  const lower = Math.max(bounds[0], min);
  const upper = Math.min(bounds[1], max);
  if (upper <= lower) return bounds;
  return [lower, upper];
}

function expandOrShrink(domain: Domain, factor: number, bounds: Domain): Domain {
  const center = (domain[0] + domain[1]) / 2;
  const half = ((domain[1] - domain[0]) * factor) / 2;
  return clampDomain([center - half, center + half], bounds);
}

const Explorer = () => {
  const [results, setResults] = useState<ResultRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [xDomain, setXDomain] = useState<Domain>(fullDomain);
  const [yDomain, setYDomain] = useState<Domain>(fullDomain);

  const [refAreaLeft, setRefAreaLeft] = useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<number | null>(null);
  const [refAreaTop, setRefAreaTop] = useState<number | null>(null);
  const [refAreaBottom, setRefAreaBottom] = useState<number | null>(null);

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

  const [structureData, setStructureData] = useState<string | null>(null);
  const [structureLoading, setStructureLoading] = useState(false);

  const fetchResults = useCallback(async () => {
    try {
      const jobs = await listJobs();
      setResults(buildResultRecords(jobs, getRunIdForJob));
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    (async () => {
      await fetchResults();
      if (!mounted) return;
    })();
    const timer = setInterval(fetchResults, 5000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [fetchResults]);

  const data = useMemo(() => results
    .filter(r => r.status === 'success' && r.SNCI_Norm !== null && r.SCDI_Norm !== null)
    .map(r => ({ ...r, x: r.SNCI_Norm as number, y: r.SCDI_Norm as number })), [results]);

  const bounds = useMemo(() => {
    if (!data.length) return { x: fullDomain, y: fullDomain };
    const xs = data.map(d => d.x);
    const ys = data.map(d => d.y);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xPad = (xMax - xMin || 1) * 0.08;
    const yPad = (yMax - yMin || 1) * 0.08;
    return { x: [xMin - xPad, xMax + xPad] as Domain, y: [yMin - yPad, yMax + yPad] as Domain };
  }, [data]);

  const selected = useMemo(() => data.find(d => d.id === selectedId) ?? data[0] ?? null, [data, selectedId]);

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

  // Fetch 3D Structure content when selected changes
  useEffect(() => {
    if (!selected) {
      setStructureData(null);
      return;
    }
    let activeFetch = true;
    setStructureLoading(true);
    setStructureData(null);

    getInputContent(selected.id)
      .then(content => {
        if (activeFetch) {
          setStructureData(content);
          setStructureLoading(false);
        }
      })
      .catch(() => {
        if (activeFetch) {
          setStructureData(null);
          setStructureLoading(false);
        }
      });

    return () => {
      activeFetch = false;
    };
  }, [selected]);

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
        if (typeof val === 'number' && val > maxVal) {
          maxVal = val;
        }
      });
    });
    return maxVal;
  }, [data]);

  const zoom = (direction: 'in' | 'out') => {
    const factor = direction === 'in' ? 0.8 : 1.25;
    setXDomain(d => expandOrShrink(d, factor, bounds.x));
    setYDomain(d => expandOrShrink(d, factor, bounds.y));
  };

  const resetZoom = () => {
    if (!data.length) return;
    setXDomain(bounds.x);
    setYDomain(bounds.y);
  };

  const renderDot = (props: any) => {
    const { cx, cy, payload } = props;
    if (cx == null || cy == null) return null;
    const isSelected = selected?.id === payload.id;
    return (
      <circle
        cx={cx}
        cy={cy}
        r={isSelected ? 7 : 4.5}
        fill={quadrantColors[payload.quadrant as Quadrant]}
        stroke={isSelected ? '#ffffff' : 'rgba(255,255,255,0.15)'}
        strokeWidth={isSelected ? 2.5 : 1}
        className="transition-all duration-150 cursor-pointer"
        onClick={(e) => {
          e.stopPropagation();
          setSelectedId(payload.id);
        }}
      />
    );
  };

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-[1600px] animate-fade-in-up">
      <header className="space-y-1">
        <div className="flex items-center gap-2 text-primary font-mono tracking-widest text-[9px] uppercase">
          <Sparkles className="w-3.5 h-3.5" />
          Interactive Visualizer
        </div>
        <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">
          Scatter <span className="text-gradient">Explorer</span>
        </h1>
        <p className="text-xs text-muted-foreground">Compare structure-activity descriptors on a 2D quadrant space</p>
      </header>

      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-lg border border-glass bg-card/40">
        <div className="flex items-center gap-2">
          <button onClick={() => zoom('in')} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-xs text-foreground font-semibold hover:bg-zinc-900/50 transition-colors">
            <ZoomIn className="w-3.5 h-3.5" /> Zoom In
          </button>
          <button onClick={() => zoom('out')} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-xs text-foreground font-semibold hover:bg-zinc-900/50 transition-colors">
            <ZoomOut className="w-3.5 h-3.5" /> Zoom Out
          </button>
          <button onClick={resetZoom} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-xs text-foreground font-semibold hover:bg-zinc-900/50 transition-colors">
            <RotateCcw className="w-3.5 h-3.5" /> Reset Layout
          </button>
        </div>
        <div className="text-[10px] text-muted-foreground inline-flex items-center gap-2 font-mono">
          <CircleDot className="w-3 h-3 text-primary animate-pulse" /> Click points in the cloud to view structural profiles
        </div>
      </div>

      {/* Grid Canvas + Inspector Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-6 items-start">
        {/* Scatter Plot Card */}
        <div className="rounded-lg border border-glass bg-card/45 p-5 min-h-[700px] flex flex-col justify-between">
          {loading ? (
            <div className="h-[620px] flex flex-col items-center justify-center text-xs text-muted-foreground gap-2">
              <span className="w-5 h-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              Generating scatter chart...
            </div>
          ) : data.length === 0 ? (
            <EmptyState title="No plot data available" description="Run a descriptor analysis on molecular structures first." />
          ) : (
            <>
              {/* Region Counters Legend */}
              <div className="flex flex-wrap gap-4 text-xs text-muted-foreground border-b border-glass pb-4 mb-4">
                {(['Q1', 'Q2', 'Q3', 'Q4'] as const).map(q => (
                  <span key={q} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.03] border border-glass/40">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: quadrantColors[q] }} />
                    <span className="font-semibold text-foreground/80">{q}</span>
                    <span className="font-mono text-white pl-1">{stats[q]}</span>
                  </span>
                ))}
                <span className="ml-auto font-mono text-foreground/75 inline-flex items-center">
                  Total Compounds: {data.length}
                </span>
              </div>

              {/* Chart Core Viewport */}
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

        {/* Selected Compound Profile Inspector Sidebar */}
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

              {/* 3D Structure Viewer Block */}
              <div className="relative rounded-lg border border-zinc-800 bg-zinc-950 overflow-hidden aspect-video flex flex-col items-center justify-center min-h-[180px]">
                {structureLoading ? (
                  <div className="flex flex-col items-center gap-2 text-muted-foreground text-[10px] font-mono">
                    <span className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                    Generating 3D model...
                  </div>
                ) : structureData ? (
                  <div className="w-full h-full relative z-0">
                    <MoleculeViewer 
                      data={structureData} 
                      format={selected.fileName.split('.').pop()?.toLowerCase() || 'xyz'}
                      style={{ stick: { radius: 0.18, colorscheme: 'Jmol' } }}
                    />
                    <div className="absolute bottom-2 right-2 z-10 inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/60 text-[9px] text-muted-foreground border border-glass pointer-events-none">
                      <HelpCircle className="w-3 h-3" /> Click & Drag to Rotate
                    </div>
                  </div>
                ) : (
                  <div className="text-center p-4 text-[10px] text-muted-foreground">
                    <span className="font-semibold block text-white/80">3D Visualizer Offline</span>
                    <span>Coordinates not cached locally.</span>
                  </div>
                )}
              </div>

              {/* Normalized Metrics grid */}
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

              {/* 9-Dimensional Descriptor Gauges */}
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
                        {/* Micro progress indicator */}
                        <div className="w-full bg-zinc-850 h-[3px] rounded-full overflow-hidden mt-1.5">
                          <div 
                            className="h-full" 
                            style={{ 
                              width: `${ratio * 100}%`,
                              backgroundColor: quadrantColors[selected.quadrant]
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
  );
};

export default Explorer;
