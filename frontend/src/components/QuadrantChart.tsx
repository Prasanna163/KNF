import { useMemo, useState } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Label,
} from 'recharts';
import type { ResultRecord, QuadrantData } from '@/types';
import { cn } from '@/lib/utils';

const quadrantColors: Record<string, string> = {
  Q1: '#22d3ee', // cyan
  Q2: '#a78bfa', // violet
  Q3: '#fb923c', // orange
  Q4: '#4ade80', // green
};

interface QuadrantChartProps {
  results: ResultRecord[];
  quadrantData: QuadrantData;
  compact?: boolean;
}

export function QuadrantChart({ results, quadrantData, compact = false }: QuadrantChartProps) {
  const [showLabels, setShowLabels] = useState(true);
  const [showMedians, setShowMedians] = useState(true);

  const data = useMemo(() =>
    results
      .filter(r => r.status === 'success' && r.SNCI_Norm !== null && r.SCDI_Norm !== null)
      .map(r => ({
      x: r.SNCI_Norm as number,
      y: r.SCDI_Norm as number,
      fileName: r.fileName,
      quadrant: r.quadrant,
      SNCI: r.SNCI,
      SCDI: r.SCDI,
    })),
    [results]
  );

  const grouped = useMemo(() => {
    const g: Record<string, typeof data> = { Q1: [], Q2: [], Q3: [], Q4: [] };
    data.forEach(d => g[d.quadrant].push(d));
    return g;
  }, [data]);

  const height = compact ? 280 : 450;

  return (
    <div className="space-y-3">
      {!compact && (
        <div className="flex items-center gap-4 text-xs">
          <label className="flex items-center gap-1.5 text-muted-foreground cursor-pointer">
            <input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} className="rounded" />
            Quadrant labels
          </label>
          <label className="flex items-center gap-1.5 text-muted-foreground cursor-pointer">
            <input type="checkbox" checked={showMedians} onChange={e => setShowMedians(e.target.checked)} className="rounded" />
            Median lines
          </label>
          <span className="ml-auto text-muted-foreground">
            Median SNCI: <span className="font-mono text-foreground">{quadrantData.medianSNCI}</span> | 
            Median SCDI: <span className="font-mono text-foreground">{quadrantData.medianSCDI}</span>
          </span>
        </div>
      )}
      <div className="rounded-xl border border-border bg-card p-4" role="img" aria-label={`Quadrant scatter chart showing ${data.length} molecules. Q1: ${quadrantData.counts.Q1}, Q2: ${quadrantData.counts.Q2}, Q3: ${quadrantData.counts.Q3}, Q4: ${quadrantData.counts.Q4}`}>
        <ResponsiveContainer width="100%" height={height}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 25, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 20%, 16%)" />
            <XAxis
              type="number" dataKey="x" domain={[0, 1]} tickCount={6}
              stroke="hsl(215, 15%, 50%)" fontSize={11} fontFamily="JetBrains Mono"
            >
              <Label value="SNCI_Norm" position="bottom" offset={10} style={{ fill: 'hsl(215, 15%, 50%)', fontSize: 12, fontFamily: 'Space Grotesk' }} />
            </XAxis>
            <YAxis
              type="number" dataKey="y" domain={[0, 1]} tickCount={6}
              stroke="hsl(215, 15%, 50%)" fontSize={11} fontFamily="JetBrains Mono"
            >
              <Label value="SCDI Batch Norm" angle={-90} position="insideLeft" offset={5} style={{ fill: 'hsl(215, 15%, 50%)', fontSize: 12, fontFamily: 'Space Grotesk' }} />
            </YAxis>
            {showMedians && (
              <>
                <ReferenceLine x={quadrantData.medianSNCI} stroke="hsl(215, 15%, 35%)" strokeDasharray="6 4" />
                <ReferenceLine y={quadrantData.medianSCDI} stroke="hsl(215, 15%, 35%)" strokeDasharray="6 4" />
              </>
            )}
            <Tooltip
              content={({ payload }) => {
                if (!payload?.[0]) return null;
                const d = payload[0].payload;
                return (
                  <div className="rounded-lg border border-border bg-popover p-3 text-xs shadow-lg">
                    <p className="font-medium text-foreground mb-1">{d.fileName}</p>
                    <p className="text-muted-foreground">SNCI_Norm: <span className="font-mono text-foreground">{d.x.toFixed(6)}</span></p>
                    <p className="text-muted-foreground">SCDI_Norm: <span className="font-mono text-foreground">{d.y.toFixed(6)}</span></p>
                    <p className="text-muted-foreground">Quadrant: <span className="font-mono" style={{ color: quadrantColors[d.quadrant] }}>{d.quadrant}</span></p>
                  </div>
                );
              }}
            />
            {Object.entries(grouped).map(([q, points]) => (
              <Scatter key={q} name={q} data={points} fill={quadrantColors[q]} fillOpacity={0.7} r={compact ? 4 : 6} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
        {showLabels && !compact && (
          <div className="flex justify-center gap-6 mt-2 text-xs">
            {(['Q1', 'Q2', 'Q3', 'Q4'] as const).map(q => (
              <span key={q} className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: quadrantColors[q] }} />
                <span className="text-muted-foreground">{q}: <span className="font-mono text-foreground">{quadrantData.counts[q]}</span></span>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
