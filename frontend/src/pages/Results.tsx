import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowDown, ArrowUp, ArrowUpDown, Search, Columns, Sparkles, X, ChevronLeft, ChevronRight, Download, HelpCircle } from 'lucide-react';
import { EmptyState } from '@/components/shared/EmptyState';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { MoleculeViewer } from '@/components/MoleculeViewer';
import type { ResultRecord, Quadrant } from '@/types';
import { listJobs, getInputContent } from '@/lib/api';
import { getRunIdForJob } from '@/lib/runs';
import { buildResultRecords } from '@/lib/descriptors';

type ColKey = keyof ResultRecord;

const columnMeta: { key: ColKey; label: string; numeric: boolean }[] = [
  { key: 'fileName', label: 'Molecular File', numeric: false },
  { key: 'SNCI', label: 'SNCI', numeric: true },
  { key: 'SCDI_variance', label: 'SCDI Var', numeric: true },
  { key: 'SNCI_Norm', label: 'SNCI Norm', numeric: true },
  { key: 'SCDI_Norm', label: 'SCDI Batch Norm', numeric: true },
];

const descriptorKeys: ('f1' | 'f2' | 'f3' | 'f4' | 'f5' | 'f6' | 'f7' | 'f8' | 'f9')[] = [
  'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'
];

const quadrantPills: { key: Quadrant; label: string; bg: string; border: string; text: string }[] = [
  { key: 'Q1', label: 'Quadrant 1 (Sky)', bg: 'bg-[#0ea5e9]/10', border: 'border-[#0ea5e9]/30', text: 'text-[#0ea5e9]' },
  { key: 'Q2', label: 'Quadrant 2 (Indigo)', bg: 'bg-[#6366f1]/10', border: 'border-[#6366f1]/30', text: 'text-[#6366f1]' },
  { key: 'Q3', label: 'Quadrant 3 (Amber)', bg: 'bg-[#f59e0b]/10', border: 'border-[#f59e0b]/30', text: 'text-[#f59e0b]' },
  { key: 'Q4', label: 'Quadrant 4 (Emerald)', bg: 'bg-[#10b981]/10', border: 'border-[#10b981]/30', text: 'text-[#10b981]' },
];

const quadrantColors: Record<Quadrant, string> = {
  Q1: 'bg-[#0ea5e9]/15 text-[#0ea5e9] border border-[#0ea5e9]/20',
  Q2: 'bg-[#6366f1]/15 text-[#6366f1] border border-[#6366f1]/20',
  Q3: 'bg-[#f59e0b]/15 text-[#f59e0b] border border-[#f59e0b]/20',
  Q4: 'bg-[#10b981]/15 text-[#10b981] border border-[#10b981]/20',
};

function cmp(a: unknown, b: unknown): number {
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b));
}

const Results = () => {
  const [results, setResults] = useState<ResultRecord[]>([]);
  const [search, setSearch] = useState('');
  const [selectedQuad, setSelectedQuad] = useState<Quadrant | null>(null);
  const [viewMode, setViewMode] = useState<'profile' | 'raw'>('profile');
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<ColKey | null>('fileName');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [active, setActive] = useState(false);
  const [selectedRow, setSelectedRow] = useState<ResultRecord | null>(null);
  const [structureData, setStructureData] = useState<string | null>(null);
  const [structureLoading, setStructureLoading] = useState(false);

  // Pagination states
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(15);

  const fetchResults = useCallback(async (mounted: boolean) => {
    try {
      const jobs = await listJobs();
      if (!mounted) return;
      const list = buildResultRecords(jobs, getRunIdForJob);
      setResults(list);
      setActive(list.some(r => r.status === 'running' || r.status === 'queued'));
    } catch {
      if (mounted) setResults([]);
    }
  }, []);

  useEffect(() => {
    const mounted = true;
    (async () => {
      await fetchResults(mounted);
      if (mounted) setLoading(false);
    })();
  }, [fetchResults]);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => fetchResults(true), 3000);
    return () => clearInterval(id);
  }, [active, fetchResults]);

  // Fetch 3D Structure content
  useEffect(() => {
    if (!selectedRow) {
      setStructureData(null);
      return;
    }
    let activeFetch = true;
    setStructureLoading(true);
    setStructureData(null);

    getInputContent(selectedRow.id)
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
  }, [selectedRow]);

  const handleSort = useCallback(
    (key: ColKey) => {
      setSortDir(prev => (sortKey === key && prev === 'asc' ? 'desc' : 'asc'));
      setSortKey(key);
    },
    [sortKey],
  );

  const filtered = useMemo(() => {
    let list = results;
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(r => r.fileName.toLowerCase().includes(q) || r.runId.toLowerCase().includes(q));
    }
    if (selectedQuad) {
      list = list.filter(r => r.quadrant === selectedQuad);
    }
    return list;
  }, [results, search, selectedQuad]);

  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    return [...filtered].sort((a, b) => {
      const result = cmp(a[sortKey], b[sortKey]);
      return sortDir === 'asc' ? result : -result;
    });
  }, [filtered, sortKey, sortDir]);

  // Pagination calculations
  const totalPages = useMemo(() => Math.ceil(sorted.length / pageSize) || 1, [sorted.length, pageSize]);

  useEffect(() => {
    setPage(1);
  }, [search, selectedQuad]);

  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, totalPages]);

  const paginatedItems = useMemo(() => {
    const activePage = Math.min(page, totalPages);
    return sorted.slice((activePage - 1) * pageSize, activePage * pageSize);
  }, [sorted, page, pageSize, totalPages]);

  const getPageNumbers = () => {
    const pages: (number | string)[] = [];
    if (totalPages <= 5) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);
      if (page > 3) {
        pages.push('...');
      }
      const start = Math.max(2, page - 1);
      const end = Math.min(totalPages - 1, page + 1);
      for (let i = start; i <= end; i++) {
        if (!pages.includes(i)) pages.push(i);
      }
      if (page < totalPages - 2) {
        pages.push('...');
      }
      if (!pages.includes(totalPages)) pages.push(totalPages);
    }
    return pages;
  };

  // Max visual value for normalizing descriptor micro-bars
  const globalMaxDescriptor = useMemo(() => {
    let maxVal = 0.1;
    results.forEach(r => {
      descriptorKeys.forEach(k => {
        const val = r[k];
        if (typeof val === 'number' && val > maxVal) {
          maxVal = val;
        }
      });
    });
    return maxVal;
  }, [results]);

  const renderCell = (col: (typeof columnMeta)[number], r: ResultRecord) => {
    const val = r[col.key];
    if (val === null || val === undefined) return '-';
    if (col.numeric && typeof val === 'number') return val.toFixed(6);
    return String(val);
  };

  const downloadCSV = () => {
    const csvValue = (value: unknown) => typeof value === 'number' ? value.toFixed(6) : (value ?? '');
    const headers = ['fileName', 'runId', 'SNCI', 'SCDI_variance', 'SNCI_Norm', 'SCDI_Norm', 'quadrant', ...descriptorKeys];
    const csvContent = [
      headers.join(','),
      ...results.map(r => [
        r.fileName,
        r.runId,
        csvValue(r.SNCI),
        csvValue(r.SCDI_variance),
        csvValue(r.SNCI_Norm),
        csvValue(r.SCDI_Norm),
        r.quadrant,
        ...descriptorKeys.map(k => csvValue(r[k]))
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `knf_results_library_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-[1600px] animate-fade-in-up relative min-h-screen">
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-display font-bold text-foreground tracking-tight">
            Results <span className="text-gradient">Library</span>
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Spreadsheet ledger and structural inspection workspace</p>
        </div>
        <div className="flex items-center gap-2">
          {active && (
            <span className="inline-flex items-center gap-1.5 text-xs text-primary font-medium px-2.5 py-1 rounded-full bg-primary/10 border border-primary/20 mr-2">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              Syncing
            </span>
          )}
          <button
            onClick={downloadCSV}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-glass bg-white/5 hover:bg-white/10 text-xs text-foreground font-medium transition-colors"
          >
            <Download className="w-3.5 h-3.5" /> Export CSV
          </button>
        </div>
      </header>

      {/* Filter and View Toggles Bar */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 p-4 rounded-lg border border-glass bg-card/40">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 flex-1 min-w-0">
          <div className="relative w-full sm:max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search molecular files..."
              className="w-full rounded-lg bg-background/50 border border-glass pl-9 pr-3 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus:ring-1 focus:ring-ring outline-none"
            />
          </div>
          {/* Quadrant filter buttons */}
          <div className="flex flex-wrap gap-1.5">
            {quadrantPills.map(p => {
              const selected = selectedQuad === p.key;
              return (
                <button
                  key={p.key}
                  onClick={() => setSelectedQuad(selected ? null : p.key)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all ${
                    selected 
                      ? `${p.bg} ${p.border} ${p.text}`
                      : 'border-zinc-800 bg-zinc-900/20 hover:bg-zinc-900/50 text-muted-foreground'
                  }`}
                >
                  {p.key}
                </button>
              );
            })}
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setViewMode('profile')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              viewMode === 'profile'
                ? 'bg-primary/10 border-primary/20 text-primary'
                : 'bg-zinc-900/20 border-zinc-800 text-muted-foreground hover:bg-zinc-900/50'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" /> Shape Profiles
          </button>
          <button
            onClick={() => setViewMode('raw')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              viewMode === 'raw'
                ? 'bg-primary/10 border-primary/20 text-primary'
                : 'bg-zinc-900/20 border-zinc-800 text-muted-foreground hover:bg-zinc-900/50'
            }`}
          >
            <Columns className="w-3.5 h-3.5" /> Numeric Grid
          </button>
        </div>
      </div>

      {/* Main Content Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-6 items-start">
        {/* Table Column */}
        <div className="rounded-lg border border-glass bg-card/45 overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-xs text-muted-foreground">
              <span className="inline-flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse" />
                Retrieving descriptor calculations...
              </span>
            </div>
          ) : sorted.length === 0 ? (
            <div className="p-8">
              <EmptyState title="No calculation records found" description="Start a descriptor run from the Run Manager to generate results." />
            </div>
          ) : (
            <>
              <div className="overflow-x-auto max-h-[70vh]">
                <Table>
                <TableHeader className="sticky top-0 bg-zinc-950 z-10 border-b border-glass">
                  <TableRow className="border-b border-glass hover:bg-transparent">
                    {columnMeta.map(col => (
                      <TableHead
                        key={col.key}
                        className={`whitespace-nowrap cursor-pointer select-none py-3 px-4 hover:text-foreground transition-colors font-semibold ${col.numeric ? 'text-right' : 'text-left'}`}
                        onClick={() => handleSort(col.key)}
                      >
                        <span className="inline-flex items-center gap-1">
                          {col.label}
                          {sortKey === col.key ? (
                            sortDir === 'asc' ? <ArrowUp className="w-3 h-3 text-primary" /> : <ArrowDown className="w-3 h-3 text-primary" />
                          ) : (
                            <ArrowUpDown className="w-3 h-3 opacity-20" />
                          )}
                        </span>
                      </TableHead>
                    ))}

                    {/* Conditional Headers */}
                    {viewMode === 'raw' ? (
                      descriptorKeys.map(k => (
                        <TableHead
                          key={k}
                          className="whitespace-nowrap cursor-pointer select-none text-right font-semibold hover:text-foreground font-mono"
                          onClick={() => handleSort(k)}
                        >
                          <span className="inline-flex items-center gap-1">
                            {k}
                            {sortKey === k ? (
                              sortDir === 'asc' ? <ArrowUp className="w-3.5 h-3.5 text-primary" /> : <ArrowDown className="w-3.5 h-3.5 text-primary" />
                            ) : (
                              <ArrowUpDown className="w-3 h-3 opacity-20" />
                            )}
                          </span>
                        </TableHead>
                      ))
                    ) : (
                      <TableHead className="text-left font-semibold">KNF Shape Fingerprint (f1–f9)</TableHead>
                    )}

                    <TableHead className="text-center font-semibold">Region</TableHead>
                    <TableHead className="w-10"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedItems.map(r => {
                    const isSelected = selectedRow?.id === r.id;
                    return (
                      <TableRow
                        key={r.id}
                        onClick={() => setSelectedRow(isSelected ? null : r)}
                        className={`cursor-pointer border-b border-glass/40 transition-colors hover:bg-white/5 ${
                          isSelected ? 'bg-primary/10 hover:bg-primary/15' : ''
                        }`}
                      >
                        {columnMeta.map(col => (
                          <TableCell
                            key={col.key}
                            className={`font-mono text-xs py-3 px-4 ${col.numeric ? 'text-right text-muted-foreground' : 'font-medium text-foreground'} max-w-[200px] truncate`}
                          >
                            {renderCell(col, r)}
                          </TableCell>
                        ))}

                        {/* Conditional Columns */}
                        {viewMode === 'raw' ? (
                          descriptorKeys.map(k => (
                            <TableCell key={k} className="text-right text-xs font-mono text-muted-foreground px-4">
                              {typeof r[k] === 'number' ? r[k].toFixed(6) : '-'}
                            </TableCell>
                          ))
                        ) : (
                          <TableCell className="py-3 px-4 min-w-[200px]">
                            {/* Graphic sparkline descriptor profile */}
                            <div className="flex items-end gap-[3px] h-6 w-36 bg-zinc-950 p-1 rounded border border-zinc-800">
                              {descriptorKeys.map(k => {
                                const val = r[k];
                                const ratio = typeof val === 'number' ? val / globalMaxDescriptor : 0;
                                const heightPercent = Math.min(100, Math.max(10, ratio * 100));
                                return (
                                  <div
                                    key={k}
                                    title={`${k}: ${typeof val === 'number' ? val.toFixed(6) : '-'}`}
                                    className="flex-1 rounded-t-sm transition-all duration-300"
                                    style={{
                                      height: `${heightPercent}%`,
                                      backgroundColor: r.quadrant === 'Q1' ? '#0ea5e9' : r.quadrant === 'Q2' ? '#6366f1' : r.quadrant === 'Q3' ? '#f59e0b' : '#10b981'
                                    }}
                                  />
                                );
                              })}
                            </div>
                          </TableCell>
                        )}

                        <TableCell className="text-center py-3 px-4">
                          <span
                            className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold leading-5 ${quadrantColors[r.quadrant]}`}
                          >
                            {r.quadrant}
                          </span>
                        </TableCell>
                        <TableCell className="text-right pr-4">
                          <ChevronRight className={`w-4 h-4 text-muted-foreground transition-transform duration-200 ${isSelected ? 'rotate-90 text-primary' : ''}`} />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>

            {/* Pagination Controls */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 border-t border-glass bg-zinc-950/10 text-xs">
              <div className="flex flex-wrap items-center gap-4 text-muted-foreground">
                <span>
                  Showing <span className="font-semibold text-foreground">{sorted.length === 0 ? 0 : (page - 1) * pageSize + 1}</span>–
                  <span className="font-semibold text-foreground">{Math.min(page * pageSize, sorted.length)}</span> of{' '}
                  <span className="font-semibold text-foreground">{sorted.length}</span> entries
                </span>
                <div className="flex items-center gap-2">
                  <span>Show:</span>
                  <select
                    value={pageSize}
                    onChange={(e) => {
                      setPageSize(Number(e.target.value));
                      setPage(1);
                    }}
                    className="glass-input rounded-lg text-foreground text-xs px-2 py-1 cursor-pointer outline-none"
                  >
                    {[10, 15, 25, 50, 100].map((size) => (
                      <option key={size} value={size} className="bg-zinc-950 text-foreground">
                        {size} entries
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  onClick={() => setPage(1)}
                  disabled={page === 1}
                  className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-muted-foreground hover:bg-zinc-900/50 hover:text-foreground transition-all disabled:opacity-40 disabled:hover:bg-zinc-900/20 disabled:hover:text-muted-foreground disabled:cursor-not-allowed"
                  title="First Page"
                >
                  <div className="flex items-center">
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <ChevronLeft className="w-3.5 h-3.5 -ml-2" />
                  </div>
                </button>
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-muted-foreground hover:bg-zinc-900/50 hover:text-foreground transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-3.5 h-3.5" /> Prev
                </button>

                <div className="flex items-center gap-1 mx-1.5">
                  {getPageNumbers().map((pageNum, idx) => {
                    if (pageNum === '...') {
                      return (
                        <span key={`ell-${idx}`} className="px-2 py-1 text-muted-foreground font-mono">
                          ...
                        </span>
                      );
                    }
                    const isCurrent = pageNum === page;
                    return (
                      <button
                        key={pageNum}
                        onClick={() => setPage(Number(pageNum))}
                        className={`w-8 h-8 flex items-center justify-center rounded-lg text-xs font-semibold border transition-all ${
                          isCurrent
                            ? 'bg-primary text-primary-foreground border-primary shadow-sm'
                            : 'border-zinc-800 bg-zinc-900/20 text-muted-foreground hover:bg-zinc-900/50 hover:text-foreground'
                        }`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                </div>

                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-muted-foreground hover:bg-zinc-900/50 hover:text-foreground transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Next <ChevronRight className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setPage(totalPages)}
                  disabled={page === totalPages}
                  className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900/20 text-muted-foreground hover:bg-zinc-900/50 hover:text-foreground transition-all disabled:opacity-40 disabled:hover:bg-zinc-900/20 disabled:hover:text-muted-foreground disabled:cursor-not-allowed"
                  title="Last Page"
                >
                  <div className="flex items-center">
                    <ChevronRight className="w-3.5 h-3.5" />
                    <ChevronRight className="w-3.5 h-3.5 -ml-2" />
                  </div>
                </button>
              </div>
            </div>
          </>
        )}
        </div>

        {/* Dynamic Detail slide-out panel */}
        {selectedRow && (
          <aside className="w-full lg:w-[420px] rounded-lg border border-glass bg-card/45 p-5 space-y-6 animate-scale-in max-h-[80vh] overflow-y-auto sticky top-8">
            <div className="flex items-start justify-between gap-4 border-b border-glass pb-4">
              <div className="min-w-0">
                <span className="text-[9px] text-primary font-mono tracking-widest uppercase">Inspect Compound</span>
                <h2 className="text-xl font-display font-bold text-foreground truncate mt-0.5" title={selectedRow.fileName}>
                  {selectedRow.fileName}
                </h2>
                <p className="text-xs text-muted-foreground font-mono mt-0.5" title={selectedRow.runId}>
                  Run: {selectedRow.runId}
                </p>
              </div>
              <button
                onClick={() => setSelectedRow(null)}
                className="p-1 rounded-lg border border-zinc-800 bg-zinc-900/20 hover:bg-zinc-900/50 text-muted-foreground hover:text-foreground transition-colors shrink-0"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* 3D Molecule Viewport Area */}
            <div className="relative rounded-lg border border-zinc-800 bg-zinc-950 overflow-hidden aspect-video flex flex-col items-center justify-center min-h-[200px]">
              {structureLoading ? (
                <div className="flex flex-col items-center gap-2 text-muted-foreground text-[10px] font-mono">
                  <span className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                  Generating 3D model...
                </div>
              ) : structureData ? (
                <div className="w-full h-full relative z-0">
                  <MoleculeViewer 
                    data={structureData} 
                    format={selectedRow.fileName.split('.').pop()?.toLowerCase() || 'xyz'} 
                    style={{ stick: { radius: 0.18, colorscheme: 'Jmol' } }}
                  />
                  <div className="absolute bottom-2 right-2 z-10 inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/60 text-[9px] text-muted-foreground border border-glass pointer-events-none">
                    <HelpCircle className="w-3 h-3" /> Click & Drag to Rotate
                  </div>
                </div>
              ) : (
                <div className="text-center p-4 text-[10px] text-muted-foreground space-y-1">
                  <span className="font-semibold block text-white/80">3D Viewport Offline</span>
                  <span>Calculated values only. original file coordinate data not on local storage.</span>
                </div>
              )}
            </div>

            {/* Metrics Breakdown Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/10 p-3">
                <span className="text-[9px] text-muted-foreground uppercase font-semibold">SNCI Norm</span>
                <p className="text-lg font-mono font-bold text-foreground mt-0.5">{selectedRow.SNCI_Norm?.toFixed(6) ?? '—'}</p>
              </div>
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/10 p-3">
                <span className="text-[9px] text-muted-foreground uppercase font-semibold">SCDI Batch Norm</span>
                <p className="text-lg font-mono font-bold text-foreground mt-0.5">{selectedRow.SCDI_Norm?.toFixed(6) ?? '—'}</p>
              </div>
              <div className="col-span-2 rounded-lg border border-zinc-800 bg-zinc-900/10 p-3 flex items-center justify-between">
                <div>
                  <span className="text-[9px] text-muted-foreground uppercase font-semibold block">Interaction Fingerprint (KUID)</span>
                  <span className="text-sm font-mono text-white tracking-wide block mt-0.5">{selectedRow.KUID || 'N/A'}</span>
                </div>
                {selectedRow.KUID_Cluster && (
                  <span className="text-[9px] font-mono font-semibold px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20 shrink-0">
                    Cluster {selectedRow.KUID_Cluster}
                  </span>
                )}
              </div>
            </div>

            {/* Numerical Descriptors list (f1-f9) */}
            <div className="space-y-3 border-t border-glass pt-4">
              <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">Descriptor Feature Vector</h3>
              <div className="grid grid-cols-3 gap-2">
                {descriptorKeys.map(k => {
                  const val = selectedRow[k];
                  const ratio = typeof val === 'number' ? val / globalMaxDescriptor : 0;
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
                            backgroundColor: selectedRow.quadrant === 'Q1' ? '#0ea5e9' : selectedRow.quadrant === 'Q2' ? '#6366f1' : selectedRow.quadrant === 'Q3' ? '#f59e0b' : '#10b981'
                          }} 
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
};

export default Results;
