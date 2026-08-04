import { cn } from '@/lib/utils';
import type { RunStatus, JobStatus } from '@/types';

type StatusType = RunStatus | JobStatus;

const statusConfig: Record<string, { label: string; className: string }> = {
  idle: { label: 'Idle', className: 'bg-muted text-muted-foreground' },
  validating: { label: 'Validating', className: 'bg-info/10 text-info' },
  queued: { label: 'Queued', className: 'bg-muted text-muted-foreground' },
  processing: { label: 'Processing', className: 'bg-primary/10 text-primary animate-pulse-glow' },
  running: { label: 'Running', className: 'bg-primary/10 text-primary animate-pulse-glow' },
  stop_requested: { label: 'Stopping', className: 'bg-warning/10 text-warning' },
  finalizing: { label: 'Finalizing', className: 'bg-info/10 text-info' },
  completed: { label: 'Completed', className: 'bg-success/10 text-success' },
  success: { label: 'Success', className: 'bg-success/10 text-success' },
  failed: { label: 'Failed', className: 'bg-destructive/10 text-destructive' },
  stopped: { label: 'Stopped', className: 'bg-warning/10 text-warning' },
};

export function StatusBadge({ status }: { status: StatusType }) {
  const config = statusConfig[status] || { label: status, className: 'bg-muted text-muted-foreground' };
  return (
    <span className={cn('inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium', config.className)}>
      {(status === 'processing' || status === 'running') && (
        <span className="w-1.5 h-1.5 rounded-full bg-current" />
      )}
      {config.label}
    </span>
  );
}
