import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunConfig } from '@/types';
import { ApiError, getJob } from './api';
import { hydrateRun, type RunRecord } from './runs';

vi.mock('./api', async importOriginal => {
  const actual = await importOriginal<typeof import('./api')>();
  return {
    ...actual,
    getJob: vi.fn(),
  };
});

const config = {} as RunConfig;

describe('hydrateRun', () => {
  beforeEach(() => {
    vi.mocked(getJob).mockReset();
  });

  it('marks jobs lost across a backend restart as terminal failures', async () => {
    vi.mocked(getJob).mockRejectedValue(new ApiError('Job not found', 404));
    const record: RunRecord = {
      id: 'run-stale',
      name: 'Stale run',
      createdAt: '2026-07-29T00:00:00Z',
      jobIds: ['missing-1', 'missing-2'],
      config,
    };

    const run = await hydrateRun(record);

    expect(run.status).toBe('failed');
    expect(run.totalFiles).toBe(2);
    expect(run.completedFiles).toBe(2);
    expect(run.failedFiles).toBe(2);
  });

  it('keeps transient API failures retryable', async () => {
    vi.mocked(getJob).mockRejectedValue(new TypeError('fetch failed'));
    const record: RunRecord = {
      id: 'run-offline',
      name: 'Temporarily unavailable run',
      createdAt: '2026-07-29T00:00:00Z',
      jobIds: ['unavailable-1'],
      config,
    };

    const run = await hydrateRun(record);

    expect(run.status).toBe('processing');
    expect(run.completedFiles).toBe(0);
    expect(run.failedFiles).toBe(0);
  });
});
