import type { Metrics, PoolConfig, SequenceView, Snapshot } from '../api/protocol';

export const config: PoolConfig = {
  num_blocks: 16,
  block_size: 8,
  max_batch_size: 4,
  max_waiting: 8,
  tick_ms: 400,
};

export function sequence(
  id: number,
  blocks: number[],
  overrides: Partial<SequenceView> = {},
): SequenceView {
  return {
    id,
    parent: null,
    origin: 'traffic',
    tokens: blocks.length * 8 - 3,
    generated: 2,
    max_new_tokens: 10,
    blocks,
    ...overrides,
  };
}

export function metrics(tick: number, overrides: Partial<Metrics> = {}): Metrics {
  return { tick, used_blocks: 0, shared_blocks: 0, running: 0, waiting: 0, ...overrides };
}

/** A snapshot whose ref counts are derived from `running`, like the server's. */
export function snapshot(tick: number, running: SequenceView[] = []): Snapshot {
  const refCounts = new Array<number>(config.num_blocks).fill(0);
  for (const seq of running) for (const b of seq.blocks) refCounts[b] = (refCounts[b] ?? 0) + 1;
  return {
    tick,
    paused: false,
    traffic: true,
    ref_counts: refCounts,
    running,
    waiting: [],
    metrics: metrics(tick, {
      used_blocks: refCounts.filter((rc) => rc > 0).length,
      shared_blocks: refCounts.filter((rc) => rc > 1).length,
      running: running.length,
    }),
    totals: {
      submitted: 0,
      admitted: 0,
      finished: 0,
      preempted: 0,
      forks: 0,
      cow_copies: 0,
      tokens_generated: 0,
    },
  };
}
