import { useMemo } from 'react';

import type { PoolConfig, SequenceView, Snapshot } from '../../api/protocol';
import { formatInt, tokenSpan } from '../../lib/format';
import { blockOwners, blockState } from '../../state/ownership';
import styles from './BlockPool.module.css';

interface Props {
  config: PoolConfig;
  snapshot: Snapshot;
  /** The sequence whose block table is traced through the pool. */
  active: SequenceView | null;
  /** Blocks that received a copy-on-write in the latest step. */
  copied: ReadonlySet<number>;
}

export function BlockPool({ config, snapshot, active, copied }: Props) {
  const owners = useMemo(
    () => blockOwners(snapshot.running, config.num_blocks),
    [snapshot.running, config.num_blocks],
  );
  const { used_blocks: used, shared_blocks: shared } = snapshot.metrics;
  const summary = `${formatInt(used)} of ${formatInt(config.num_blocks)} blocks in use, ${formatInt(shared)} shared`;
  const columns = Math.ceil(Math.sqrt(config.num_blocks));

  return (
    <section className={styles.board} aria-labelledby="pool-heading">
      <div className={styles.header}>
        <h2 id="pool-heading" className={styles.heading}>
          Block pool
        </h2>
        <p className={styles.summary}>{summary}</p>
      </div>

      <div
        className={styles.grid}
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
        role="img"
        aria-label={summary}
        data-tracing={active ? '' : undefined}
      >
        {snapshot.ref_counts.map((refCount, block) => {
          const state = blockState(refCount);
          const blockOwnersHere = owners[block] ?? [];
          const traced = active ? blockOwnersHere.find((o) => o.seqId === active.id) : undefined;
          let label: string | null = null;
          if (traced) label = `${traced.position}`;
          else if (state === 'shared') label = `×${refCount}`;
          else if (state === 'owned' && blockOwnersHere[0]) label = `#${blockOwnersHere[0].seqId}`;

          return (
            <div
              key={block}
              className={styles.block}
              data-state={state}
              data-traced={traced ? '' : undefined}
              title={blockTitle(block, blockOwnersHere)}
            >
              <span className={styles.address}>{block}</span>
              {label && <span className={styles.label}>{label}</span>}
              {copied.has(block) && <span key={snapshot.tick} className={styles.flash} />}
            </div>
          );
        })}
      </div>

      <ul className={styles.legend} aria-label="Legend">
        <li>
          <span className={styles.swatch} data-state="free" /> Free
        </li>
        <li>
          <span className={styles.swatch} data-state="owned" /> Owned by one sequence
        </li>
        <li>
          <span className={styles.swatch} data-state="shared" /> Shared after a fork
        </li>
      </ul>

      <BlockTable sequence={active} blockSize={config.block_size} refCounts={snapshot.ref_counts} />
    </section>
  );
}

function blockTitle(block: number, owners: { seqId: number; position: number }[]): string {
  if (owners.length === 0) return `Block ${block}: free`;
  const list = owners.map((o) => `#${o.seqId} (position ${o.position})`).join(', ');
  return `Block ${block}: ${list}`;
}

interface BlockTableProps {
  sequence: SequenceView | null;
  blockSize: number;
  refCounts: readonly number[];
}

function BlockTable({ sequence, blockSize, refCounts }: BlockTableProps) {
  if (!sequence) {
    return (
      <p className={styles.hint}>
        Select a running sequence to trace its block table through the pool. Its blocks are numbered
        by position, showing how a contiguous sequence maps onto scattered memory.
      </p>
    );
  }
  return (
    <div className={styles.table}>
      <h3 className={styles.tableHeading}>Block table for #{sequence.id}</h3>
      <ol className={styles.entries}>
        {sequence.blocks.map((block, position) => (
          <li
            key={position}
            className={styles.entry}
            data-shared={(refCounts[block] ?? 0) > 1 ? '' : undefined}
          >
            <span className={styles.position}>{position}</span>
            <span>block {block}</span>
            <span className={styles.span}>
              tokens {tokenSpan(position, blockSize, sequence.tokens)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
