import { useState } from 'react';

import type { CommandResult, SequenceView } from '../../api/protocol';
import { formatInt, systemPromptLabel } from '../../lib/format';
import { blockState } from '../../state/ownership';
import styles from './RunningTable.module.css';

const MAX_BLOCK_CHIPS = 8;

interface Props {
  running: readonly SequenceView[];
  maxBatchSize: number;
  refCounts: readonly number[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  onHover: (id: number | null) => void;
  onFork: (id: number) => Promise<CommandResult>;
}

export function RunningTable({
  running,
  maxBatchSize,
  refCounts,
  selectedId,
  onSelect,
  onHover,
  onFork,
}: Props) {
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);
  const batchFull = running.length >= maxBatchSize;

  const fork = async (id: number) => {
    const result = await onFork(id);
    setStatus(
      result.status === 'ok'
        ? { ok: true, text: `Forked #${id} into #${result.seq_id ?? '?'}.` }
        : { ok: false, text: result.message },
    );
  };

  return (
    <section className={styles.section} aria-labelledby="running-heading">
      <div className={styles.header}>
        <h2 id="running-heading" className={styles.heading}>
          Running
        </h2>
        <span className={styles.count}>
          {running.length} of {maxBatchSize}
        </span>
      </div>

      {running.length === 0 ? (
        <p className={styles.empty}>Nothing is running. Add a request or turn on traffic.</p>
      ) : (
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Sequence</th>
              <th scope="col">Generated</th>
              <th scope="col">Blocks</th>
              <th scope="col">
                <span className="visually-hidden">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {running.map((seq) => (
              <tr
                key={seq.id}
                data-selected={seq.id === selectedId ? '' : undefined}
                onMouseEnter={() => {
                  onHover(seq.id);
                }}
                onMouseLeave={() => {
                  onHover(null);
                }}
              >
                <td>
                  <button
                    type="button"
                    className={styles.select}
                    aria-pressed={seq.id === selectedId}
                    onClick={() => {
                      onSelect(seq.id === selectedId ? null : seq.id);
                    }}
                  >
                    #{seq.id}
                  </button>
                  <Lineage seq={seq} />
                </td>
                <td>
                  <Progress generated={seq.generated} total={seq.max_new_tokens} />
                </td>
                <td>
                  <BlockChips blocks={seq.blocks} refCounts={refCounts} />
                </td>
                <td className={styles.action}>
                  <button
                    type="button"
                    className={styles.fork}
                    disabled={batchFull}
                    title={batchFull ? 'The batch is full' : `Fork #${seq.id}`}
                    aria-label={`Fork #${seq.id}`}
                    onClick={() => void fork(seq.id)}
                  >
                    Fork
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <p className={styles.status} data-ok={status?.ok ? '' : undefined} aria-live="polite">
        {status?.text}
      </p>
    </section>
  );
}

function Lineage({ seq }: { seq: SequenceView }) {
  const tags = [
    seq.system_prompt !== null && (
      <span key="system">prompt {systemPromptLabel(seq.system_prompt)}</span>
    ),
    seq.cached_tokens > 0 && (
      <span key="cached" className={styles.cached}>
        {formatInt(seq.cached_tokens)} cached
      </span>
    ),
    seq.parent !== null && <span key="parent">fork of #{seq.parent}</span>,
    seq.origin === 'user' && (
      <span key="yours" className={styles.yours}>
        yours
      </span>
    ),
  ].filter(Boolean);
  if (tags.length === 0) return null;
  return <span className={styles.lineage}>{tags}</span>;
}

function Progress({ generated, total }: { generated: number; total: number }) {
  return (
    <span className={styles.progress}>
      <progress className={styles.bar} value={generated} max={total} />
      <span className={styles.progressText}>
        {formatInt(generated)}/{formatInt(total)}
      </span>
    </span>
  );
}

function BlockChips({ blocks, refCounts }: { blocks: number[]; refCounts: readonly number[] }) {
  const shown = blocks.slice(0, MAX_BLOCK_CHIPS);
  const hidden = blocks.length - shown.length;
  return (
    <span className={styles.chips}>
      {shown.map((block, position) => (
        <span key={position} className={styles.chip} data-state={blockState(refCounts[block] ?? 0)}>
          {block}
        </span>
      ))}
      {hidden > 0 && <span className={styles.more}>+{hidden}</span>}
    </span>
  );
}
