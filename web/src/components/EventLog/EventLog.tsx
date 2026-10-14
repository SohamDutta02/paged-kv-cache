import { useId, useMemo, useState } from 'react';

import type { Event, EventKind } from '../../api/protocol';
import { formatInt, systemPromptLabel } from '../../lib/format';
import styles from './EventLog.module.css';

type Filter = 'all' | 'requests' | 'hits' | 'cow' | 'preempted';

const FILTERS: { value: Filter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'requests', label: 'Requests' },
  { value: 'hits', label: 'Cache hits' },
  { value: 'cow', label: 'Forks and copies' },
  { value: 'preempted', label: 'Preemptions' },
];

const MATCHES: Record<Filter, (kind: EventKind) => boolean> = {
  all: () => true,
  requests: (k) => k.type === 'submitted' || k.type === 'admitted' || k.type === 'finished',
  hits: (k) => k.type === 'admitted' && k.cached_tokens > 0,
  cow: (k) => k.type === 'forked' || k.type === 'cow',
  preempted: (k) => k.type === 'preempted',
};

export function EventLog({ events }: { events: readonly Event[] }) {
  const id = useId();
  const [filter, setFilter] = useState<Filter>('all');
  const rows = useMemo(
    () => events.filter((e) => MATCHES[filter](e.kind)).reverse(),
    [events, filter],
  );

  return (
    <section className={styles.section} aria-labelledby={`${id}-heading`}>
      <div className={styles.header}>
        <h2 id={`${id}-heading`} className={styles.heading}>
          Events
        </h2>
        <fieldset className={styles.filters}>
          <legend className="visually-hidden">Show</legend>
          {FILTERS.map((f) => (
            <label key={f.value} className={styles.filter}>
              <input
                type="radio"
                name={`${id}-filter`}
                value={f.value}
                checked={filter === f.value}
                onChange={() => {
                  setFilter(f.value);
                }}
              />
              {f.label}
            </label>
          ))}
        </fieldset>
      </div>

      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Tick</th>
              <th scope="col">Sequence</th>
              <th scope="col">Event</th>
              <th scope="col">Details</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e, i) => (
              <tr key={`${e.tick}-${e.seq_id}-${e.kind.type}-${i}`}>
                <td className={styles.tick}>{formatInt(e.tick)}</td>
                <td className={styles.seq}>#{e.seq_id}</td>
                <td>
                  <span className={styles.kind} data-kind={e.kind.type}>
                    {KIND_LABEL[e.kind.type]}
                  </span>
                </td>
                <td className={styles.details}>{describe(e.kind)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <p className={styles.empty}>No matching events yet.</p>}
      </div>
    </section>
  );
}

const KIND_LABEL: Record<EventKind['type'], string> = {
  submitted: 'Submitted',
  forked: 'Forked',
  admitted: 'Admitted',
  cow: 'Copy-on-write',
  preempted: 'Preempted',
  finished: 'Finished',
};

function describe(kind: EventKind): string {
  switch (kind.type) {
    case 'submitted': {
      const system =
        kind.system_prompt === null
          ? ''
          : `System prompt ${systemPromptLabel(kind.system_prompt)} + `;
      const by = kind.origin === 'user' ? ', by you' : '';
      return `${system}${formatInt(kind.prompt_len)} prompt tokens, ${formatInt(kind.max_new_tokens)} to generate${by}`;
    }
    case 'forked':
      return `From #${kind.parent}, sharing all of its blocks`;
    case 'admitted': {
      const total = kind.cached_tokens + kind.computed_tokens;
      return kind.cached_tokens > 0
        ? `${formatInt(kind.cached_tokens)} of ${formatInt(total)} tokens reused from the cache`
        : `All ${formatInt(total)} tokens computed`;
    }
    case 'cow':
      return `Shared block ${kind.src_block} copied to block ${kind.dst_block}`;
    case 'preempted':
      return 'Blocks freed; back to the front of the queue';
    case 'finished':
      return 'Blocks freed';
  }
}
