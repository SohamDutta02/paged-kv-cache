import type { Totals as TotalsData } from '../../api/protocol';
import { formatInt } from '../../lib/format';
import styles from './Totals.module.css';

interface Row {
  label: string;
  value: string;
}

interface Props {
  totals: TotalsData;
  prefixCaching: boolean;
}

export function Totals({ totals, prefixCaching }: Props) {
  const prompt = totals.prompt_tokens_cached + totals.prompt_tokens_computed;
  const hitRate =
    prompt === 0 ? '–' : `${Math.round((100 * totals.prompt_tokens_cached) / prompt)}%`;

  const scheduler: Row[] = [
    { label: 'Requests submitted', value: formatInt(totals.submitted) },
    { label: 'Admissions', value: formatInt(totals.admitted) },
    { label: 'Finished', value: formatInt(totals.finished) },
    { label: 'Preemptions', value: formatInt(totals.preempted) },
    { label: 'Forks', value: formatInt(totals.forks) },
    { label: 'Copy-on-write copies', value: formatInt(totals.cow_copies) },
    { label: 'Tokens generated', value: formatInt(totals.tokens_generated) },
  ];
  const cache: Row[] = [
    { label: 'Prompt tokens computed', value: formatInt(totals.prompt_tokens_computed) },
    { label: 'Prompt tokens from cache', value: formatInt(totals.prompt_tokens_cached) },
    { label: 'Cache hit rate', value: hitRate },
    { label: 'Cache evictions', value: formatInt(totals.cache_evictions) },
  ];

  return (
    <section className={styles.section} aria-labelledby="totals-heading">
      <h2 id="totals-heading" className={styles.heading}>
        This session
      </h2>
      <List rows={scheduler} />
      {prefixCaching && (
        <>
          <h3 className={styles.subheading}>Prefix cache</h3>
          <List rows={cache} />
        </>
      )}
    </section>
  );
}

function List({ rows }: { rows: Row[] }) {
  return (
    <dl className={styles.list}>
      {rows.map(({ label, value }) => (
        <div key={label} className={styles.row}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
