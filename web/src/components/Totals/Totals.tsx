import type { Totals as TotalsData } from '../../api/protocol';
import { formatInt } from '../../lib/format';
import styles from './Totals.module.css';

const ROWS: { key: keyof TotalsData; label: string }[] = [
  { key: 'submitted', label: 'Requests submitted' },
  { key: 'admitted', label: 'Admissions' },
  { key: 'finished', label: 'Finished' },
  { key: 'preempted', label: 'Preemptions' },
  { key: 'forks', label: 'Forks' },
  { key: 'cow_copies', label: 'Copy-on-write copies' },
  { key: 'tokens_generated', label: 'Tokens generated' },
];

export function Totals({ totals }: { totals: TotalsData }) {
  return (
    <section className={styles.section} aria-labelledby="totals-heading">
      <h2 id="totals-heading" className={styles.heading}>
        This session
      </h2>
      <dl className={styles.list}>
        {ROWS.map(({ key, label }) => (
          <div key={key} className={styles.row}>
            <dt>{label}</dt>
            <dd>{formatInt(totals[key])}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
