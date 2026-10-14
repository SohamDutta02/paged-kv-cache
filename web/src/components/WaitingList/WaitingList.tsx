import type { WaitingView } from '../../api/protocol';
import { formatInt, plural } from '../../lib/format';
import styles from './WaitingList.module.css';

interface Props {
  waiting: readonly WaitingView[];
  maxWaiting: number;
}

export function WaitingList({ waiting, maxWaiting }: Props) {
  return (
    <section className={styles.section} aria-labelledby="waiting-heading">
      <div className={styles.header}>
        <h2 id="waiting-heading" className={styles.heading}>
          Waiting
        </h2>
        <span className={styles.count}>
          {formatInt(waiting.length)} of {formatInt(maxWaiting)}
        </span>
      </div>
      {waiting.length === 0 ? (
        <p className={styles.empty}>The queue is empty. Requests are admitted front first.</p>
      ) : (
        <ol className={styles.queue} aria-label="Queue, front first">
          {waiting.map((w) => (
            <li key={w.id} className={styles.item}>
              <span className={styles.id}>#{w.id}</span>
              <span className={styles.detail}>{plural(w.context_len, 'token')} to prefill</span>
              {w.preempted && <span className={styles.tag}>preempted</span>}
              {w.parent !== null && <span className={styles.detail}>fork of #{w.parent}</span>}
              {w.origin === 'user' && <span className={styles.yours}>yours</span>}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
