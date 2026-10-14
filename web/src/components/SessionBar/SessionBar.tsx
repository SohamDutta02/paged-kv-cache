import { useEffect, useId, useState } from 'react';

import type { Command, CommandResult, Snapshot } from '../../api/protocol';
import { formatInt } from '../../lib/format';
import type { Connection } from '../../state/simulation';
import styles from './SessionBar.module.css';

interface Props {
  connection: Connection;
  snapshot: Snapshot | null;
  send: (command: Command) => Promise<CommandResult>;
  reconnect: () => void;
}

export function SessionBar({ connection, snapshot, send, reconnect }: Props) {
  const live = connection.status === 'open' && snapshot !== null;
  const restartHintId = useId();
  const paused = snapshot?.paused ?? false;

  return (
    <div className={styles.bar}>
      <Status connection={connection} snapshot={snapshot} reconnect={reconnect} />
      <div className={styles.controls} role="group" aria-label="Simulation controls">
        <button
          type="button"
          className={styles.button}
          disabled={!live}
          onClick={() => void send({ type: paused ? 'resume' : 'pause' })}
        >
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button
          type="button"
          className={styles.button}
          disabled={!live || !paused}
          title={paused ? 'Advance one tick' : 'Pause to step one tick at a time'}
          onClick={() => void send({ type: 'step' })}
        >
          Step
        </button>
        <button
          type="button"
          className={styles.button}
          disabled={!live}
          onClick={() => void send({ type: 'reset' })}
        >
          Reset
        </button>
        <button
          type="button"
          role="switch"
          className={styles.switch}
          aria-checked={snapshot?.traffic ?? false}
          disabled={!live}
          onClick={() => void send({ type: 'set_traffic', enabled: !(snapshot?.traffic ?? false) })}
        >
          <span className={styles.track} aria-hidden="true" />
          Traffic
        </button>
        <button
          type="button"
          role="switch"
          className={styles.switch}
          aria-checked={snapshot?.prefix_caching ?? false}
          aria-describedby={restartHintId}
          disabled={!live}
          onClick={() =>
            void send({ type: 'set_prefix_caching', enabled: !(snapshot?.prefix_caching ?? false) })
          }
        >
          <span className={styles.track} aria-hidden="true" />
          Prefix caching
        </button>
        <span id={restartHintId} className="visually-hidden">
          Changing this restarts the session.
        </span>
      </div>
    </div>
  );
}

function Status({
  connection,
  snapshot,
  reconnect,
}: Pick<Props, 'connection' | 'snapshot' | 'reconnect'>) {
  switch (connection.status) {
    case 'connecting':
      return <p className={styles.status}>Connecting</p>;
    case 'retrying':
      return <Retrying retryAt={connection.retryAt} />;
    case 'closed':
      return (
        <p className={styles.status} data-tone="error">
          {connection.reason}{' '}
          <button type="button" className={styles.link} onClick={reconnect}>
            Start a new session
          </button>
        </p>
      );
    case 'open':
      return (
        <p className={styles.status} data-tone={snapshot?.paused ? undefined : 'live'}>
          <span className={styles.dot} aria-hidden="true" />
          {snapshot?.paused ? 'Paused' : 'Live'} at tick {formatInt(snapshot?.tick ?? 0)}
        </p>
      );
  }
}

function Retrying({ retryAt }: { retryAt: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => {
      setNow(Date.now());
    }, 500);
    return () => {
      clearInterval(timer);
    };
  }, []);
  const seconds = Math.max(0, Math.ceil((retryAt - now) / 1000));
  return (
    <p className={styles.status} data-tone="error">
      Connection lost. Retrying in {seconds} s.
    </p>
  );
}
