import { useMemo, useState } from 'react';

import type { Command, CommandResult } from './api/protocol';
import styles from './App.module.css';
import { AddRequestForm } from './components/AddRequestForm/AddRequestForm';
import { BlockPool } from './components/BlockPool/BlockPool';
import { EventLog } from './components/EventLog/EventLog';
import { RunningTable } from './components/RunningTable/RunningTable';
import { SessionBar } from './components/SessionBar/SessionBar';
import { TimeSeries } from './components/TimeSeries/TimeSeries';
import { Totals } from './components/Totals/Totals';
import { WaitingList } from './components/WaitingList/WaitingList';
import { useSimulation } from './hooks/useSimulation';
import { formatInt } from './lib/format';
import { HISTORY_LEN, type Session } from './state/simulation';

const REPO_URL = 'https://github.com/SohamDutta02/paged-kv-cache';

export function App() {
  const { state, send, reconnect } = useSimulation();

  return (
    <div className={styles.page}>
      <header className={styles.masthead}>
        <div className={styles.intro}>
          <h1 className={styles.title}>paged-kv-cache</h1>
          <p className={styles.lede}>
            A live run of the KV cache manager in this repository. Requests take fixed-size blocks
            from a shared pool and reuse cached blocks for any prompt prefix seen before. Forks
            share blocks until one side writes, and the scheduler preempts the newest sequence when
            memory runs out. This session is yours alone.
          </p>
        </div>
        <SessionBar
          connection={state.connection}
          snapshot={state.session?.snapshot ?? null}
          send={send}
          reconnect={reconnect}
        />
        {state.protocolError && (
          <p className={styles.protocolError} role="alert">
            {state.protocolError}
          </p>
        )}
      </header>

      {state.session ? (
        <Dashboard session={state.session} send={send} />
      ) : (
        <p className={styles.placeholder}>Starting a session…</p>
      )}

      <footer className={styles.footer}>
        <a href={REPO_URL}>Source on GitHub</a>
        <span>
          The scheduler is paged-kv-core running unmodified on the server. This page is React and
          TypeScript, with protocol types generated from the Rust code.
        </span>
      </footer>
    </div>
  );
}

interface DashboardProps {
  session: Session;
  send: (command: Command) => Promise<CommandResult>;
}

function Dashboard({ session, send }: DashboardProps) {
  const { config, snapshot, history } = session;
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  const active =
    snapshot.running.find((s) => s.id === hoveredId) ??
    snapshot.running.find((s) => s.id === selectedId) ??
    null;
  const copied = useMemo(
    () =>
      new Set(
        session.lastStepEvents.flatMap((e) => (e.kind.type === 'cow' ? [e.kind.dst_block] : [])),
      ),
    [session.lastStepEvents],
  );

  const ticks = history.map((m) => m.tick);
  const latest = snapshot.metrics;
  const peakQueue = Math.max(config.max_batch_size, ...history.map((m) => m.waiting));
  const sequencesMax = Math.ceil(peakQueue / 2) * 2;

  return (
    <main className={styles.main}>
      <div className={styles.primary}>
        <BlockPool config={config} snapshot={snapshot} active={active} copied={copied} />
        <div className={styles.side}>
          <RunningTable
            running={snapshot.running}
            maxBatchSize={config.max_batch_size}
            refCounts={snapshot.ref_counts}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onHover={setHoveredId}
            onFork={(seqId) => send({ type: 'fork', seq_id: seqId })}
          />
          <WaitingList waiting={snapshot.waiting} maxWaiting={config.max_waiting} />
          <AddRequestForm
            capacityTokens={config.num_blocks * config.block_size}
            systemPrompts={config.system_prompts}
            onAdd={(systemPrompt, promptLen, maxNewTokens) =>
              send({
                type: 'add_request',
                system_prompt: systemPrompt,
                prompt_len: promptLen,
                max_new_tokens: maxNewTokens,
              })
            }
          />
        </div>
      </div>

      <div className={styles.trends}>
        <TimeSeries
          title="Block usage"
          ticks={ticks}
          series={[
            {
              key: 'used',
              label: 'In use',
              color: 'var(--owned)',
              values: history.map((m) => m.used_blocks),
              area: true,
            },
            {
              key: 'shared',
              label: 'Shared',
              color: 'var(--shared)',
              values: history.map((m) => m.shared_blocks),
            },
            ...(snapshot.prefix_caching
              ? [
                  {
                    key: 'cached',
                    label: 'Cached, no owner',
                    shortLabel: 'Cached',
                    color: 'var(--ink-3)',
                    values: history.map((m) => m.evictable_blocks),
                  },
                ]
              : []),
          ]}
          yMax={config.num_blocks}
          windowTicks={HISTORY_LEN}
          tickMs={config.tick_ms}
          summary={`Now ${formatInt(latest.used_blocks)} of ${formatInt(config.num_blocks)} blocks in use, ${formatInt(latest.shared_blocks)} shared and ${formatInt(latest.evictable_blocks)} cached with no owner.`}
        />
        <TimeSeries
          title="Sequences"
          ticks={ticks}
          series={[
            {
              key: 'running',
              step: true,
              label: 'Running',
              color: 'var(--ink-1)',
              values: history.map((m) => m.running),
            },
            {
              key: 'waiting',
              step: true,
              label: 'Waiting',
              color: 'var(--ink-3)',
              values: history.map((m) => m.waiting),
            },
          ]}
          yMax={sequencesMax}
          windowTicks={HISTORY_LEN}
          tickMs={config.tick_ms}
          summary={`Now ${formatInt(latest.running)} running and ${formatInt(latest.waiting)} waiting.`}
        />
        <Totals totals={snapshot.totals} prefixCaching={snapshot.prefix_caching} />
      </div>

      <div className={styles.sheet}>
        <EventLog events={session.events} />
      </div>
    </main>
  );
}
