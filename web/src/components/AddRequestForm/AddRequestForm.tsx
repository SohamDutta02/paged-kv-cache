import { useId, useState, type SubmitEvent } from 'react';

import type { CommandResult } from '../../api/protocol';
import { formatInt, systemPromptLabel } from '../../lib/format';
import styles from './AddRequestForm.module.css';

interface Props {
  capacityTokens: number;
  /** Token length of each shared system prompt. */
  systemPrompts: readonly number[];
  onAdd: (
    systemPrompt: number | null,
    promptLen: number,
    maxNewTokens: number,
  ) => Promise<CommandResult>;
}

const NONE = 'none';

export function AddRequestForm({ capacityTokens, systemPrompts, onAdd }: Props) {
  const id = useId();
  const [system, setSystem] = useState(systemPrompts.length > 0 ? '0' : NONE);
  const [prompt, setPrompt] = useState('16');
  const [generate, setGenerate] = useState('24');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);

  const systemPrompt = system === NONE ? null : Number(system);
  const systemLen = systemPrompt === null ? 0 : (systemPrompts[systemPrompt] ?? 0);
  const promptLen = Number(prompt);
  const maxNewTokens = Number(generate);
  const problem = validate(systemLen, promptLen, maxNewTokens, capacityTokens);

  const submit = async (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (problem) return;
    setBusy(true);
    const result = await onAdd(systemPrompt, promptLen, maxNewTokens);
    setBusy(false);
    setStatus(
      result.status === 'ok'
        ? { ok: true, text: `Added #${result.seq_id ?? '?'} to the queue.` }
        : { ok: false, text: result.message },
    );
  };

  return (
    <section className={styles.section} aria-labelledby={`${id}-heading`}>
      <h2 id={`${id}-heading`} className={styles.heading}>
        Add a request
      </h2>
      <form className={styles.form} onSubmit={(e) => void submit(e)} noValidate>
        {systemPrompts.length > 0 && (
          <div className={styles.field}>
            <label htmlFor={`${id}-system`}>System prompt</label>
            <select
              id={`${id}-system`}
              value={system}
              onChange={(e) => {
                setSystem(e.target.value);
              }}
            >
              <option value={NONE}>None</option>
              {systemPrompts.map((len, i) => (
                <option key={i} value={i}>
                  {systemPromptLabel(i)}, {formatInt(len)} tokens
                </option>
              ))}
            </select>
          </div>
        )}
        <div className={styles.field}>
          <label htmlFor={`${id}-prompt`}>Prompt tokens</label>
          <input
            id={`${id}-prompt`}
            type="number"
            inputMode="numeric"
            min={1}
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value);
            }}
          />
        </div>
        <div className={styles.field}>
          <label htmlFor={`${id}-generate`}>Tokens to generate</label>
          <input
            id={`${id}-generate`}
            type="number"
            inputMode="numeric"
            min={1}
            value={generate}
            onChange={(e) => {
              setGenerate(e.target.value);
            }}
          />
        </div>
        <button type="submit" className={styles.submit} disabled={busy || problem !== null}>
          Add request
        </button>
      </form>
      <p
        className={styles.status}
        data-tone={problem ? 'hint' : status?.ok === false ? 'error' : 'ok'}
        aria-live="polite"
      >
        {problem ?? status?.text ?? `Up to ${formatInt(capacityTokens)} tokens in total.`}
      </p>
    </section>
  );
}

function validate(
  systemLen: number,
  promptLen: number,
  maxNewTokens: number,
  capacity: number,
): string | null {
  if (!Number.isInteger(promptLen) || promptLen < 1)
    return 'Prompt tokens must be a whole number of at least 1.';
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1)
    return 'Tokens to generate must be a whole number of at least 1.';
  const total = systemLen + promptLen + maxNewTokens;
  if (total > capacity) {
    return `That is ${formatInt(total)} tokens; the pool holds ${formatInt(capacity)}.`;
  }
  return null;
}
