import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { Event } from '../../api/protocol';
import { EventLog } from './EventLog';

const events: Event[] = [
  {
    tick: 1,
    seq_id: 1,
    kind: {
      type: 'submitted',
      origin: 'user',
      system_prompt: null,
      prompt_len: 12,
      max_new_tokens: 4,
    },
  },
  { tick: 2, seq_id: 1, kind: { type: 'admitted', cached_tokens: 0, computed_tokens: 12 } },
  {
    tick: 2,
    seq_id: 3,
    kind: {
      type: 'submitted',
      origin: 'traffic',
      system_prompt: 1,
      prompt_len: 5,
      max_new_tokens: 9,
    },
  },
  { tick: 3, seq_id: 3, kind: { type: 'admitted', cached_tokens: 48, computed_tokens: 13 } },
  { tick: 3, seq_id: 2, kind: { type: 'forked', parent: 1 } },
  { tick: 4, seq_id: 2, kind: { type: 'cow', src_block: 5, dst_block: 6 } },
  { tick: 5, seq_id: 1, kind: { type: 'preempted' } },
];

function rowTexts() {
  return screen
    .getAllByRole('row')
    .slice(1)
    .map((row) => row.textContent);
}

describe('EventLog', () => {
  it('lists events newest first with readable details', () => {
    render(<EventLog events={events} />);
    const rows = rowTexts();
    expect(rows).toHaveLength(7);
    expect(rows[0]).toContain('Preempted');
    expect(rows[1]).toContain('Shared block 5 copied to block 6');
    expect(rows[3]).toContain('48 of 61 tokens reused from the cache');
    expect(rows[4]).toContain('System prompt B + 5 prompt tokens, 9 to generate');
    expect(rows[5]).toContain('All 12 tokens computed');
    expect(rows[6]).toContain('12 prompt tokens, 4 to generate, by you');
  });

  it('filters by kind', async () => {
    render(<EventLog events={events} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole('radio', { name: 'Forks and copies' }));
    expect(rowTexts().map((r) => r.slice(0, 1))).toEqual(['4', '3']);

    await user.click(screen.getByRole('radio', { name: 'Cache hits' }));
    expect(rowTexts()).toHaveLength(1);
    expect(rowTexts()[0]).toContain('48 of 61');
  });
});
