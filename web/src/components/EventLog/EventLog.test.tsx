import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { Event } from '../../api/protocol';
import { EventLog } from './EventLog';

const events: Event[] = [
  {
    tick: 1,
    seq_id: 1,
    kind: { type: 'submitted', origin: 'user', prompt_len: 12, max_new_tokens: 4 },
  },
  { tick: 2, seq_id: 1, kind: { type: 'admitted' } },
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
    expect(rows).toHaveLength(5);
    expect(rows[0]).toContain('Preempted');
    expect(rows[1]).toContain('Shared block 5 copied to block 6');
    expect(rows[4]).toContain('12 prompt tokens, 4 to generate, by you');
  });

  it('filters by kind', async () => {
    render(<EventLog events={events} />);
    await userEvent.setup().click(screen.getByRole('radio', { name: 'Forks and copies' }));
    expect(rowTexts().map((r) => r.slice(0, 1))).toEqual(['4', '3']);
  });
});
