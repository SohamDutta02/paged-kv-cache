import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { sequence, snapshot } from '../../test/fixtures';
import { RunningTable } from './RunningTable';

const running = [
  sequence(1, [3, 7], { system_prompt: 0, cached_tokens: 8 }),
  sequence(2, [3, 9], { parent: 1, origin: 'user' }),
];
const { ref_counts: refCounts } = snapshot(1, running);

function renderTable(overrides: Partial<Parameters<typeof RunningTable>[0]> = {}) {
  const props = {
    running,
    maxBatchSize: 4,
    refCounts,
    selectedId: null,
    onSelect: vi.fn(),
    onHover: vi.fn(),
    onFork: vi.fn().mockResolvedValue({ status: 'ok', seq_id: 5 }),
    ...overrides,
  };
  render(<RunningTable {...props} />);
  return props;
}

describe('RunningTable', () => {
  it('shows lineage and ownership', () => {
    renderTable();
    expect(screen.getByText('fork of #1')).toBeInTheDocument();
    expect(screen.getByText('yours')).toBeInTheDocument();
    expect(screen.getByText('2 of 4')).toBeInTheDocument();
  });

  it('shows the system prompt and how much of the prompt came from the cache', () => {
    renderTable();
    expect(screen.getByText('prompt A')).toBeInTheDocument();
    expect(screen.getByText('8 cached')).toBeInTheDocument();
  });

  it('selects and deselects a sequence', async () => {
    const props = renderTable({ selectedId: 1 });
    const user = userEvent.setup();
    const first = screen.getByRole('button', { name: '#1' });
    expect(first).toHaveAttribute('aria-pressed', 'true');

    await user.click(first);
    expect(props.onSelect).toHaveBeenCalledWith(null);
    await user.click(screen.getByRole('button', { name: '#2' }));
    expect(props.onSelect).toHaveBeenCalledWith(2);
  });

  it('forks and reports the result', async () => {
    const props = renderTable();
    await userEvent.setup().click(screen.getByRole('button', { name: 'Fork #1' }));
    expect(props.onFork).toHaveBeenCalledWith(1);
    expect(await screen.findByText('Forked #1 into #5.')).toBeInTheDocument();
  });

  it('disables forking when the batch is full', () => {
    renderTable({ maxBatchSize: 2 });
    expect(screen.getByRole('button', { name: 'Fork #1' })).toBeDisabled();
  });
});
