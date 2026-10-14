import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { config, sequence, snapshot } from '../../test/fixtures';
import { BlockPool } from './BlockPool';

const parent = sequence(1, [3, 7]);
const child = sequence(2, [3, 9], { parent: 1 });
const snap = snapshot(5, [parent, child]);

function blocks() {
  return Array.from(screen.getByRole('img').children) as HTMLElement[];
}

describe('BlockPool', () => {
  it('draws every block with its state and summarizes usage', () => {
    render(<BlockPool config={config} snapshot={snap} active={null} copied={new Set()} />);

    expect(
      screen.getByRole('img', { name: '3 of 16 blocks in use, 1 shared' }),
    ).toBeInTheDocument();
    const cells = blocks();
    expect(cells).toHaveLength(config.num_blocks);
    expect(cells[3]).toHaveAttribute('data-state', 'shared');
    expect(cells[3]).toHaveTextContent('×2');
    expect(cells[7]).toHaveAttribute('data-state', 'owned');
    expect(cells[7]).toHaveTextContent('#1');
    expect(cells[0]).toHaveAttribute('data-state', 'free');
  });

  it('traces the active sequence by block-table position', () => {
    render(<BlockPool config={config} snapshot={snap} active={child} copied={new Set()} />);

    const cells = blocks();
    expect(cells[3]).toHaveAttribute('data-traced');
    expect(cells[3]).toHaveTextContent('0');
    expect(cells[9]).toHaveTextContent('1');
    expect(cells[7]).not.toHaveAttribute('data-traced');

    const table = screen.getByRole('heading', { name: 'Block table for #2' }).parentElement;
    if (!table) throw new Error('missing block table');
    const entries = within(table).getAllByRole('listitem');
    expect(entries.map((e) => e.textContent)).toEqual([
      '0block 3tokens 0–7',
      '1block 9tokens 8–12',
    ]);
  });

  it('marks blocks that were just copied on write', () => {
    render(<BlockPool config={config} snapshot={snap} active={null} copied={new Set([9])} />);
    expect(blocks()[9]?.children).toHaveLength(3);
    expect(blocks()[7]?.children).toHaveLength(2);
  });
});
