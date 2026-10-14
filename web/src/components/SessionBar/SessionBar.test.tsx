import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { snapshot } from '../../test/fixtures';
import { SessionBar } from './SessionBar';

describe('SessionBar', () => {
  it('toggles prefix caching and says it restarts the session', async () => {
    const send = vi.fn().mockResolvedValue({ status: 'ok', seq_id: null });
    render(
      <SessionBar
        connection={{ status: 'open' }}
        snapshot={snapshot(3)}
        send={send}
        reconnect={vi.fn()}
      />,
    );

    const toggle = screen.getByRole('switch', { name: 'Prefix caching' });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    expect(toggle).toHaveAccessibleDescription('Changing this restarts the session.');

    await userEvent.setup().click(toggle);
    expect(send).toHaveBeenCalledWith({ type: 'set_prefix_caching', enabled: false });
  });

  it('only allows stepping while paused', () => {
    render(
      <SessionBar
        connection={{ status: 'open' }}
        snapshot={snapshot(3)}
        send={vi.fn()}
        reconnect={vi.fn()}
      />,
    );
    expect(screen.getByRole('button', { name: 'Step' })).toBeDisabled();
    expect(screen.getByText('Live at tick 3')).toBeInTheDocument();
  });
});
