import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { AddRequestForm } from './AddRequestForm';

async function fill(prompt: string, generate: string) {
  const user = userEvent.setup();
  await user.clear(screen.getByLabelText('Prompt tokens'));
  await user.type(screen.getByLabelText('Prompt tokens'), prompt);
  await user.clear(screen.getByLabelText('Tokens to generate'));
  await user.type(screen.getByLabelText('Tokens to generate'), generate);
  return user;
}

describe('AddRequestForm', () => {
  it('sends the request and confirms the new sequence', async () => {
    const onAdd = vi.fn().mockResolvedValue({ status: 'ok', seq_id: 42 });
    render(<AddRequestForm capacityTokens={512} onAdd={onAdd} />);

    const user = await fill('30', '10');
    await user.click(screen.getByRole('button', { name: 'Add request' }));

    expect(onAdd).toHaveBeenCalledWith(30, 10);
    expect(await screen.findByText('Added #42 to the queue.')).toBeInTheDocument();
  });

  it('shows the server’s reason when a request is rejected', async () => {
    const onAdd = vi.fn().mockResolvedValue({ status: 'rejected', message: 'The queue is full.' });
    render(<AddRequestForm capacityTokens={512} onAdd={onAdd} />);

    await userEvent.setup().click(screen.getByRole('button', { name: 'Add request' }));

    expect(await screen.findByText('The queue is full.')).toBeInTheDocument();
  });

  it('blocks requests larger than the pool before sending', async () => {
    const onAdd = vi.fn();
    render(<AddRequestForm capacityTokens={64} onAdd={onAdd} />);

    await fill('60', '10');

    expect(screen.getByRole('button', { name: 'Add request' })).toBeDisabled();
    expect(screen.getByText('That is 70 tokens; the pool holds 64.')).toBeInTheDocument();
    expect(onAdd).not.toHaveBeenCalled();
  });
});
