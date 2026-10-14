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

const systemPrompts = [40, 56];

describe('AddRequestForm', () => {
  it('sends the request with the chosen system prompt and confirms it', async () => {
    const onAdd = vi.fn().mockResolvedValue({ status: 'ok', seq_id: 42 });
    render(<AddRequestForm capacityTokens={512} systemPrompts={systemPrompts} onAdd={onAdd} />);

    const user = await fill('30', '10');
    await user.selectOptions(screen.getByLabelText('System prompt'), 'B, 56 tokens');
    await user.click(screen.getByRole('button', { name: 'Add request' }));

    expect(onAdd).toHaveBeenCalledWith(1, 30, 10);
    expect(await screen.findByText('Added #42 to the queue.')).toBeInTheDocument();
  });

  it('can send a request without a system prompt', async () => {
    const onAdd = vi.fn().mockResolvedValue({ status: 'ok', seq_id: 1 });
    render(<AddRequestForm capacityTokens={512} systemPrompts={systemPrompts} onAdd={onAdd} />);

    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText('System prompt'), 'None');
    await user.click(screen.getByRole('button', { name: 'Add request' }));
    expect(onAdd).toHaveBeenCalledWith(null, 16, 24);
  });

  it('shows the server’s reason when a request is rejected', async () => {
    const onAdd = vi.fn().mockResolvedValue({ status: 'rejected', message: 'The queue is full.' });
    render(<AddRequestForm capacityTokens={512} systemPrompts={systemPrompts} onAdd={onAdd} />);

    await userEvent.setup().click(screen.getByRole('button', { name: 'Add request' }));

    expect(await screen.findByText('The queue is full.')).toBeInTheDocument();
  });

  it('counts the system prompt toward the pool before sending', async () => {
    const onAdd = vi.fn();
    render(<AddRequestForm capacityTokens={100} systemPrompts={systemPrompts} onAdd={onAdd} />);

    await fill('40', '30');

    expect(screen.getByRole('button', { name: 'Add request' })).toBeDisabled();
    expect(screen.getByText('That is 110 tokens; the pool holds 100.')).toBeInTheDocument();
    expect(onAdd).not.toHaveBeenCalled();
  });
});
