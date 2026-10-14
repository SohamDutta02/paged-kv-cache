import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ClientMessage, ServerMessage } from '../api/protocol';
import { config, metrics, snapshot } from '../test/fixtures';
import { useSimulation } from './useSimulation';

class FakeSocket {
  static readonly OPEN = 1;
  static instances: FakeSocket[] = [];

  readonly url: string;
  readyState = FakeSocket.OPEN;
  sent: ClientMessage[] = [];
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(JSON.parse(data) as ClientMessage);
  }

  close() {
    this.serverCloses(1000);
  }

  serverSends(message: ServerMessage) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }

  serverCloses(code: number, reason = '') {
    this.readyState = 3;
    this.onclose?.({ code, reason });
  }
}

const init: ServerMessage = {
  type: 'init',
  config,
  snapshot: snapshot(0),
  history: [metrics(0)],
  events: [],
};

function latest(): FakeSocket {
  const socket = FakeSocket.instances.at(-1);
  if (!socket) throw new Error('no socket opened');
  return socket;
}

beforeEach(() => {
  FakeSocket.instances = [];
  vi.stubGlobal('WebSocket', FakeSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('useSimulation', () => {
  it('opens the session on init', () => {
    const { result } = renderHook(() => useSimulation('ws://test/api/ws'));
    expect(result.current.state.connection.status).toBe('connecting');

    act(() => {
      latest().serverSends(init);
    });
    expect(result.current.state.connection.status).toBe('open');
    expect(result.current.state.session?.config).toEqual(config);
  });

  it('resolves a command with the matching ack', async () => {
    const { result } = renderHook(() => useSimulation('ws://test/api/ws'));
    act(() => {
      latest().serverSends(init);
    });

    const reply = result.current.send({ type: 'pause' });
    const sent = latest().sent[0];
    expect(sent?.command).toEqual({ type: 'pause' });
    act(() => {
      latest().serverSends({
        type: 'ack',
        id: sent?.id ?? -1,
        result: { status: 'ok', seq_id: null },
      });
    });
    await expect(reply).resolves.toEqual({ status: 'ok', seq_id: null });
  });

  it('rejects pending commands and retries when the connection drops', async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useSimulation('ws://test/api/ws'));
    act(() => {
      latest().serverSends(init);
    });
    const reply = result.current.send({ type: 'step' });

    act(() => {
      latest().serverCloses(1006);
    });
    await expect(reply).resolves.toMatchObject({ status: 'rejected' });
    expect(result.current.state.connection.status).toBe('retrying');

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(FakeSocket.instances).toHaveLength(2);
    expect(result.current.state.connection.status).toBe('connecting');
  });

  it('stops retrying when the server turns the session away', () => {
    const { result } = renderHook(() => useSimulation('ws://test/api/ws'));
    act(() => {
      latest().serverCloses(1013, 'Too many sessions are open.');
    });
    expect(result.current.state.connection).toEqual({
      status: 'closed',
      reason: 'Too many sessions are open.',
    });

    act(() => {
      result.current.reconnect();
    });
    expect(FakeSocket.instances).toHaveLength(2);
  });
});
