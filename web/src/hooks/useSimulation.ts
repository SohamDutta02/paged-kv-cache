import { useCallback, useEffect, useReducer, useRef, useState } from 'react';

import type { ClientMessage, Command, CommandResult, ServerMessage } from '../api/protocol';
import { initialState, simulationReducer, type SimulationState } from '../state/simulation';

/** Close codes after which reconnecting automatically would not help. */
const CLOSE_TRY_AGAIN_LATER = 1013;
const CLOSE_IDLE = 4000;
const MAX_RETRY_DELAY_MS = 15_000;

export function sessionUrl(location: Location = window.location): string {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${scheme}://${location.host}/api/ws`;
}

export interface Simulation {
  state: SimulationState;
  send: (command: Command) => Promise<CommandResult>;
  reconnect: () => void;
}

export function useSimulation(url: string = sessionUrl()): Simulation {
  const [state, dispatch] = useReducer(simulationReducer, initialState);
  const [generation, setGeneration] = useState(0);
  const socket = useRef<WebSocket | null>(null);
  const pending = useRef(new Map<number, (result: CommandResult) => void>());
  const nextId = useRef(1);

  useEffect(() => {
    let disposed = false;
    let retries = 0;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    const waiting = pending.current;

    const connect = () => {
      const ws = new WebSocket(url);
      socket.current = ws;

      ws.onmessage = (event: MessageEvent<string>) => {
        const message = JSON.parse(event.data) as ServerMessage;
        if (message.type === 'ack') {
          waiting.get(message.id)?.(message.result);
          waiting.delete(message.id);
          return;
        }
        if (message.type === 'init') retries = 0;
        dispatch({ type: 'message', message });
      };

      ws.onclose = (event) => {
        socket.current = null;
        for (const resolve of waiting.values()) {
          resolve({
            status: 'rejected',
            message: 'The connection closed before the server replied.',
          });
        }
        waiting.clear();
        if (disposed) return;

        if (event.code === CLOSE_TRY_AGAIN_LATER || event.code === CLOSE_IDLE) {
          const reason = event.reason || 'The session ended.';
          dispatch({ type: 'connection', connection: { status: 'closed', reason } });
          return;
        }
        const delay = Math.min(1000 * 2 ** retries, MAX_RETRY_DELAY_MS);
        retries += 1;
        dispatch({
          type: 'connection',
          connection: { status: 'retrying', retryAt: Date.now() + delay },
        });
        retryTimer = setTimeout(() => {
          dispatch({ type: 'connection', connection: { status: 'connecting' } });
          connect();
        }, delay);
      };
    };

    connect();
    return () => {
      disposed = true;
      clearTimeout(retryTimer);
      socket.current?.close();
    };
  }, [url, generation]);

  const send = useCallback((command: Command): Promise<CommandResult> => {
    const ws = socket.current;
    if (ws?.readyState !== WebSocket.OPEN) {
      return Promise.resolve({ status: 'rejected', message: 'Not connected to the server.' });
    }
    const id = nextId.current++;
    const message: ClientMessage = { id, command };
    return new Promise((resolve) => {
      pending.current.set(id, resolve);
      ws.send(JSON.stringify(message));
    });
  }, []);

  const reconnect = useCallback(() => {
    dispatch({ type: 'connection', connection: { status: 'connecting' } });
    setGeneration((g) => g + 1);
  }, []);

  return { state, send, reconnect };
}
