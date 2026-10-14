import type { Event, Metrics, PoolConfig, ServerMessage, Snapshot } from '../api/protocol';

export const HISTORY_LEN = 150;
export const EVENT_LOG_LEN = 200;

export type Connection =
  | { status: 'connecting' }
  | { status: 'open' }
  | { status: 'retrying'; retryAt: number }
  | { status: 'closed'; reason: string };

export interface Session {
  config: PoolConfig;
  snapshot: Snapshot;
  history: Metrics[];
  /** Oldest first. */
  events: Event[];
  /** Events from the most recent scheduler step, for one-off highlights. */
  lastStepEvents: Event[];
}

export interface SimulationState {
  connection: Connection;
  session: Session | null;
  protocolError: string | null;
}

export type SessionMessage = Exclude<ServerMessage, { type: 'ack' }>;

export type SimulationAction =
  { type: 'connection'; connection: Connection } | { type: 'message'; message: SessionMessage };

export const initialState: SimulationState = {
  connection: { status: 'connecting' },
  session: null,
  protocolError: null,
};

export function simulationReducer(
  state: SimulationState,
  action: SimulationAction,
): SimulationState {
  switch (action.type) {
    case 'connection':
      return { ...state, connection: action.connection };
    case 'message':
      return applyMessage(state, action.message);
  }
}

function applyMessage(state: SimulationState, message: SessionMessage): SimulationState {
  switch (message.type) {
    case 'init':
      return {
        connection: { status: 'open' },
        protocolError: null,
        session: {
          config: message.config,
          snapshot: message.snapshot,
          history: message.history.slice(-HISTORY_LEN),
          events: message.events.slice(-EVENT_LOG_LEN),
          lastStepEvents: [],
        },
      };
    case 'tick': {
      const { session } = state;
      if (!session) return state;
      return {
        ...state,
        session: {
          ...session,
          snapshot: message.snapshot,
          history: appendCapped(session.history, [message.snapshot.metrics], HISTORY_LEN),
          events: appendCapped(session.events, message.events, EVENT_LOG_LEN),
          lastStepEvents: message.events,
        },
      };
    }
    case 'state': {
      const { session } = state;
      if (!session) return state;
      return {
        ...state,
        session: {
          ...session,
          snapshot: message.snapshot,
          events: appendCapped(session.events, message.events, EVENT_LOG_LEN),
        },
      };
    }
    case 'error':
      return { ...state, protocolError: message.message };
  }
}

function appendCapped<T>(items: T[], added: T[], cap: number): T[] {
  if (added.length === 0) return items;
  const next = items.concat(added);
  return next.length > cap ? next.slice(next.length - cap) : next;
}
