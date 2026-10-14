import { describe, expect, it } from 'vitest';

import type { Event } from '../api/protocol';
import { config, metrics, snapshot } from '../test/fixtures';
import {
  EVENT_LOG_LEN,
  HISTORY_LEN,
  initialState,
  simulationReducer,
  type SimulationState,
} from './simulation';

const finished = (tick: number, seqId: number): Event => ({
  tick,
  seq_id: seqId,
  kind: { type: 'finished' },
});

function started(): SimulationState {
  return simulationReducer(initialState, {
    type: 'message',
    message: {
      type: 'init',
      config,
      snapshot: snapshot(0),
      history: [metrics(0)],
      events: [],
    },
  });
}

describe('simulationReducer', () => {
  it('opens a session on init', () => {
    const state = started();
    expect(state.connection).toEqual({ status: 'open' });
    expect(state.session?.config).toEqual(config);
    expect(state.session?.history).toHaveLength(1);
  });

  it('ignores ticks that arrive before init', () => {
    const state = simulationReducer(initialState, {
      type: 'message',
      message: { type: 'tick', snapshot: snapshot(1), events: [] },
    });
    expect(state).toBe(initialState);
  });

  it('appends a history point and events on every tick, capped', () => {
    let state = started();
    for (let tick = 1; tick <= HISTORY_LEN + 10; tick++) {
      state = simulationReducer(state, {
        type: 'message',
        message: {
          type: 'tick',
          snapshot: snapshot(tick),
          events: [finished(tick, tick), finished(tick, -tick)],
        },
      });
    }
    const session = state.session;
    expect(session?.history).toHaveLength(HISTORY_LEN);
    expect(session?.history.at(-1)?.tick).toBe(HISTORY_LEN + 10);
    expect(session?.events).toHaveLength(EVENT_LOG_LEN);
    expect(session?.events.at(-1)?.seq_id).toBe(-(HISTORY_LEN + 10));
    expect(session?.lastStepEvents).toHaveLength(2);
  });

  it('updates the snapshot on state messages without adding history', () => {
    const before = started();
    const after = simulationReducer(before, {
      type: 'message',
      message: {
        type: 'state',
        snapshot: { ...snapshot(0), paused: true },
        events: [finished(0, 3)],
      },
    });
    expect(after.session?.snapshot.paused).toBe(true);
    expect(after.session?.history).toBe(before.session?.history);
    expect(after.session?.events).toHaveLength(1);
  });

  it('replaces everything on a second init, as after a reset', () => {
    let state = started();
    state = simulationReducer(state, {
      type: 'message',
      message: { type: 'tick', snapshot: snapshot(1), events: [finished(1, 1)] },
    });
    state = simulationReducer(state, {
      type: 'message',
      message: { type: 'init', config, snapshot: snapshot(0), history: [metrics(0)], events: [] },
    });
    expect(state.session?.events).toEqual([]);
    expect(state.session?.history).toHaveLength(1);
  });

  it('records protocol errors', () => {
    const state = simulationReducer(started(), {
      type: 'message',
      message: { type: 'error', message: 'Unreadable message' },
    });
    expect(state.protocolError).toBe('Unreadable message');
  });
});
