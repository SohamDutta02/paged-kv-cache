import { describe, expect, it } from 'vitest';

import { sequence } from '../test/fixtures';
import { blockOwners, blockState } from './ownership';

describe('blockOwners', () => {
  it('records each owner with the block’s position in its table', () => {
    const owners = blockOwners([sequence(1, [5, 2]), sequence(2, [5, 9])], 10);
    expect(owners[5]).toEqual([
      { seqId: 1, position: 0 },
      { seqId: 2, position: 0 },
    ]);
    expect(owners[2]).toEqual([{ seqId: 1, position: 1 }]);
    expect(owners[0]).toEqual([]);
  });

  it('ignores block ids outside the pool', () => {
    expect(() => blockOwners([sequence(1, [99])], 4)).not.toThrow();
  });
});

describe('blockState', () => {
  it('maps ref counts to states', () => {
    expect([0, 1, 2, 5].map(blockState)).toEqual(['free', 'owned', 'shared', 'shared']);
  });
});
