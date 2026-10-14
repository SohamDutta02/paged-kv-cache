import type { SequenceView } from '../api/protocol';

export type BlockState = 'free' | 'owned' | 'shared';

export interface BlockOwner {
  seqId: number;
  /** Index of the block in the owner's block table. */
  position: number;
}

export function blockState(refCount: number): BlockState {
  if (refCount === 0) return 'free';
  return refCount === 1 ? 'owned' : 'shared';
}

export function blockOwners(running: readonly SequenceView[], numBlocks: number): BlockOwner[][] {
  const owners = Array.from({ length: numBlocks }, (): BlockOwner[] => []);
  for (const seq of running) {
    seq.blocks.forEach((block, position) => {
      owners[block]?.push({ seqId: seq.id, position });
    });
  }
  return owners;
}
