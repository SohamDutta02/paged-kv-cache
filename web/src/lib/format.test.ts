import { describe, expect, it } from 'vitest';

import { formatInt, plural, systemPromptLabel, tokenSpan } from './format';

describe('format', () => {
  it('groups thousands', () => {
    expect(formatInt(12345)).toBe('12,345');
  });

  it('pluralizes', () => {
    expect(plural(1, 'token')).toBe('1 token');
    expect(plural(2, 'token')).toBe('2 tokens');
  });

  it('letters system prompts', () => {
    expect([0, 1, 2].map(systemPromptLabel)).toEqual(['A', 'B', 'C']);
  });

  it('gives the token span held by a block, clipped to the tokens written', () => {
    expect(tokenSpan(0, 8, 20)).toBe('0–7');
    expect(tokenSpan(2, 8, 20)).toBe('16–19');
    expect(tokenSpan(2, 8, 17)).toBe('16');
  });
});
