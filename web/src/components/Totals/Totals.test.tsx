import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { snapshot } from '../../test/fixtures';
import { Totals } from './Totals';

function value(label: string) {
  return screen.getByText(label).nextElementSibling?.textContent;
}

describe('Totals', () => {
  it('derives the cache hit rate from prompt tokens', () => {
    const totals = {
      ...snapshot(0).totals,
      prompt_tokens_cached: 300,
      prompt_tokens_computed: 100,
      cache_evictions: 1234,
    };
    render(<Totals totals={totals} prefixCaching />);
    expect(value('Cache hit rate')).toBe('75%');
    expect(value('Cache evictions')).toBe('1,234');
  });

  it('shows no hit rate before any prompt is admitted', () => {
    render(<Totals totals={snapshot(0).totals} prefixCaching />);
    expect(value('Cache hit rate')).toBe('–');
  });

  it('leaves out the cache rows when caching is off', () => {
    render(<Totals totals={snapshot(0).totals} prefixCaching={false} />);
    expect(screen.queryByText('Cache hit rate')).toBeNull();
  });
});
