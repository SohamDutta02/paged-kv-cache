const integer = new Intl.NumberFormat('en-US');

export function formatInt(n: number): string {
  return integer.format(n);
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${formatInt(n)} ${n === 1 ? one : many}`;
}

/** Token positions held by block `position` of a sequence, e.g. "8–15". */
export function tokenSpan(position: number, blockSize: number, tokens: number): string {
  const first = position * blockSize;
  const last = Math.min(first + blockSize, tokens) - 1;
  return first === last ? `${first}` : `${first}–${last}`;
}
