import { useState, type PointerEvent } from 'react';

import { useElementWidth } from '../../hooks/useElementWidth';
import { formatInt } from '../../lib/format';
import styles from './TimeSeries.module.css';

export interface Series {
  key: string;
  label: string;
  /** Any CSS color, typically a token like `var(--owned)`. */
  color: string;
  values: readonly number[];
  area?: boolean;
  /** Draw as steps, for counts that change discretely. */
  step?: boolean;
}

interface Props {
  title: string;
  /** One tick per value, oldest first. */
  ticks: readonly number[];
  series: readonly Series[];
  yMax: number;
  windowTicks: number;
  tickMs: number;
  summary: string;
  height?: number;
}

const MARGIN = { top: 10, right: 12, bottom: 22, left: 30 };
/** The x-axis starts this wide so a new session's first ticks aren't stretched. */
const MIN_SPAN_TICKS = 30;
/** End labels closer than this are dropped rather than nudged apart. */
const MIN_LABEL_GAP = 13;

export function TimeSeries({
  title,
  ticks,
  series,
  yMax,
  windowTicks,
  tickMs,
  summary,
  height = 150,
}: Props) {
  const [ref, width] = useElementWidth<HTMLDivElement>(320);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const lastTick = ticks.at(-1) ?? 0;
  const span = Math.min(Math.max(ticks.length, MIN_SPAN_TICKS), windowTicks);
  const firstTick = lastTick - span + 1;
  const endLabelWidth = 64;
  const plotWidth = Math.max(width - MARGIN.left - MARGIN.right - endLabelWidth, 40);
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  const x = (tick: number) =>
    MARGIN.left + ((tick - firstTick) / Math.max(span - 1, 1)) * plotWidth;
  const y = (value: number) => MARGIN.top + plotHeight - (value / Math.max(yMax, 1)) * plotHeight;
  const baseline = y(0);

  const gridValues = [0, Math.round(yMax / 2), yMax];
  const spanSeconds = Math.round((span * tickMs) / 1000);

  const ends = series.map((s) => ({ s, value: s.values.at(-1) ?? 0 }));
  const endYs = ends.map((e) => y(e.value)).sort((a, b) => a - b);
  const labelsFit = endYs.every((v, i) => i === 0 || v - (endYs[i - 1] ?? 0) >= MIN_LABEL_GAP);

  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (ticks.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const tick = firstTick + ((event.clientX - rect.left - MARGIN.left) / plotWidth) * (span - 1);
    const offset = ticks.length - 1 - Math.round(lastTick - tick);
    setHoverIndex(Math.min(Math.max(offset, 0), ticks.length - 1));
  };

  const hoverTick = hoverIndex === null ? null : ticks[hoverIndex];
  const tooltipLeft = hoverTick === undefined || hoverTick === null ? 0 : x(hoverTick);

  return (
    <figure className={styles.figure}>
      <figcaption className={styles.caption}>
        <span className={styles.title}>{title}</span>
        {series.length > 1 && (
          <span className={styles.legend}>
            {series.map((s) => (
              <span key={s.key} className={styles.legendItem}>
                <span className={styles.key} style={{ background: s.color }} />
                {s.label}
              </span>
            ))}
          </span>
        )}
      </figcaption>
      <p className="visually-hidden">{summary}</p>

      <div ref={ref} className={styles.plot}>
        <svg
          width={width}
          height={height}
          aria-hidden="true"
          onPointerMove={onPointerMove}
          onPointerLeave={() => {
            setHoverIndex(null);
          }}
        >
          {gridValues.map((v) => (
            <g key={v}>
              <line
                className={styles.grid}
                x1={MARGIN.left}
                x2={MARGIN.left + plotWidth}
                y1={y(v)}
                y2={y(v)}
              />
              <text
                className={styles.axis}
                x={MARGIN.left - 6}
                y={y(v)}
                dy="0.32em"
                textAnchor="end"
              >
                {formatInt(v)}
              </text>
            </g>
          ))}
          <text className={styles.axis} x={MARGIN.left} y={height - 4}>
            {spanSeconds} s ago
          </text>
          <text className={styles.axis} x={MARGIN.left + plotWidth} y={height - 4} textAnchor="end">
            now
          </text>

          {series.map((s) => {
            const line = linePath(s, ticks, x, y);
            if (!line) return null;
            const firstX = x(ticks[0] ?? 0);
            const lastX = x(lastTick);
            return (
              <g key={s.key}>
                {s.area && (
                  <path
                    d={`${line}L${lastX},${baseline}L${firstX},${baseline}Z`}
                    fill={s.color}
                    fillOpacity={0.12}
                  />
                )}
                <path d={line} className={styles.line} stroke={s.color} />
              </g>
            );
          })}

          {ends.map(({ s, value }) => (
            <g key={s.key}>
              <circle className={styles.dot} cx={x(lastTick)} cy={y(value)} r={4} fill={s.color} />
              {labelsFit && (
                <text className={styles.endLabel} x={x(lastTick) + 9} y={y(value)} dy="0.32em">
                  {series.length > 1 ? `${s.label} ${formatInt(value)}` : formatInt(value)}
                </text>
              )}
            </g>
          ))}

          {hoverTick !== null && hoverTick !== undefined && hoverIndex !== null && (
            <g>
              <line
                className={styles.crosshair}
                x1={x(hoverTick)}
                x2={x(hoverTick)}
                y1={MARGIN.top}
                y2={baseline}
              />
              {series.map((s) => (
                <circle
                  key={s.key}
                  className={styles.dot}
                  cx={x(hoverTick)}
                  cy={y(s.values[hoverIndex] ?? 0)}
                  r={4}
                  fill={s.color}
                />
              ))}
            </g>
          )}
        </svg>

        {hoverTick !== null && hoverTick !== undefined && hoverIndex !== null && (
          <div
            className={styles.tooltip}
            style={
              tooltipLeft > width / 2
                ? { right: width - tooltipLeft + 10 }
                : { left: tooltipLeft + 10 }
            }
          >
            <div className={styles.tooltipTitle}>Tick {formatInt(hoverTick)}</div>
            {series.map((s) => (
              <div key={s.key} className={styles.tooltipRow}>
                <span className={styles.key} style={{ background: s.color }} />
                <span>{s.label}</span>
                <span className={styles.tooltipValue}>{formatInt(s.values[hoverIndex] ?? 0)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}

function linePath(
  s: Series,
  ticks: readonly number[],
  x: (tick: number) => number,
  y: (value: number) => number,
): string | null {
  if (s.values.length === 0) return null;
  return s.values
    .map((v, i) => {
      const px = x(ticks[i] ?? 0);
      const py = y(v);
      if (i === 0) return `M${px},${py}`;
      return s.step ? `H${px}V${py}` : `L${px},${py}`;
    })
    .join('');
}
