import { useMemo } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useChartPalette } from "@/lib/theme";
import { fmtCompact, fmtDate, fmtNumber, fmtShortDate, isNum } from "@/lib/format";

function ChartTooltip({ active, payload, label, palette }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const row = (color, name, value) => (
    <div className="flex items-center justify-between gap-6">
      <span className="flex items-center gap-1.5 text-ink-2">
        <span className="inline-block h-0.5 w-3" style={{ background: color }} />
        {name}
      </span>
      <span className="tabular-nums text-ink">{value}</span>
    </div>
  );
  return (
    <div className="min-w-[180px] rounded border border-line bg-surface px-3 py-2 text-xs shadow-pop">
      <p className="mb-1.5 font-medium text-ink">{fmtDate(label)}</p>
      <div className="space-y-1">
        {isNum(p.history) && row(palette.actual, "Actual", fmtNumber(p.history))}
        {isNum(p.forecast) && !isNum(p.history) && row(palette.forecast, "Forecast", fmtNumber(p.forecast))}
        {Array.isArray(p.band) && !isNum(p.history) && row(palette.band, "95% range", `${fmtNumber(p.band[0])} – ${fmtNumber(p.band[1])}`)}
      </div>
    </div>
  );
}

/** Actuals + forecast line with its 95% band. `range` = periods of history shown. */
export default function ForecastChart({ result, range = 180, height = 340 }) {
  const palette = useChartPalette();
  const best = result?.best_model;

  const data = useMemo(() => {
    if (!best?.forecast?.length) return [];
    const hDates = result.historical?.dates || [];
    const hVals = result.historical?.values || [];
    const start = Math.max(0, hDates.length - range);
    const hist = hDates.slice(start).map((date, i) => ({ date, history: hVals[start + i] }));
    const hasBand = Array.isArray(best.forecast_lower) && Array.isArray(best.forecast_upper);
    const fc = result.forecast_dates.map((date, i) => ({
      date,
      forecast: best.forecast[i],
      band: hasBand ? [best.forecast_lower[i], best.forecast_upper[i]] : undefined,
    }));
    // Join the two lines at the last actual point.
    if (hist.length) {
      const last = hist[hist.length - 1];
      last.forecast = last.history;
      if (hasBand) last.band = [last.history, last.history];
    }
    return [...hist, ...fc];
  }, [result, best, range]);

  if (!data.length) return null;
  const lastActual = result.historical?.dates?.at(-1);
  const hasBand = Array.isArray(best.forecast_lower);
  const tick = { fontSize: 11, fill: palette.axis };

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-4 px-1 text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4" style={{ background: palette.actual }} /> Actual
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4" style={{ background: palette.forecast }} /> Forecast
        </span>
        {hasBand && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-4 rounded-sm opacity-20" style={{ background: palette.band }} /> 95% range
          </span>
        )}
      </div>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke={palette.grid} vertical={false} />
            <XAxis
              dataKey="date"
              tick={tick}
              tickLine={false}
              axisLine={{ stroke: palette.line }}
              minTickGap={56}
              tickFormatter={fmtShortDate}
            />
            <YAxis tick={tick} tickLine={false} axisLine={false} tickFormatter={fmtCompact} width={52} />
            <Tooltip content={<ChartTooltip palette={palette} />} cursor={{ stroke: palette.line }} />
            {lastActual && (
              <ReferenceLine
                x={lastActual}
                stroke={palette.axis}
                strokeDasharray="3 3"
                label={{ value: "Forecast start", position: "insideTopLeft", fontSize: 10, fill: palette.axis }}
              />
            )}
            {hasBand && <Area dataKey="band" stroke="none" fill={palette.band} fillOpacity={0.12} isAnimationActive={false} />}
            <Line dataKey="history" stroke={palette.actual} strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line dataKey="forecast" stroke={palette.forecast} strokeWidth={2} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
