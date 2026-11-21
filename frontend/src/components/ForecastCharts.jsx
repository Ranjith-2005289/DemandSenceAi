import { useMemo, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Download } from "lucide-react";
import { fmtCompact, fmtNumber, isNum, modelLabel } from "../lib/format";

const COLORS = {
  history: "#64748b",   // slate-500
  forecast: "#2563eb",  // blue-600
  band: "#3b82f6",      // blue-500
  grid: "#eef2f6",
};

const RANGES = [
  { id: 90, label: "90" },
  { id: 180, label: "180" },
  { id: 365, label: "365" },
];

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 font-medium text-slate-900">{label}</p>
      {isNum(p.history) && (
        <p className="text-slate-600"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: COLORS.history }} />Actual: <span className="font-mono">{fmtNumber(p.history)}</span></p>
      )}
      {isNum(p.forecast) && (
        <p className="text-slate-600"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: COLORS.forecast }} />Forecast: <span className="font-mono">{fmtNumber(p.forecast)}</span></p>
      )}
      {Array.isArray(p.band) && (
        <p className="text-slate-500">95% interval: <span className="font-mono">{fmtNumber(p.band[0])} – {fmtNumber(p.band[1])}</span></p>
      )}
    </div>
  );
}

function downloadCsv(results) {
  const b = results.best_model;
  const rows = [["date", "forecast", "lower_95", "upper_95"]];
  results.forecast_dates.forEach((d, i) => {
    rows.push([d, b.forecast[i], b.forecast_lower?.[i] ?? "", b.forecast_upper?.[i] ?? ""]);
  });
  const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `forecast_${results.target_col || "series"}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ForecastCharts({ results }) {
  const [range, setRange] = useState(180);
  const [showTable, setShowTable] = useState(false);
  const best = results?.best_model;

  const data = useMemo(() => {
    if (!best?.forecast?.length) return [];
    const hDates = results.historical?.dates || [];
    const hVals = results.historical?.values || [];
    const start = Math.max(0, hDates.length - range);
    const hist = hDates.slice(start).map((date, i) => ({ date, history: hVals[start + i] }));
    const hasBand = Array.isArray(best.forecast_lower) && Array.isArray(best.forecast_upper);
    const fc = results.forecast_dates.map((date, i) => ({
      date,
      forecast: best.forecast[i],
      band: hasBand ? [best.forecast_lower[i], best.forecast_upper[i]] : undefined,
    }));
    // Join the two lines visually at the last actual point.
    if (hist.length) {
      const last = hist[hist.length - 1];
      last.forecast = last.history;
      if (hasBand) last.band = [last.history, last.history];
    }
    return [...hist, ...fc];
  }, [results, best, range]);

  if (!best?.forecast?.length || !results?.forecast_dates?.length) return null;

  const lastActualDate = results.historical?.dates?.at(-1);
  const hasBand = Array.isArray(best.forecast_lower);
  const importance = best.winner_details?.feature_importance;
  const importanceData = importance
    ? Object.entries(importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([feature, value]) => ({ feature: feature.replaceAll("_", " "), value: Number(value) }))
    : [];

  return (
    <div className="space-y-6">
      <Card className="shadow-sm">
        <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3 px-6 pt-5">
          <div>
            <CardTitle className="text-lg font-semibold text-slate-900">Forecast</CardTitle>
            <p className="mt-1 text-sm text-slate-500">
              {`${modelLabel(best.best_model_name)} · ${results.forecast_dates.length} periods ahead${hasBand ? " · shaded band is the 95% interval" : ""}`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-lg bg-slate-100 p-0.5 text-xs font-medium">
              {RANGES.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  onClick={() => setRange(r.id)}
                  className={`rounded-md px-2.5 py-1 transition ${range === r.id ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}
                >
                  {r.label}
                </button>
              ))}
            </div>
            <span className="hidden text-xs text-slate-400 sm:inline">periods of history</span>
          </div>
        </CardHeader>
        <div className="h-[380px] px-2 pb-4 pt-4 sm:px-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 0 }}>
              <CartesianGrid stroke={COLORS.grid} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={{ stroke: "#e2e8f0" }} minTickGap={48} />
              <YAxis tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} tickFormatter={fmtCompact} width={56} />
              <Tooltip content={<ChartTooltip />} />
              <Legend verticalAlign="top" align="right" height={28} iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
              {lastActualDate && <ReferenceLine x={lastActualDate} stroke="#cbd5e1" strokeDasharray="4 4" label={{ value: "Today", position: "insideTopRight", fontSize: 10, fill: "#94a3b8" }} />}
              {hasBand && <Area dataKey="band" name="95% interval" stroke="none" fill={COLORS.band} fillOpacity={0.14} isAnimationActive={false} legendType="square" />}
              <Line dataKey="history" name="Actual" stroke={COLORS.history} strokeWidth={1.6} dot={false} isAnimationActive={false} connectNulls={false} />
              <Line dataKey="forecast" name="Forecast" stroke={COLORS.forecast} strokeWidth={2.4} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 px-6 py-3">
          <button type="button" onClick={() => setShowTable((v) => !v)} className="text-xs font-medium text-slate-600 hover:text-slate-900">
            {showTable ? "Hide" : "Show"} forecast table
          </button>
          <Button variant="outline" size="sm" onClick={() => downloadCsv(results)} className="gap-1.5">
            <Download className="h-3.5 w-3.5" /> Download CSV
          </Button>
        </div>
        {showTable && (
          <div className="max-h-80 overflow-y-auto border-t border-slate-100">
            <table className="min-w-full text-sm">
              <thead className="sticky top-0 bg-slate-50">
                <tr className="text-xs font-semibold text-slate-600">
                  <th className="px-6 py-2 text-left">Date</th>
                  <th className="px-6 py-2 text-right">Forecast</th>
                  <th className="px-6 py-2 text-right">Lower 95%</th>
                  <th className="px-6 py-2 text-right">Upper 95%</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-xs tabular-nums">
                {results.forecast_dates.map((d, i) => (
                  <tr key={d}>
                    <td className="px-6 py-1.5 text-slate-700">{d}</td>
                    <td className="px-6 py-1.5 text-right text-slate-900">{fmtNumber(best.forecast[i])}</td>
                    <td className="px-6 py-1.5 text-right text-slate-500">{fmtNumber(best.forecast_lower?.[i])}</td>
                    <td className="px-6 py-1.5 text-right text-slate-500">{fmtNumber(best.forecast_upper?.[i])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {importanceData.length > 0 && (
        <Card className="shadow-sm">
          <CardHeader className="px-6 pt-5">
            <CardTitle className="text-lg font-semibold text-slate-900">What drives the forecast</CardTitle>
            <p className="text-sm text-slate-500">{`Top features by importance in ${modelLabel(best.best_model_name)}`}</p>
          </CardHeader>
          <div className="px-4 py-4" style={{ height: importanceData.length * 34 + 40 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importanceData} layout="vertical" margin={{ top: 0, right: 24, left: 8, bottom: 0 }} barSize={14}>
                <CartesianGrid stroke={COLORS.grid} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 12, fill: "#334155" }} tickLine={false} axisLine={false} />
                <Tooltip formatter={(v) => [v.toFixed(4), "Importance"]} cursor={{ fill: "rgba(99,102,241,0.06)" }} />
                <Bar dataKey="value" fill={COLORS.forecast} radius={[0, 4, 4, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  );
}
