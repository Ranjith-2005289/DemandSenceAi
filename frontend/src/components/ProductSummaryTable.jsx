import { useEffect, useState } from "react";
import { getProductSummary, runForecast, errorMessage } from "../api";
import { Alert, Badge, Button, Card, CardHeader, Spinner } from "./ui";
import { fmtNumber } from "../lib/format";

const TREND = {
  growing: { label: "Growing", tone: "success", arrow: "▲" },
  declining: { label: "Declining", tone: "danger", arrow: "▼" },
  flat: { label: "Flat", tone: "neutral", arrow: "▬" },
  insufficient_data: { label: "Not enough data", tone: "neutral", arrow: "—" },
};

export default function ProductSummaryTable({
  filename,
  date_col,
  target_col,
  group_col,
  forecast_horizon,
  country,
  sessionId,
  activeGroup,
  onForecastGroup,
}) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forecasting, setForecasting] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getProductSummary(filename, date_col, target_col, group_col, 50)
      .then((data) => !cancelled && setSummary(data))
      .catch((err) => !cancelled && setError(errorMessage(err, "Failed to load the breakdown.")))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [filename, date_col, target_col, group_col]);

  const handleForecast = async (value) => {
    setForecasting(value);
    setError(null);
    try {
      const result = await runForecast(filename, date_col, target_col, forecast_horizon, group_col, value, country || null, sessionId || null);
      onForecastGroup?.(result);
    } catch (err) {
      setError(errorMessage(err, `Forecast failed for ${value}.`));
    } finally {
      setForecasting(null);
    }
  };

  const maxShare = Math.max(1, ...(summary?.groups || []).map((g) => g.share_pct || 0));

  return (
    <Card>
      <CardHeader
        title={`Breakdown by ${group_col}`}
        subtitle="Ranked by total. Trend compares the second half of each history to the first half (±5% = flat)."
        actions={summary && <Badge>{summary.total_groups} total{summary.truncated ? ` · top ${summary.groups.length}` : ""}</Badge>}
      />
      {error && <div className="px-6 pt-4"><Alert>{error}</Alert></div>}
      {loading ? (
        <div className="flex items-center gap-2 px-6 py-8 text-sm text-slate-500"><Spinner /> Loading breakdown…</div>
      ) : summary?.groups?.length ? (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50/70 text-xs font-semibold text-slate-600">
                <th className="px-4 py-2.5 text-left">#</th>
                <th className="px-4 py-2.5 text-left">{group_col}</th>
                <th className="px-4 py-2.5 text-right">Total</th>
                <th className="px-4 py-2.5 text-left">Share</th>
                <th className="px-4 py-2.5 text-left">Trend</th>
                <th className="px-4 py-2.5" />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {summary.groups.map((row) => {
                const t = TREND[row.trend] || TREND.flat;
                const isActive = String(row.group_value) === String(activeGroup);
                return (
                  <tr key={row.group_value} className={isActive ? "bg-brand-50/60" : "hover:bg-slate-50/70"}>
                    <td className="px-4 py-2.5 font-mono text-xs text-slate-400">{row.rank}</td>
                    <td className="px-4 py-2.5 font-medium text-slate-900">{row.group_value}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-xs tabular-nums text-slate-800">{fmtNumber(row.total)}</td>
                    <td className="min-w-[160px] px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 rounded-full bg-slate-100">
                          <div className="h-full rounded-full bg-brand-400" style={{ width: `${(row.share_pct / maxShare) * 100}%` }} />
                        </div>
                        <span className="w-12 text-right font-mono text-xs tabular-nums text-slate-600">{row.share_pct}%</span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5">
                      <Badge tone={t.tone}>{t.arrow} {t.label}</Badge>
                      {row.growth_rate_pct != null && (
                        <span className="ml-2 font-mono text-xs text-slate-500">{row.growth_rate_pct > 0 ? "+" : ""}{row.growth_rate_pct}%</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <Button
                        variant={isActive ? "primary" : "secondary"}
                        size="sm"
                        onClick={() => handleForecast(row.group_value)}
                        disabled={forecasting !== null || row.trend === "insufficient_data"}
                      >
                        {forecasting === row.group_value ? <><Spinner className="h-3 w-3" /> Running…</> : isActive ? "Showing" : "Forecast"}
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="px-6 py-8 text-sm text-slate-500">No groups found.</p>
      )}
    </Card>
  );
}
