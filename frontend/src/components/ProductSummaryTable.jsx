import { useEffect, useState } from "react";
import { getProductSummary, runForecast } from "../api";

const TREND_STYLES = {
  growing: { label: "▲ Growing", className: "bg-green-100 text-green-800" },
  declining: { label: "▼ Declining", className: "bg-red-100 text-red-800" },
  flat: { label: "▬ Flat", className: "bg-slate-100 text-slate-700" },
  insufficient_data: { label: "— Not enough data", className: "bg-slate-100 text-slate-500" },
};

export default function ProductSummaryTable({
  filename,
  date_col,
  target_col,
  group_col,
  forecast_horizon,
  country,
  sessionId,
  onForecastGroup,
}) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forecastingGroup, setForecastingGroup] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getProductSummary(filename, date_col, target_col, group_col, 50)
      .then((data) => {
        if (!cancelled) setSummary(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err?.response?.data?.detail || err?.message || "Failed to load product summary.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [filename, date_col, target_col, group_col]);

  const handleForecastGroup = async (groupValue) => {
    setForecastingGroup(groupValue);
    try {
      const result = await runForecast(
        filename,
        date_col,
        target_col,
        forecast_horizon,
        group_col,
        groupValue,
        country || null,
        sessionId || null
      );
      onForecastGroup?.(result);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Forecast failed for this group.");
    } finally {
      setForecastingGroup(null);
    }
  };

  if (loading) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-600 shadow-sm">
        Loading {group_col} breakdown...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-300 bg-red-50 p-4 text-sm text-red-800 shadow-sm">
        {error}
      </div>
    );
  }

  if (!summary?.groups?.length) return null;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold text-slate-900">
          Sales by {group_col}
        </h3>
        <p className="text-xs text-slate-600">
          {summary.total_groups} total
          {summary.truncated ? ` (showing top ${summary.groups.length})` : ""}
        </p>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs font-semibold uppercase text-slate-600">
              <th className="py-2 pr-3">Rank</th>
              <th className="py-2 pr-3">{group_col}</th>
              <th className="py-2 pr-3">Total</th>
              <th className="py-2 pr-3">Share</th>
              <th className="py-2 pr-3">Trend</th>
              <th className="py-2 pr-3"></th>
            </tr>
          </thead>
          <tbody>
            {summary.groups.map((row) => {
              const trend = TREND_STYLES[row.trend] || TREND_STYLES.flat;
              return (
                <tr key={row.group_value} className="border-b border-slate-100">
                  <td className="py-2 pr-3 text-slate-600">{row.rank}</td>
                  <td className="py-2 pr-3 font-medium text-slate-900">{row.group_value}</td>
                  <td className="py-2 pr-3 font-mono text-slate-900">
                    {row.total.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className="py-2 pr-3 text-slate-700">{row.share_pct}%</td>
                  <td className="py-2 pr-3">
                    <span className={`rounded px-2 py-1 text-xs font-semibold ${trend.className}`}>
                      {trend.label}
                    </span>
                    {row.growth_rate_pct != null && (
                      <span className="ml-2 text-xs text-slate-500">
                        {row.growth_rate_pct > 0 ? "+" : ""}
                        {row.growth_rate_pct}%
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-3">
                    <button
                      onClick={() => handleForecastGroup(row.group_value)}
                      disabled={forecastingGroup !== null || row.trend === "insufficient_data"}
                      className="rounded-md border border-blue-300 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-900 transition hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {forecastingGroup === row.group_value ? "Forecasting..." : "Forecast this one"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
