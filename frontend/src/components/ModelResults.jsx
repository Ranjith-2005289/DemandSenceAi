import { useMemo, useState } from "react";
import { Badge, Card, CardHeader } from "./ui";
import { MODEL_CATEGORY, fmtNumber, fmtPct, fmtSeconds, isNum, modelLabel } from "../lib/format";

const COLUMNS = [
  { key: "rank", label: "#", align: "left" },
  { key: "model_name", label: "Model", align: "left" },
  { key: "mase", label: "MASE", align: "left", hint: "Mean absolute scaled error on the backtest windows. Below 1.00 beats a seasonal-naive forecast." },
  { key: "rmse", label: "RMSE", align: "right" },
  { key: "mae", label: "MAE", align: "right" },
  { key: "mape", label: "MAPE", align: "right" },
  { key: "time", label: "Train time", align: "right" },
];

export default function ModelResults({ results }) {
  const [sort, setSort] = useState({ key: "rank", dir: 1 });
  const [showFailed, setShowFailed] = useState(false);

  const all = results?.all_models || [];
  const bestName = results?.best_model?.best_model_name;

  const rows = useMemo(() => {
    const ranked = all.map((m, i) => ({
      ...m,
      rank: i + 1,
      time: isNum(m.train_time_sec) ? m.train_time_sec : m.elapsed_sec,
    }));
    const val = (r) => (sort.key === "model_name" ? r.model_name : isNum(r[sort.key]) ? r[sort.key] : Infinity);
    return ranked.sort((a, b) => {
      const va = val(a), vb = val(b);
      if (typeof va === "string") return va.localeCompare(vb) * sort.dir;
      return (va - vb) * sort.dir;
    });
  }, [all, sort]);

  if (!all.length) return null;

  const ok = rows.filter((r) => r.status === "success");
  const failed = rows.filter((r) => r.status !== "success");
  const maxMase = Math.max(1.5, ...ok.map((r) => (isNum(r.mase) ? r.mase : 0)));
  const backtested = results?.model_run_info?.evaluation === "holdout_backtest";

  const toggleSort = (key) =>
    setSort((s) => (s.key === key ? { key, dir: -s.dir } : { key, dir: 1 }));

  return (
    <Card>
      <CardHeader
        title="Model leaderboard"
        subtitle={
          backtested
            ? `All models scored on the same ${results.model_run_info.backtest_origins} held-out window(s) of ${results.model_run_info.backtest_horizon} periods.`
            : "Series too short for a common backtest — ranked by each model's internal cross-validation."
        }
        actions={
          <>
            <Badge tone="success">{ok.length} succeeded</Badge>
            {failed.length > 0 && <Badge tone="danger">{failed.length} failed</Badge>}
          </>
        }
      />
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50/70">
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  title={c.hint}
                  className={`whitespace-nowrap px-4 py-2.5 text-xs font-semibold text-slate-600 ${c.align === "right" ? "text-right" : "text-left"}`}
                >
                  <button type="button" onClick={() => toggleSort(c.key)} className="inline-flex items-center gap-1 hover:text-slate-900">
                    {c.label}
                    {sort.key === c.key && <span className="text-slate-400">{sort.dir === 1 ? "↑" : "↓"}</span>}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {ok.map((m) => {
              const isBest = m.model_name === bestName;
              const pct = isNum(m.mase) ? Math.min(100, (m.mase / maxMase) * 100) : 0;
              const naivePct = (1 / maxMase) * 100;
              return (
                <tr key={m.model_name} className={isBest ? "bg-brand-50/60" : "hover:bg-slate-50/70"}>
                  <td className="px-4 py-3 font-mono text-xs text-slate-400">{m.rank}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className={`font-medium ${isBest ? "text-brand-800" : "text-slate-900"}`}>{modelLabel(m.model_name)}</span>
                      {isBest && <Badge tone="brand">Selected</Badge>}
                    </div>
                    <p className="mt-0.5 text-xs text-slate-500">
                      {m.ensemble_members?.length
                        ? `Average of ${m.ensemble_members.map(modelLabel).join(", ")}`
                        : MODEL_CATEGORY[m.model_name] || "Model"}
                    </p>
                  </td>
                  <td className="min-w-[180px] px-4 py-3">
                    {isNum(m.mase) ? (
                      <div className="flex items-center gap-3">
                        <div className="relative h-2 flex-1 rounded-full bg-slate-100">
                          <div
                            className={`absolute inset-y-0 left-0 rounded-full ${m.mase < 1 ? (isBest ? "bg-brand-500" : "bg-brand-300") : "bg-slate-300"}`}
                            style={{ width: `${pct}%` }}
                          />
                          <div className="absolute -top-1 h-4 w-px bg-slate-400" style={{ left: `${naivePct}%` }} title="Seasonal naive = 1.00" />
                        </div>
                        <span className={`w-12 text-right font-mono text-xs tabular-nums ${m.mase < 1 ? "text-slate-900" : "text-slate-500"}`}>
                          {m.mase.toFixed(3)}
                        </span>
                      </div>
                    ) : (
                      <span className="text-xs text-slate-400">n/a</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs tabular-nums text-slate-700">{fmtNumber(m.rmse)}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs tabular-nums text-slate-700">{fmtNumber(m.mae)}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs tabular-nums text-slate-700">{fmtPct(m.mape)}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs tabular-nums text-slate-500">{fmtSeconds(m.time)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {failed.length > 0 && (
        <div className="border-t border-slate-100 px-6 py-3">
          <button type="button" onClick={() => setShowFailed((v) => !v)} className="text-xs font-medium text-slate-600 hover:text-slate-900">
            {showFailed ? "Hide" : "Show"} {failed.length} model{failed.length > 1 ? "s" : ""} that could not run
          </button>
          {showFailed && (
            <ul className="mt-2 space-y-1">
              {failed.map((m) => (
                <li key={m.model_name} className="text-xs text-slate-600">
                  <span className="font-medium text-slate-800">{modelLabel(m.model_name)}</span>
                  {" — "}{m.error || m.status}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      <p className="border-t border-slate-100 px-6 py-3 text-xs text-slate-500">
        MASE below 1.00 (left of the marker) means the model beat simply repeating the last season.
        Click a column header to sort.
      </p>
    </Card>
  );
}
