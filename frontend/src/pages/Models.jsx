import { useState } from "react";
import PageHeader, { PageBody } from "@/components/shell/PageHeader";
import { Chip, Panel, PanelFooter, PanelHeader, Segmented } from "@/components/ui";
import { useAppState } from "@/state/AppState";
import { forecastSummary } from "@/lib/insights";
import { MODEL_CATEGORY, fmtNumber, fmtPct, fmtSeconds, isNum, modelLabel } from "@/lib/format";

// Rows arrive from the backend already ranked by MASE on a common holdout
// backtest. MASE below 1 beats a seasonal-naive forecast.
function MaseBar({ value, max }) {
  if (!isNum(value)) return <span className="text-ink-3">—</span>;
  const w = Math.min(100, (value / max) * 100);
  const naive = Math.min(100, (1 / max) * 100);
  return (
    <div className="flex items-center justify-end gap-2.5">
      <div className="relative hidden h-1.5 w-24 rounded-full bg-surface-muted md:block">
        <div className={`h-full rounded-full ${value < 1 ? "bg-ok" : "bg-warn"}`} style={{ width: `${w}%` }} />
        <div className="absolute -top-0.5 h-2.5 w-px bg-ink-3" style={{ left: `${naive}%` }} title="Seasonal-naive baseline (1.00)" />
      </div>
      <span className={`w-12 tabular-nums ${value < 1 ? "font-medium text-ok" : ""}`}>{value.toFixed(3)}</span>
    </div>
  );
}

export default function ModelsPage() {
  const { view } = useAppState();
  const [filter, setFilter] = useState("all");
  if (!view) return null;

  const s = forecastSummary(view);
  const info = view.model_run_info || {};
  const backtested = info.evaluation === "holdout_backtest";
  const bestName = s.best?.best_model_name;
  const maxMase = Math.max(1.5, ...s.ok.map((m) => (isNum(m.mase) ? m.mase : 0)));
  const rows = filter === "ok" ? s.ok : filter === "failed" ? s.failed : [...s.ok, ...s.failed];

  return (
    <>
      <PageHeader
        title="Model comparison"
        description={
          backtested
            ? `Every model forecast the same ${info.backtest_origins} held-out window${info.backtest_origins > 1 ? "s" : ""} of ${info.backtest_horizon} periods and is ranked by MASE.`
            : "The series is too short for a common backtest, so models are ranked by their own cross-validation error."
        }
        meta={[
          ["Models run", s.models.length],
          ["Succeeded", s.ok.length],
          ["Beat naive baseline", s.beatNaive.length],
          ["Selected", modelLabel(bestName)],
          s.best?.ensemble_members?.length > 0 && ["Ensemble of", s.best.ensemble_members.map(modelLabel).join(", ")],
        ]}
      />
      <PageBody>
        <Panel>
          <PanelHeader
            title="Leaderboard"
            subtitle={view.group ? `${view.group.group_col}: ${view.group.group_value}` : "All data"}
            actions={
              <Segmented
                value={filter}
                onChange={setFilter}
                options={[
                  { value: "all", label: `All (${s.models.length})` },
                  { value: "ok", label: `Succeeded (${s.ok.length})` },
                  { value: "failed", label: `Failed (${s.failed.length})` },
                ]}
              />
            }
          />
          <div className="scroll-thin overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="w-10">Rank</th>
                  <th>Model</th>
                  <th>Family</th>
                  <th className="num" title="Mean absolute scaled error on the backtest windows">
                    MASE
                  </th>
                  <th className="num">RMSE</th>
                  <th className="num">MAE</th>
                  <th className="num">MAPE</th>
                  <th className="num">Train time</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((m) => {
                  const ok = m.status === "success";
                  const rank = ok ? s.ok.indexOf(m) + 1 : null;
                  const isBest = m.model_name === bestName;
                  const time = isNum(m.train_time_sec) ? m.train_time_sec : m.elapsed_sec;
                  return (
                    <tr key={m.model_name} className={isBest ? "[&>td]:bg-brand-soft" : ""}>
                      <td className="text-ink-3">{rank ?? "—"}</td>
                      <td>
                        <span className="font-medium">{modelLabel(m.model_name)}</span>
                        {!ok && m.error && (
                          <p className="mt-0.5 max-w-sm truncate text-xs text-bad" title={m.error}>
                            {m.error}
                          </p>
                        )}
                      </td>
                      <td className="text-ink-2">{MODEL_CATEGORY[m.model_name] || "—"}</td>
                      <td className="num">{ok ? <MaseBar value={m.mase} max={maxMase} /> : "—"}</td>
                      <td className="num">{fmtNumber(m.rmse)}</td>
                      <td className="num">{fmtNumber(m.mae)}</td>
                      <td className="num">{fmtPct(m.mape)}</td>
                      <td className="num text-ink-2">{fmtSeconds(time)}</td>
                      <td>
                        {isBest ? <Chip tone="brand">Selected</Chip> : ok ? <Chip tone="ok">Succeeded</Chip> : <Chip tone="bad">Failed</Chip>}
                      </td>
                    </tr>
                  );
                })}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={9} className="py-8 text-center text-ink-3">
                      No models in this view.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <PanelFooter>
            <p className="text-xs text-ink-3">
              MASE below 1.00 means the model beat a seasonal-naive forecast on the held-out windows. The marker on each bar is that 1.00 baseline.
            </p>
          </PanelFooter>
        </Panel>
      </PageBody>
    </>
  );
}
