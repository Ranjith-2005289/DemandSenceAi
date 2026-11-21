import { useEffect, useState } from "react";
import { Card, Icon, Spinner } from "./ui";
import { MODEL_CATEGORY, modelLabel } from "../lib/format";

// The backend runs as one request, so we can't stream real per-model status.
// Rather than faking checkmarks on a timer (which the previous version did),
// this shows what is genuinely happening plus an honest elapsed-time clock.
const STAGES = [
  { title: "Preprocessing", detail: "Parsing dates, aggregating to a regular frequency, filling gaps, capping outliers." },
  { title: "Training & backtesting", detail: "Every model is fit on your history and re-fit from earlier cut-off points to forecast windows it has not seen." },
  { title: "Ranking & ensembling", detail: "Models are ranked by MASE on those windows; the top three are averaged into an ensemble." },
  { title: "Final forecast", detail: "The winner forecasts your horizon with a 95% interval calibrated on its own backtest errors." },
];

const MODELS = Object.keys(MODEL_CATEGORY).filter((m) => m !== "Ensemble");

export default function StatusTracker({ meta }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const t = setInterval(() => setElapsed((Date.now() - start) / 1000), 250);
    return () => clearInterval(t);
  }, []);

  // Rough stage hint from elapsed time — labelled as an estimate in the UI.
  const active = elapsed < 3 ? 0 : elapsed < 45 ? 1 : elapsed < 70 ? 2 : 3;
  const mm = Math.floor(elapsed / 60);
  const ss = Math.floor(elapsed % 60).toString().padStart(2, "0");

  return (
    <div className="mx-auto max-w-3xl space-y-6 animate-fade-in">
      <Card className="overflow-hidden">
        <div className="h-1 w-full overflow-hidden bg-brand-100">
          <div className="h-full w-2/5 animate-progress rounded-full bg-brand-500" />
        </div>
        <div className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-brand-600">Forecast in progress</p>
            <h2 className="mt-1 text-xl font-semibold text-slate-900">
              Training {MODELS.length} models on <span className="font-mono text-lg">{meta?.target_col}</span>
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              {meta?.filename} · {meta?.forecast_horizon}-period horizon
              {meta?.group_value ? ` · ${meta.group_col} = ${meta.group_value}` : ""}
            </p>
          </div>
          <div className="text-left sm:text-right">
            <p className="font-mono text-3xl font-semibold tabular-nums text-slate-900">{mm}:{ss}</p>
            <p className="text-xs text-slate-500">elapsed · typically 30–120s</p>
          </div>
        </div>
      </Card>

      <Card className="p-6">
        <ol className="space-y-5">
          {STAGES.map((s, i) => {
            const state = i < active ? "done" : i === active ? "active" : "pending";
            return (
              <li key={s.title} className="flex gap-4">
                <span
                  className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                    state === "done" ? "bg-emerald-500 text-white"
                      : state === "active" ? "bg-brand-600 text-white"
                      : "bg-slate-100 text-slate-400"
                  }`}
                >
                  {state === "done" ? <Icon.Check width={14} height={14} /> : state === "active" ? <Spinner className="h-3.5 w-3.5" /> : i + 1}
                </span>
                <div>
                  <p className={`text-sm font-semibold ${state === "pending" ? "text-slate-400" : "text-slate-900"}`}>{s.title}</p>
                  <p className={`mt-0.5 text-sm ${state === "pending" ? "text-slate-400" : "text-slate-600"}`}>{s.detail}</p>
                </div>
              </li>
            );
          })}
        </ol>
        <p className="mt-5 border-t border-slate-100 pt-4 text-xs text-slate-400">
          Stage indicator is an estimate from elapsed time — all models run in parallel in a single request.
        </p>
      </Card>

      <div className="flex flex-wrap justify-center gap-2">
        {MODELS.map((m) => (
          <span key={m} className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600">
            {modelLabel(m)}
          </span>
        ))}
      </div>
    </div>
  );
}
