import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Check, Loader2 } from "lucide-react";
import { MODEL_CATEGORY, modelLabel } from "../lib/format";

// The forecast runs as a single backend request, so real per-model progress
// can't be streamed. Instead of ticking fake checkmarks on a timer, this shows
// what the pipeline is doing plus an honest elapsed-time clock. The active
// stage is a rough estimate from elapsed time.
const STAGES = [
  { title: "Preprocessing data", detail: "Parsing dates, regularising the frequency, filling gaps, capping outliers." },
  { title: "Training & backtesting models", detail: "Each model is fit on your history and re-fit from earlier cut-off points to forecast windows it hasn't seen." },
  { title: "Ranking & ensembling", detail: "Models are ranked by MASE on those windows; the top three are averaged into an ensemble." },
  { title: "Generating forecast", detail: "The winner forecasts your horizon with a 95% interval calibrated on its own backtest errors." },
];

const MODELS = Object.keys(MODEL_CATEGORY).filter((m) => m !== "Ensemble");

export default function StatusTracker() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const t = setInterval(() => setElapsed((Date.now() - start) / 1000), 250);
    return () => clearInterval(t);
  }, []);

  const active = elapsed < 3 ? 0 : elapsed < 40 ? 1 : elapsed < 60 ? 2 : 3;
  const mm = Math.floor(elapsed / 60);
  const ss = Math.floor(elapsed % 60).toString().padStart(2, "0");

  return (
    <Card className="shadow-sm border border-slate-200 overflow-hidden">
      <div className="flex items-center justify-between px-7 pt-6 pb-5 border-b border-slate-100 bg-white">
        <div>
          <p className="text-xs font-bold uppercase tracking-widest text-slate-400">Status</p>
          <p className="mt-1 text-sm font-semibold text-slate-900">Training {MODELS.length} models + ensemble</p>
        </div>
        <div className="text-right">
          <p className="font-mono text-2xl font-semibold tabular-nums text-slate-900">{mm}:{ss}</p>
          <p className="text-xs text-slate-500">elapsed · usually 15–90s</p>
        </div>
      </div>

      <CardContent className="px-7 py-6 space-y-5 bg-white">
        {STAGES.map((s, i) => {
          const state = i < active ? "done" : i === active ? "active" : "pending";
          return (
            <div key={s.title} className="flex items-start gap-3">
              <div
                className={`mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full border-2 text-xs font-semibold ${
                  state === "done"
                    ? "border-green-500 bg-green-500 text-white"
                    : state === "active"
                    ? "border-blue-500 bg-blue-50 text-blue-600"
                    : "border-slate-300 bg-slate-100 text-slate-400"
                }`}
              >
                {state === "done" ? <Check className="h-4 w-4" /> : state === "active" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : i + 1}
              </div>
              <div>
                <p className={`text-sm font-medium ${state === "pending" ? "text-slate-400" : state === "done" ? "text-green-700" : "text-slate-900"}`}>
                  {s.title}
                </p>
                <p className="mt-0.5 text-xs text-slate-500">{s.detail}</p>
              </div>
            </div>
          );
        })}
      </CardContent>

      <div className="px-7 py-5 border-t border-slate-100 bg-slate-50">
        <p className="text-xs font-medium uppercase text-slate-500">Models in this run</p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {MODELS.map((m) => (
            <span key={m} className="rounded-md border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-600">
              {modelLabel(m)}
            </span>
          ))}
        </div>
      </div>
    </Card>
  );
}
