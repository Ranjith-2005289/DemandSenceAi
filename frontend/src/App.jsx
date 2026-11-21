import { useMemo, useState } from "react";
import UploadSection from "./components/UploadSection";
import StatusTracker from "./components/StatusTracker";
import ModelResults from "./components/ModelResults";
import ForecastCharts from "./components/ForecastCharts";
import ChatBot from "./components/ChatBot";
import ProductSummaryTable from "./components/ProductSummaryTable";
import InsightsPage from "./components/InsightsPage";
import { Alert, Badge, Button, Icon, Stat } from "./components/ui";
import { runForecast, errorMessage } from "./api";
import { fmtNumber, isNum, modelLabel } from "./lib/format";

const STEPS = [
  { id: "upload", label: "Upload" },
  { id: "configure", label: "Configure" },
  { id: "results", label: "Forecast" },
  { id: "insights", label: "Insights" },
];

function Stepper({ current }) {
  const order = { upload: 0, configure: 1, forecasting: 1, results: 2, insights: 3 };
  const idx = order[current] ?? 0;
  return (
    <ol className="hidden items-center gap-2 md:flex">
      {STEPS.map((s, i) => (
        <li key={s.id} className="flex items-center gap-2">
          <span
            className={`flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-semibold ${
              i < idx ? "bg-brand-600 text-white" : i === idx ? "bg-white text-brand-700 ring-2 ring-brand-500" : "bg-slate-800 text-slate-400"
            }`}
          >
            {i < idx ? <Icon.Check width={12} height={12} /> : i + 1}
          </span>
          <span className={`text-sm ${i <= idx ? "font-medium text-white" : "text-slate-400"}`}>{s.label}</span>
          {i < STEPS.length - 1 && <span className="mx-1 h-px w-6 bg-slate-700" />}
        </li>
      ))}
    </ol>
  );
}

export function ResultsSummary({ results, config }) {
  const best = results.best_model || {};
  const all = results.all_models || [];
  const naive = all.find((m) => m.model_name === "SeasonalNaive");
  const fc = best.forecast || [];
  const hist = results.historical?.values || [];

  const stats = useMemo(() => {
    const fcMean = fc.length ? fc.reduce((a, b) => a + b, 0) / fc.length : NaN;
    const fcTotal = fc.reduce((a, b) => a + b, 0);
    const recent = hist.slice(-fc.length);
    const recentMean = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : NaN;
    const change = isNum(fcMean) && isNum(recentMean) && recentMean !== 0 ? fcMean / recentMean - 1 : NaN;
    const gain = isNum(best.mase) && isNum(naive?.mase) && naive.mase > 0 ? 1 - best.mase / naive.mase : NaN;
    return { fcMean, fcTotal, change, gain };
  }, [fc, hist, best.mase, naive?.mase]);

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Stat
        label="Selected model"
        icon={<Icon.Trophy />}
        value={modelLabel(best.best_model_name)}
        sub={best.ensemble_members?.length ? `Avg. of ${best.ensemble_members.map(modelLabel).join(", ")}` : `Ranked #1 of ${all.filter((m) => m.status === "success").length}`}
      />
      <Stat
        label="Backtest error (MASE)"
        icon={<Icon.Target />}
        value={isNum(best.mase) ? best.mase.toFixed(3) : fmtNumber(best.rmse)}
        tone={isNum(stats.gain) ? (stats.gain > 0 ? "good" : "bad") : undefined}
        sub={
          isNum(stats.gain)
            ? `${Math.abs(stats.gain * 100).toFixed(0)}% ${stats.gain > 0 ? "more accurate than" : "less accurate than"} seasonal naive`
            : isNum(best.mase) ? "Lower is better · <1 beats naive" : "RMSE (internal CV)"
        }
      />
      <Stat
        label={`Forecast total · ${fc.length} periods`}
        icon={<Icon.Chart />}
        value={fmtNumber(stats.fcTotal)}
        sub={`≈ ${fmtNumber(stats.fcMean)} per period`}
      />
      <Stat
        label="vs. recent actuals"
        icon={<Icon.Calendar />}
        value={isNum(stats.change) ? `${stats.change > 0 ? "+" : ""}${(stats.change * 100).toFixed(1)}%` : "—"}
        tone={isNum(stats.change) ? (stats.change >= 0 ? "good" : "bad") : undefined}
        sub={`Forecast avg. vs. last ${fc.length} actual periods of ${config?.target_col || "the target"}`}
      />
    </div>
  );
}

export default function App() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [step, setStep] = useState("upload"); // upload | configure | forecasting | results | insights
  const [uploadData, setUploadData] = useState(null);
  const [fileLabel, setFileLabel] = useState("");
  const [config, setConfig] = useState(null);
  const [runError, setRunError] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);
  const [aggregateResult, setAggregateResult] = useState(null);
  const [runMeta, setRunMeta] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false);

  const savedAs = uploadData?.file?.saved_as;

  const handleUploaded = (data, name) => {
    setUploadData(data);
    setConfig(null);
    setFileLabel(name);
    setRunError(null);
    setStep("configure");
  };

  const handleReset = () => {
    setUploadData(null);
    setConfig(null);
    setForecastResult(null);
    setAggregateResult(null);
    setRunError(null);
    setIsChatOpen(false);
    setStep("upload");
  };

  const handleRun = async (cfg) => {
    setConfig(cfg);
    setRunError(null);
    setRunMeta({ ...cfg, filename: fileLabel });
    setStep("forecasting");
    try {
      const result = await runForecast(
        savedAs, cfg.date_col, cfg.target_col, cfg.forecast_horizon,
        null, null, cfg.country, sessionId
      );
      setForecastResult(result);
      setAggregateResult(result);
      setStep("results");
    } catch (err) {
      setRunError(errorMessage(err, "Forecast failed."));
      setStep("configure");
    }
  };

  const handleGroupForecast = (result) => {
    setForecastResult(result);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(forecastResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "forecast_results.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-6 px-4 sm:px-6">
          <button type="button" onClick={handleReset} className="flex items-center gap-2.5" title="Start over">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white">
              <Icon.Logo width={18} height={18} />
            </span>
            <span className="text-left leading-tight">
              <span className="block text-sm font-semibold text-white">DemandSense AI</span>
              <span className="block text-[11px] text-slate-400">Demand forecasting platform</span>
            </span>
          </button>
          <Stepper current={step} />
          <div className="flex items-center gap-2">
            {step === "results" && (
              <Button variant="dark" size="sm" onClick={() => setIsChatOpen(true)} className="bg-slate-800 hover:bg-slate-700">
                <Icon.Chat width={16} height={16} /> Ask AI
              </Button>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:py-10">
          {(step === "upload" || step === "configure") && (
            <UploadSection
              uploadData={uploadData}
              initialConfig={config}
              onUploaded={handleUploaded}
              onRun={handleRun}
              onReset={handleReset}
              error={runError}
            />
          )}

          {step === "forecasting" && <StatusTracker meta={runMeta} />}

          {step === "results" && forecastResult && (
            <div className="space-y-6 animate-fade-in">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="success"><Icon.Check width={12} height={12} /> Forecast ready</Badge>
                    {forecastResult.model_run_info?.evaluation === "holdout_backtest" && <Badge tone="brand">Backtested</Badge>}
                    {forecastResult.group && (
                      <Badge tone="warning">{forecastResult.group.group_col} = {forecastResult.group.group_value}</Badge>
                    )}
                  </div>
                  <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">
                    {forecastResult.target_col || config?.target_col} forecast
                  </h1>
                  <p className="mt-1 text-sm text-slate-500">
                    {fileLabel} · {fmtNumber(forecastResult.model_run_info?.row_count)} {forecastResult.model_run_info?.frequency_tier || ""} observations · {config?.forecast_horizon}-period horizon
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {forecastResult.group && aggregateResult && (
                    <Button variant="secondary" onClick={() => setForecastResult(aggregateResult)}>
                      <Icon.ArrowLeft width={16} height={16} /> All data
                    </Button>
                  )}
                  <Button variant="secondary" onClick={() => setStep("configure")}>
                    <Icon.Refresh width={16} height={16} /> Re-configure
                  </Button>
                  <Button variant="secondary" onClick={exportJson}>
                    <Icon.Download width={16} height={16} /> Export JSON
                  </Button>
                  <Button onClick={() => setStep("insights")}>
                    <Icon.Sparkles width={16} height={16} /> AI insights report
                  </Button>
                </div>
              </div>

              {forecastResult.model_run_info?.training_window_truncated && (
                <Alert tone="info">
                  Models were trained on the most recent {fmtNumber(forecastResult.model_run_info.row_count)} of{" "}
                  {fmtNumber(forecastResult.model_run_info.original_row_count)} observations — recent history drives a
                  short-horizon forecast, and this keeps training fast.
                </Alert>
              )}

              {forecastResult.best_model?.error && (
                <Alert title="No usable forecast">{forecastResult.best_model.error}</Alert>
              )}

              <ResultsSummary results={forecastResult} config={config} />
              <ForecastCharts results={forecastResult} />
              <ModelResults results={forecastResult} />

              {config?.group_col && (
                <ProductSummaryTable
                  filename={savedAs}
                  date_col={config.date_col}
                  target_col={config.target_col}
                  group_col={config.group_col}
                  forecast_horizon={config.forecast_horizon}
                  country={config.country}
                  sessionId={sessionId}
                  activeGroup={forecastResult.group?.group_value}
                  onForecastGroup={handleGroupForecast}
                />
              )}
            </div>
          )}

          {step === "insights" && aggregateResult && config && (
            <InsightsPage
              filename={savedAs}
              date_col={config.date_col}
              target_col={config.target_col}
              group_col={config.group_col}
              forecast_result={aggregateResult}
              onBack={() => setStep("results")}
            />
          )}
        </div>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-1 px-4 py-4 text-xs text-slate-500 sm:flex-row sm:justify-between sm:px-6">
          <p>© 2026 DemandSense AI · Capstone project</p>
          <p>Forecasts are statistical estimates — review intervals before making inventory decisions.</p>
        </div>
      </footer>

      {step === "results" && !isChatOpen && (
        <button
          type="button"
          onClick={() => setIsChatOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex h-14 w-14 items-center justify-center rounded-full bg-brand-600 text-white shadow-lg transition hover:bg-brand-700"
          title="Ask the AI assistant"
        >
          <Icon.Chat />
        </button>
      )}

      <ChatBot
        forecastResult={forecastResult}
        isVisible={isChatOpen && step === "results"}
        onClose={() => setIsChatOpen(false)}
        groupInfo={
          config?.group_col
            ? { filename: savedAs, date_col: config.date_col, target_col: config.target_col, group_col: config.group_col }
            : null
        }
      />
    </div>
  );
}
