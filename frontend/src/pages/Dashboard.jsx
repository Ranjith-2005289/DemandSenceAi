import { useState } from "react";
import { AlertTriangle, ArrowRight, CheckCircle2, FilePlus2, Info, Loader2, XCircle } from "lucide-react";
import PageHeader, { PageBody } from "@/components/shell/PageHeader";
import ForecastChart from "@/components/ForecastChart";
import { Banner, Button, Chip, EmptyState, Facts, Kpi, Panel, PanelBody, PanelHeader, Segmented } from "@/components/ui";
import { navigate } from "@/lib/router";
import { useAppState } from "@/state/AppState";
import { attentionItems, forecastSummary } from "@/lib/insights";
import { MODEL_CATEGORY, MODEL_COUNT, countryLabel, fmtCompact, fmtShare, fmtDate, fmtNumber, fmtSeconds, fmtSignedPct, freqLabel, isNum, modelLabel } from "@/lib/format";
import { TrendChip } from "./Products";

const RANGES = [
  { value: 90, label: "90" },
  { value: 180, label: "180" },
  { value: 365, label: "365" },
];

function NoRun() {
  const { run } = useAppState();
  return (
    <>
      <PageHeader title="Dashboard" description="Overview of the current forecast run." />
      <PageBody>
        {run.status === "failed" && (
          <Banner
            tone="bad"
            title="The last forecast run failed"
            action={
              <Button size="sm" onClick={() => navigate("new")}>
                Review and retry
              </Button>
            }
          >
            {run.error}
          </Banner>
        )}
        <Panel>
          {run.status === "running" ? (
            <EmptyState
              icon={Loader2}
              title="Forecast in progress"
              action={
                <Button variant="secondary" onClick={() => navigate("new")}>
                  View progress
                </Button>
              }
            >
              {MODEL_COUNT} models are being trained and backtested. Results will appear here when the run finishes.
            </EmptyState>
          ) : (
            <EmptyState
              icon={FilePlus2}
              title="No forecast yet"
              action={
                <Button variant="primary" icon={FilePlus2} onClick={() => navigate("new")}>
                  New forecast
                </Button>
              }
            >
              Upload a sales or demand history to train and compare {MODEL_COUNT} forecasting models. The forecast, model comparison and insights
              report will appear here.
            </EmptyState>
          )}
        </Panel>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            ["Accepted files", "CSV, Excel (.xlsx, .xlsm) or JSON records, one row per date (and per product or store, if any)."],
            ["Required columns", "A date column and a numeric column to forecast, such as units sold or revenue."],
            ["Recommended history", "At least two full seasonal cycles, for example two years of daily or weekly data."],
          ].map(([t, d]) => (
            <Panel key={t} className="px-4 py-3">
              <p className="text-[13px] font-semibold text-ink">{t}</p>
              <p className="mt-1 text-xs text-ink-3">{d}</p>
            </Panel>
          ))}
        </div>
      </PageBody>
    </>
  );
}

const ATTENTION_ICON = { bad: XCircle, warn: AlertTriangle, info: Info };
const ATTENTION_COLOR = { bad: "text-bad", warn: "text-warn", info: "text-brand" };

function Attention({ items }) {
  return (
    <Panel className="flex flex-col">
      <PanelHeader
        title="Needs attention"
        actions={items.length > 0 && <Chip tone={items.some((i) => i.tone === "bad") ? "bad" : "warn"}>{items.length}</Chip>}
      />
      {items.length === 0 ? (
        <div className="flex flex-1 items-center gap-2.5 px-4 py-6 text-[13px] text-ink-2">
          <CheckCircle2 className="h-4 w-4 text-ok" /> Nothing needs attention for this run.
        </div>
      ) : (
        <ul className="scroll-thin max-h-[360px] flex-1 divide-y divide-line overflow-y-auto">
          {items.map((it, i) => {
            const Icon = ATTENTION_ICON[it.tone];
            return (
              <li key={i} className="flex gap-2.5 px-4 py-3">
                <Icon className={`mt-px h-4 w-4 shrink-0 ${ATTENTION_COLOR[it.tone]}`} />
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] font-medium text-ink">{it.title}</p>
                  <p className="mt-0.5 text-xs text-ink-3">{it.detail}</p>
                  {it.route && (
                    <button
                      type="button"
                      onClick={() => navigate(it.route)}
                      className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-brand-ink hover:underline"
                    >
                      {it.action} <ArrowRight className="h-3 w-3" />
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}

export default function DashboardPage() {
  const { result, runMeta, run, products, report } = useAppState();
  const [range, setRange] = useState(180);
  if (!result) return <NoRun />;

  const s = forecastSummary(result);
  const mase = s.best?.mase;
  const items = attentionItems({ result, runMeta, products: products.data, report: report.data });
  const info = result.model_run_info || {};
  const duration = runMeta.finishedAt && runMeta.startedAt ? (runMeta.finishedAt - runMeta.startedAt) / 1000 : null;
  const topModels = s.ok.slice(0, 5);
  const topGroups = (products.data?.groups || []).slice(0, 5);

  return (
    <>
      <PageHeader
        title="Dashboard"
        description={`Forecast of ${runMeta.targetCol} from ${runMeta.fileName}`}
        meta={[
          ["Target", runMeta.targetCol],
          ["Frequency", freqLabel(runMeta.frequency)],
          ["Horizon", `${s.n} periods`],
          ["Forecast period", `${fmtDate(result.forecast_dates[0])} – ${fmtDate(result.forecast_dates.at(-1))}`],
          ["Last run", runMeta.finishedAt ? new Date(runMeta.finishedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—"],
        ]}
        actions={
          <>
            <Button icon={ArrowRight} onClick={() => navigate("forecast")}>
              Open forecast
            </Button>
            <Button variant="primary" icon={FilePlus2} onClick={() => navigate("new")}>
              New forecast
            </Button>
          </>
        }
      />
      <PageBody>
        {run.status === "running" && (
          <Banner tone="info" title="A new forecast is running">
            This dashboard still shows the previous run until it finishes.
          </Banner>
        )}
        {run.status === "failed" && (
          <Banner tone="bad" title="The latest run failed. Showing the previous result.">
            {run.error}
          </Banner>
        )}

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Kpi
            label={`Forecast total · next ${s.n}`}
            value={fmtCompact(s.total)}
            sub={isNum(s.change) ? `${fmtSignedPct(s.change)} vs previous ${s.n} periods` : "No comparable history"}
            tone={isNum(s.change) ? (s.change >= 0 ? "ok" : "bad") : undefined}
            title={fmtNumber(s.total)}
          />
          <Kpi
            label="Average per period"
            value={fmtNumber(s.avg)}
            sub={isNum(s.rangePct) ? `95% range ±${Math.round(s.rangePct * 100)}%` : undefined}
          />
          <Kpi
            label="Selected model"
            value={modelLabel(s.best?.best_model_name)}
            sub={isNum(mase) ? `MASE ${mase.toFixed(3)} · ${mase < 1 ? "beats" : "does not beat"} naive baseline` : undefined}
            tone={isNum(mase) ? (mase < 1 ? "ok" : "bad") : undefined}
          />
          <Kpi
            label="Models evaluated"
            value={`${s.ok.length} of ${s.models.length}`}
            sub={`${s.beatNaive.length} beat the naive baseline`}
          />
        </div>

        <div className="grid gap-4 xl:grid-cols-3">
          <Panel className="xl:col-span-2">
            <PanelHeader
              title="Actual vs forecast"
              subtitle={`${modelLabel(s.best?.best_model_name)} · shaded area is the 95% range`}
              actions={
                <>
                  <span className="hidden text-2xs text-ink-3 sm:inline">History</span>
                  <Segmented value={range} onChange={setRange} options={RANGES} />
                </>
              }
            />
            <PanelBody className="pt-3">
              <ForecastChart result={result} range={range} height={300} />
            </PanelBody>
          </Panel>
          <Attention items={items} />
        </div>

        <div className={`grid gap-4 ${runMeta.groupCol ? "xl:grid-cols-3" : "lg:grid-cols-2"}`}>
          <Panel>
            <PanelHeader
              title="Top models"
              subtitle="Ranked by MASE on the common backtest"
              actions={
                <Button size="sm" variant="ghost" onClick={() => navigate("models")}>
                  View all
                </Button>
              }
            />
            <table className="data-table">
              <thead>
                <tr>
                  <th className="w-8">#</th>
                  <th>Model</th>
                  <th className="num">MASE</th>
                </tr>
              </thead>
              <tbody>
                {topModels.map((m, i) => (
                  <tr key={m.model_name}>
                    <td className="text-ink-3">{i + 1}</td>
                    <td>
                      <span className="font-medium">{modelLabel(m.model_name)}</span>
                      <span className="ml-2 text-xs text-ink-3">{MODEL_CATEGORY[m.model_name]}</span>
                      {m.model_name === s.best?.best_model_name && (
                        <Chip tone="brand" className="ml-2">
                          Selected
                        </Chip>
                      )}
                    </td>
                    <td className={`num ${isNum(m.mase) && m.mase < 1 ? "text-ok" : ""}`}>{isNum(m.mase) ? m.mase.toFixed(3) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>

          {runMeta.groupCol && (
            <Panel>
              <PanelHeader
                title={`Top ${runMeta.groupCol} by volume`}
                actions={
                  <Button size="sm" variant="ghost" onClick={() => navigate("products")}>
                    View all
                  </Button>
                }
              />
              {products.loading ? (
                <p className="px-4 py-6 text-[13px] text-ink-3">Loading…</p>
              ) : products.error ? (
                <p className="px-4 py-6 text-[13px] text-bad">{products.error}</p>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>{runMeta.groupCol}</th>
                      <th className="num">Share</th>
                      <th>Trend</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topGroups.map((g) => (
                      <tr key={g.group_value}>
                        <td className="max-w-[160px] truncate font-medium" title={g.group_value}>
                          {g.group_value}
                        </td>
                        <td className="num">{fmtShare(g.share_pct)}</td>
                        <td>
                          <TrendChip trend={g.trend} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Panel>
          )}

          <Panel>
            <PanelHeader title="Run details" />
            <PanelBody className="py-2">
              <Facts
                items={[
                  ["Dataset", runMeta.fileName],
                  ["Rows", fmtNumber(runMeta.rows)],
                  ["Date column", runMeta.dateCol],
                  runMeta.groupCol && ["Grouped by", runMeta.groupCol],
                  runMeta.country && ["Holiday calendar", countryLabel(runMeta.country)],
                  info.evaluation === "holdout_backtest" && [
                    "Backtest",
                    `${info.backtest_origins} window${info.backtest_origins > 1 ? "s" : ""} × ${info.backtest_horizon} periods`,
                  ],
                  s.best?.ensemble_members?.length && ["Ensemble of", s.best.ensemble_members.map(modelLabel).join(", ")],
                  result.exog_features_used?.length > 0 && ["External drivers", result.exog_features_used.join(", ")],
                  duration && ["Run time", fmtSeconds(duration)],
                ]}
              />
            </PanelBody>
          </Panel>
        </div>
      </PageBody>
    </>
  );
}
