import { useEffect, useRef, useState } from "react";
import { ArrowRight, Check, FileSpreadsheet, Loader2, Play, Upload, X } from "lucide-react";
import PageHeader, { PageBody } from "@/components/shell/PageHeader";
import Stepper from "@/components/Stepper";
import { Banner, Button, Chip, Facts, Field, Panel, PanelBody, PanelFooter, PanelHeader, Select, TextInput } from "@/components/ui";
import { navigate } from "@/lib/router";
import { cn } from "@/lib/utils";
import { useAppState } from "@/state/AppState";
import { COUNTRY_NAMES, MODEL_CATEGORY, MODEL_COUNT, countryLabel, fmtNumber, fmtSeconds, freqLabel, isNum, modelLabel } from "@/lib/format";

const SUPPORTED = [".csv", ".xlsx", ".xlsm", ".json"];

const COUNTRIES = Object.entries(COUNTRY_NAMES).map(([value, label]) => ({ value, label }));

const STEPS = [
  { label: "Upload data", sub: "CSV, Excel or JSON" },
  { label: "Map columns", sub: "Date, target, horizon" },
  { label: "Run models", sub: `${MODEL_COUNT} models + ensemble` },
  { label: "Review", sub: "Results and next steps" },
];

/* ── Step 1: upload ──────────────────────────────────────────────────────── */

function UploadStep() {
  const { uploadFile, uploading, uploadError } = useAppState();
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState(null);
  const inputRef = useRef(null);

  const pick = (f) => {
    if (!f) return;
    if (!SUPPORTED.some((ext) => f.name.toLowerCase().endsWith(ext))) {
      setLocalError(`"${f.name}" is not a supported file. Use ${SUPPORTED.join(", ")}.`);
      return;
    }
    setLocalError(null);
    setFile(f);
  };

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Panel className="lg:col-span-2">
        <PanelHeader title="Upload dataset" subtitle="One row per date, or per date and product/store." />
        <PanelBody className="space-y-3">
          {(localError || uploadError) && <Banner tone="bad" title="Upload failed">{localError || uploadError}</Banner>}
          <input
            ref={inputRef}
            type="file"
            accept={SUPPORTED.join(",")}
            className="hidden"
            onChange={(e) => {
              pick(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <div
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pick(e.dataTransfer.files?.[0]);
            }}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed px-6 py-10 text-center transition-colors",
              dragging ? "border-brand bg-brand-soft" : "border-line-strong bg-surface-muted hover:border-brand"
            )}
          >
            <Upload className="h-6 w-6 text-ink-3" />
            <p className="mt-2 text-[13px] font-medium text-ink">
              Drag a file here or <span className="text-brand-ink underline-offset-2 hover:underline">browse</span>
            </p>
            <p className="mt-0.5 text-xs text-ink-3">CSV, XLSX, XLSM or JSON</p>
          </div>

          {file && (
            <div className="flex items-center gap-3 rounded border border-line px-3 py-2">
              <FileSpreadsheet className="h-5 w-5 shrink-0 text-brand" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-[13px] font-medium text-ink">{file.name}</p>
                <p className="text-xs text-ink-3">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
              <button type="button" onClick={() => setFile(null)} className="rounded p-1 text-ink-3 hover:bg-surface-hover hover:text-ink" aria-label="Remove file">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
        </PanelBody>
        <PanelFooter className="justify-end">
          <Button variant="primary" icon={ArrowRight} disabled={!file} loading={uploading} onClick={() => uploadFile(file)}>
            {uploading ? "Analysing file…" : "Upload and continue"}
          </Button>
        </PanelFooter>
      </Panel>

      <Panel>
        <PanelHeader title="File requirements" />
        <PanelBody className="py-2">
          <Facts
            items={[
              ["Formats", "CSV, XLSX, XLSM, JSON"],
              ["Required", "Date + numeric target"],
              ["Optional", "Product / store column"],
              ["Optional", "Price and promotion columns"],
              ["History", "2+ seasonal cycles"],
            ]}
          />
          <p className="mt-3 text-xs text-ink-3">
            Columns are detected automatically. Excel files use the first sheet. Currency symbols and thousands separators are handled.
          </p>
        </PanelBody>
      </Panel>
    </div>
  );
}

/* ── Step 2: map columns ─────────────────────────────────────────────────── */

const ROLE_CHIP = { date: ["brand", "Date"], target: ["ok", "Numeric"], grouping: ["neutral", "Group"] };

function MapStep() {
  const { file, upload, config, setConfig, clearDataset, startRun, run } = useAppState();
  const d = upload.detected || {};
  const set = (k) => (v) => setConfig((c) => ({ ...c, [k]: v }));
  const horizonOk = isNum(Number(config.horizon)) && Number(config.horizon) >= 1 && Number(config.horizon) <= 365;
  const blockingErrors = upload.validation?.errors || [];
  const ready = config.dateCol && config.targetCol && horizonOk && blockingErrors.length === 0;
  const columns = upload.columns || [];
  const preview = upload.preview || [];
  const exog = [...(d.exog_columns?.price_cols || []), ...(d.exog_columns?.promo_cols || [])];

  return (
    <div className="space-y-4">
      {blockingErrors.length > 0 && (
        <Banner
          tone="bad"
          title="This file can't be forecast yet"
          action={
            <Button size="sm" onClick={clearDataset}>
              Choose another file
            </Button>
          }
        >
          <ul className="list-disc space-y-0.5 pl-4">
            {blockingErrors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </Banner>
      )}
      {run.status === "failed" && (
        <Banner tone="bad" title="The forecast run failed">
          {run.error}
        </Banner>
      )}
      {upload.validation?.warnings?.length > 0 && (
        <Banner tone="warn" title="Data checks">
          <ul className="list-disc space-y-0.5 pl-4">
            {upload.validation.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Banner>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel className="lg:col-span-2">
          <PanelHeader title="Column mapping" subtitle="Detected columns are pre-selected. Check them before running." />
          <PanelBody className="grid gap-x-4 gap-y-4 sm:grid-cols-2">
            <Field label="Date column" required htmlFor="f-date">
              <Select id="f-date" value={config.dateCol} onChange={set("dateCol")} placeholder="Select…" options={d.date_columns || []} />
            </Field>
            <Field label="Value to forecast" required htmlFor="f-target" hint="For example units sold or revenue.">
              <Select id="f-target" value={config.targetCol} onChange={set("targetCol")} placeholder="Select…" options={d.numeric_columns || []} />
            </Field>
            <Field
              label="Forecast horizon"
              required
              htmlFor="f-h"
              hint={
                !horizonOk
                  ? "Enter 1–365."
                  : freqLabel(upload.suggestions?.frequency) === "—"
                  ? "Periods ahead."
                  : `Periods ahead, at ${freqLabel(upload.suggestions?.frequency).toLowerCase()} frequency.`
              }
            >
              <TextInput
                id="f-h"
                type="number"
                min={1}
                max={365}
                value={config.horizon}
                onChange={(e) => set("horizon")(e.target.value === "" ? "" : Math.max(1, parseInt(e.target.value, 10) || 1))}
                className={cn(!horizonOk && "border-bad")}
              />
            </Field>
            <Field label="Group by" htmlFor="f-group" hint="Adds a per-product breakdown you can forecast individually.">
              <Select
                id="f-group"
                value={config.groupCol}
                onChange={set("groupCol")}
                placeholder="None (whole dataset)"
                options={d.grouping_columns || []}
                disabled={!d.grouping_columns?.length}
              />
            </Field>
            <Field label="Holiday calendar" htmlFor="f-country" hint="Public holidays are used by the Prophet model.">
              <Select id="f-country" value={config.country} onChange={set("country")} placeholder="None" options={COUNTRIES} />
            </Field>
            <Field label="External drivers" hint="Price and promotion columns used by XGBoost, Random Forest and SARIMAX.">
              <div className="flex min-h-[32px] flex-wrap items-center gap-1.5">
                {exog.length ? exog.map((c) => <Chip key={c}>{c}</Chip>) : <span className="text-[13px] text-ink-3">None detected</span>}
              </div>
            </Field>
          </PanelBody>
          <PanelFooter>
            <Button variant="ghost" onClick={clearDataset}>
              Change file
            </Button>
            <Button variant="primary" icon={Play} disabled={!ready} onClick={startRun}>
              Run forecast
            </Button>
          </PanelFooter>
        </Panel>

        <Panel>
          <PanelHeader title="Dataset" />
          <PanelBody className="py-2">
            <Facts
              items={[
                ["File", file?.name],
                ["Size", file ? `${(file.size / 1024).toFixed(1)} KB` : "—"],
                ["Rows", fmtNumber(upload.shape?.rows)],
                ["Columns", fmtNumber(upload.shape?.cols)],
                ["Frequency", freqLabel(upload.suggestions?.frequency)],
                ["Missing values", fmtNumber(upload.data_quality?.total_nulls ?? 0)],
              ]}
            />
          </PanelBody>
        </Panel>
      </div>

      {preview.length > 0 && (
        <Panel>
          <PanelHeader title="Data preview" subtitle={`First ${preview.length} rows`} />
          <div className="scroll-thin overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  {columns.map((c) => {
                    const chip = ROLE_CHIP[c.role];
                    return (
                      <th key={c.name}>
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-semibold normal-case tracking-normal text-ink-2">{c.name}</span>
                          {chip && <Chip tone={chip[0]}>{chip[1]}</Chip>}
                        </div>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c.name} className={cn("whitespace-nowrap", typeof row[c.name] === "number" && "num")}>
                        {row[c.name] === null || row[c.name] === undefined ? <span className="text-ink-3">—</span> : String(row[c.name])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ── Step 3: running ─────────────────────────────────────────────────────── */

// The forecast is one backend request, so per-model progress can't be
// streamed. The active stage is an estimate from elapsed time.
const STAGES = [
  ["Preparing data", "Parsing dates, regularising frequency, filling gaps, capping outliers."],
  ["Training and backtesting", "Each model is fitted, then refitted from earlier cut-off points to forecast windows it hasn't seen."],
  ["Ranking and ensembling", "Models are ranked by MASE on those windows; the top three are averaged into an ensemble."],
  ["Producing the forecast", "The selected model forecasts the horizon with a 95% range calibrated on its backtest errors."],
];

function RunningStep() {
  const { run, config, file } = useAppState();
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((Date.now() - run.startedAt) / 1000), 500);
    return () => clearInterval(t);
  }, [run.startedAt]);
  const active = elapsed < 3 ? 0 : elapsed < 40 ? 1 : elapsed < 60 ? 2 : 3;
  const models = Object.keys(MODEL_CATEGORY).filter((m) => m !== "Ensemble");

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Panel className="lg:col-span-2">
        <PanelHeader
          title="Running forecast"
          subtitle="You can leave this page. The run continues and its status shows in the top bar."
          actions={<span className="text-lg font-semibold tabular-nums text-ink">{`${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, "0")}`}</span>}
        />
        <ol className="divide-y divide-line">
          {STAGES.map(([title, detail], i) => {
            const state = i < active ? "done" : i === active ? "active" : "pending";
            return (
              <li key={title} className="flex gap-3 px-4 py-3">
                <span
                  className={cn(
                    "mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-2xs font-semibold",
                    state === "done" && "border-ok bg-ok text-white",
                    state === "active" && "border-brand text-brand",
                    state === "pending" && "border-line-strong text-ink-3"
                  )}
                >
                  {state === "done" ? <Check className="h-3 w-3" strokeWidth={3} /> : state === "active" ? <Loader2 className="h-3 w-3 animate-spin" /> : i + 1}
                </span>
                <div>
                  <p className={cn("text-[13px] font-medium", state === "pending" ? "text-ink-3" : "text-ink")}>{title}</p>
                  <p className="mt-0.5 text-xs text-ink-3">{detail}</p>
                </div>
              </li>
            );
          })}
        </ol>
        <PanelFooter className="block">
          <p className="mb-1.5 text-2xs font-medium uppercase tracking-wide text-ink-3">Models in this run</p>
          <div className="flex flex-wrap gap-1">
            {models.map((m) => (
              <Chip key={m}>{modelLabel(m)}</Chip>
            ))}
          </div>
        </PanelFooter>
      </Panel>
      <Panel>
        <PanelHeader title="Configuration" />
        <PanelBody className="py-2">
          <Facts
            items={[
              ["File", file?.name],
              ["Date column", config.dateCol],
              ["Forecasting", config.targetCol],
              ["Horizon", `${config.horizon} periods`],
              ["Group by", config.groupCol || "None"],
              ["Holiday calendar", config.country ? countryLabel(config.country) : "None"],
            ]}
          />
          <p className="mt-3 text-xs text-ink-3">Typical run time is 15–90 seconds, depending on history length.</p>
        </PanelBody>
      </Panel>
    </div>
  );
}

/* ── Step 4: review ──────────────────────────────────────────────────────── */

function ReviewStep({ onAdjust }) {
  const { result, runMeta, clearDataset } = useAppState();
  const best = result.best_model;
  const ok = (result.all_models || []).filter((m) => m.status === "success").length;
  return (
    <Panel>
      <PanelBody className="flex flex-wrap items-start gap-4">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-ok-soft">
          <Check className="h-5 w-5 text-ok" strokeWidth={2.5} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-ink">Forecast complete</p>
          <p className="mt-0.5 text-[13px] text-ink-3">
            {ok} of {result.all_models?.length} models trained. {modelLabel(best?.best_model_name)} was selected
            {isNum(best?.mase) ? ` with MASE ${best.mase.toFixed(3)}` : ""}. Run time {fmtSeconds((runMeta.finishedAt - runMeta.startedAt) / 1000)}.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button variant="primary" icon={ArrowRight} onClick={() => navigate("dashboard")}>
              Open dashboard
            </Button>
            <Button onClick={() => navigate("forecast")}>View forecast</Button>
            <Button onClick={() => navigate("models")}>Compare models</Button>
            <div className="mx-1 w-px self-stretch bg-line" />
            <Button variant="ghost" onClick={onAdjust}>
              Adjust and re-run
            </Button>
            <Button variant="ghost" onClick={clearDataset}>
              Start with another file
            </Button>
          </div>
        </div>
      </PanelBody>
    </Panel>
  );
}

export default function NewForecastPage() {
  const { upload, run, runMeta } = useAppState();
  const [adjusting, setAdjusting] = useState(false);

  const finishedThisUpload = run.status === "ready" && upload && runMeta?.savedAs === upload.file?.saved_as;
  let step = 0;
  if (upload) step = 1;
  if (run.status === "running") step = 2;
  else if (finishedThisUpload && !adjusting) step = 3;

  useEffect(() => {
    if (run.status === "running") setAdjusting(false);
  }, [run.status]);

  return (
    <>
      <PageHeader title="New forecast" description="Upload a dataset, map its columns, and run every model on a common backtest.">
        <div className="mx-auto max-w-[1400px] border-t border-line px-6 py-3">
          <Stepper steps={STEPS} current={step === 3 ? 4 : step} failedIndex={run.status === "failed" && step === 1 ? 2 : undefined} />
        </div>
      </PageHeader>
      <PageBody>
        {step === 0 && <UploadStep />}
        {step === 1 && <MapStep />}
        {step === 2 && <RunningStep />}
        {step === 3 && <ReviewStep onAdjust={() => setAdjusting(true)} />}
      </PageBody>
    </>
  );
}
