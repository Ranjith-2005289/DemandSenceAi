import { useEffect, useRef, useState } from "react";
import { uploadCsv, errorMessage } from "../api";
import { Alert, Badge, Button, Card, CardHeader, Field, Icon, Spinner, inputClass } from "./ui";
import { MODEL_COUNT, fmtNumber } from "../lib/format";

const SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xlsm", ".json"];

// Countries supported by the `holidays` package that Prophet uses.
const COUNTRY_OPTIONS = [
  { code: "US", label: "United States" },
  { code: "GB", label: "United Kingdom" },
  { code: "IN", label: "India" },
  { code: "CA", label: "Canada" },
  { code: "AU", label: "Australia" },
  { code: "DE", label: "Germany" },
  { code: "FR", label: "France" },
  { code: "BR", label: "Brazil" },
  { code: "MX", label: "Mexico" },
  { code: "JP", label: "Japan" },
  { code: "CN", label: "China" },
  { code: "SG", label: "Singapore" },
  { code: "AE", label: "United Arab Emirates" },
  { code: "ZA", label: "South Africa" },
  { code: "NZ", label: "New Zealand" },
];

// The backend reports frequency as a word ("daily", "weekly (estimated)",
// "minute-level") or occasionally a raw pandas alias ("W-SUN") — normalise both.
const FREQ_WORDS = [
  ["minute", "Minutely", "minutes"], ["hour", "Hourly", "hours"], ["business", "Business days", "business days"],
  ["daily", "Daily", "days"], ["week", "Weekly", "weeks"], ["month", "Monthly", "months"],
  ["quarter", "Quarterly", "quarters"], ["annual", "Yearly", "years"], ["year", "Yearly", "years"],
];
const ALIAS = { D: "daily", B: "business", W: "week", M: "month", MS: "month", ME: "month",
                Q: "quarter", QS: "quarter", H: "hour", T: "minute", MIN: "minute", A: "annual", Y: "year" };
function freqInfo(f) {
  if (!f || f === "unknown") return { label: "Irregular", unit: "periods", estimated: false };
  const raw = String(f);
  const word = ALIAS[raw.split("-")[0].toUpperCase()] || raw.toLowerCase();
  const hit = FREQ_WORDS.find(([k]) => word.includes(k));
  return {
    label: hit ? hit[1] : raw,
    unit: hit ? hit[2] : "periods",
    estimated: raw.includes("estimated"),
  };
}

export default function UploadSection({ uploadData, initialConfig, onUploaded, onRun, onReset, error: runError }) {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const inputRef = useRef(null);

  const [dateCol, setDateCol] = useState("");
  const [targetCol, setTargetCol] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [country, setCountry] = useState("");
  const [horizon, setHorizon] = useState(30);

  // Seed the form from the backend's suggestions whenever a new file lands.
  // (or from the previous run's settings when returning to re-configure).
  useEffect(() => {
    if (!uploadData) return;
    const c = initialConfig || {};
    setDateCol(c.date_col || uploadData.suggestions?.date_col || "");
    setTargetCol(c.target_col || uploadData.suggestions?.target_col || "");
    setGroupCol(c.group_col || "");
    setCountry(c.country || "");
    if (c.forecast_horizon) setHorizon(c.forecast_horizon);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadData]);

  const acceptFile = (f) => {
    if (!f) return;
    if (!SUPPORTED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))) {
      setUploadError(`Unsupported file type. Use ${SUPPORTED_EXTENSIONS.join(", ")}.`);
      return;
    }
    setUploadError(null);
    setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await uploadCsv(formData);
      onUploaded(result, file.name);
    } catch (err) {
      setUploadError(errorMessage(err, "Upload failed."));
    } finally {
      setUploading(false);
    }
  };

  // ── Step 1: file drop ────────────────────────────────────────────────────
  if (!uploadData) {
    return (
      <div className="mx-auto max-w-3xl animate-fade-in">
        <div className="mb-8 text-center">
          <Badge tone="brand" className="mb-4">
            {MODEL_COUNT} models + ensemble · backtested · explained
          </Badge>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
            Forecast demand from any sales history
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-base text-slate-600">
            Upload a time series. Every model is trained, tested on the same held-out windows,
            and the most accurate one — or an ensemble of the best — produces your forecast.
          </p>
        </div>

        <Card className="p-6">
          {uploadError && <div className="mb-4"><Alert title="Upload failed">{uploadError}</Alert></div>}

          <input
            ref={inputRef}
            type="file"
            accept={SUPPORTED_EXTENSIONS.join(",")}
            onChange={(e) => acceptFile(e.target.files?.[0])}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); acceptFile(e.dataTransfer.files?.[0]); }}
            className={`flex w-full flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition ${
              dragging ? "border-brand-400 bg-brand-50" : "border-slate-200 bg-slate-50/60 hover:border-brand-300 hover:bg-brand-50/40"
            }`}
          >
            <span className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-brand-600 shadow-card ring-1 ring-slate-200">
              <Icon.Upload />
            </span>
            <span className="mt-4 text-sm font-semibold text-slate-900">
              Drop your file here, or <span className="text-brand-600">browse</span>
            </span>
            <span className="mt-1 text-xs text-slate-500">CSV, Excel (.xlsx) or JSON · one row per date (or per transaction)</span>
          </button>

          {file && (
            <div className="mt-4 flex items-center justify-between gap-3 rounded-xl border border-slate-200 px-4 py-3">
              <div className="flex min-w-0 items-center gap-3">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600"><Icon.File /></span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-900">{file.name}</p>
                  <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={() => setFile(null)} disabled={uploading}>Remove</Button>
                <Button onClick={handleUpload} disabled={uploading}>
                  {uploading ? <><Spinner /> Analyzing…</> : "Upload & analyze"}
                </Button>
              </div>
            </div>
          )}
        </Card>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          {[
            ["Auto-detects columns", "Dates, numeric targets, product/store groupings and price/promo signals."],
            ["Honest model selection", "Each model forecasts the same unseen windows; ranked by MASE vs. a naive baseline."],
            ["AI explanations", "Ask why a model won or how to act on the forecast, grounded in your numbers."],
          ].map(([t, d]) => (
            <div key={t} className="rounded-xl border border-slate-200/80 bg-white/60 p-4">
              <p className="text-sm font-semibold text-slate-900">{t}</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">{d}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── Step 2: configure ────────────────────────────────────────────────────
  const dateColumns = uploadData.detected?.date_columns || [];
  const numericColumns = uploadData.detected?.numeric_columns || [];
  const groupingColumns = uploadData.detected?.grouping_columns || [];
  const preview = uploadData.preview || [];
  const previewCols = preview.length ? Object.keys(preview[0]) : [];
  const freq = uploadData.suggestions?.frequency;
  const canRun = dateCol && targetCol && horizon >= 1 && horizon <= 365;

  const submit = (e) => {
    e.preventDefault();
    if (!canRun) return;
    onRun({
      date_col: dateCol,
      target_col: targetCol,
      group_col: groupCol || null,
      country: country || null,
      forecast_horizon: horizon,
    });
  };

  return (
    <div className="grid gap-6 animate-fade-in lg:grid-cols-5">
      <Card className="lg:col-span-2">
        <CardHeader title="Configure forecast" subtitle="We pre-filled the columns we detected." />
        <form onSubmit={submit} className="space-y-5 p-6">
          {runError && <Alert title="Forecast failed">{runError}</Alert>}
          {(uploadData.validation?.errors || []).map((e, i) => (
            <Alert key={`e${i}`} title="Data problem">{e}</Alert>
          ))}
          {(uploadData.validation?.warnings || []).length > 0 && (
            <Alert tone="warning" title="Heads up">
              <ul className="list-disc space-y-0.5 pl-4">
                {uploadData.validation.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </Alert>
          )}

          <Field label="Date column">
            <select value={dateCol} onChange={(e) => setDateCol(e.target.value)} className={inputClass}>
              <option value="">Select…</option>
              {dateColumns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </Field>

          <Field label="What to forecast" hint="The numeric column to predict (sales, units, revenue…).">
            <select value={targetCol} onChange={(e) => setTargetCol(e.target.value)} className={inputClass}>
              <option value="">Select…</option>
              {numericColumns.map((c) => (
                <option key={c} value={c}>
                  {c}{uploadData.detected?.numeric_meta?.[c]?.is_likely_target ? "  · suggested" : ""}
                </option>
              ))}
            </select>
          </Field>

          <Field
            label={`Forecast horizon (${freqInfo(freq).unit})`}
            hint={`How far ahead to predict, 1–365 ${freqInfo(freq).unit}.`}
          >
            <input
              type="number" min="1" max="365" value={horizon}
              onChange={(e) => setHorizon(Math.min(365, Math.max(1, parseInt(e.target.value, 10) || 1)))}
              className={inputClass}
            />
          </Field>

          {groupingColumns.length > 0 && (
            <Field label="Break down by (optional)" hint="Adds a ranked table per group, each forecastable on its own.">
              <select value={groupCol} onChange={(e) => setGroupCol(e.target.value)} className={inputClass}>
                <option value="">None — whole dataset</option>
                {groupingColumns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </Field>
          )}

          <Field label="Holiday calendar (optional)" hint="Lets Prophet account for that country's public holidays.">
            <select value={country} onChange={(e) => setCountry(e.target.value)} className={inputClass}>
              <option value="">None</option>
              {COUNTRY_OPTIONS.map(({ code, label }) => <option key={code} value={code}>{label}</option>)}
            </select>
          </Field>

          <div className="flex gap-3 pt-1">
            <Button type="button" variant="secondary" onClick={onReset} className="flex-1">Change file</Button>
            <Button type="submit" disabled={!canRun} className="flex-[2]">
              <Icon.Sparkles width={16} height={16} /> Run forecast
            </Button>
          </div>
        </form>
      </Card>

      <div className="space-y-6 lg:col-span-3">
        <div className="grid grid-cols-3 gap-4">
          {[
            ["Rows", fmtNumber(uploadData.shape?.rows)],
            ["Columns", fmtNumber(uploadData.shape?.cols)],
            ["Frequency", freqInfo(freq).label + (freqInfo(freq).estimated ? " (est.)" : "")],
          ].map(([l, v]) => (
            <Card key={l} className="p-4">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{l}</p>
              <p className="mt-1 text-xl font-semibold text-slate-900">{v}</p>
            </Card>
          ))}
        </div>

        <Card>
          <CardHeader
            title="Data preview"
            subtitle={uploadData.file?.filename}
            actions={<Badge>{previewCols.length} columns</Badge>}
          />
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50/70">
                  {previewCols.map((c) => (
                    <th key={c} className="whitespace-nowrap px-4 py-2.5 text-left text-xs font-semibold text-slate-600">
                      <span className="inline-flex items-center gap-1.5">
                        {c}
                        {c === dateCol && <Badge tone="brand">date</Badge>}
                        {c === targetCol && <Badge tone="success">target</Badge>}
                        {c === groupCol && <Badge tone="warning">group</Badge>}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {preview.map((row, i) => (
                  <tr key={i}>
                    {previewCols.map((c) => (
                      <td key={c} className="whitespace-nowrap px-4 py-2 font-mono text-xs text-slate-700">
                        {row[c] === null || row[c] === undefined ? <span className="text-slate-300">null</span> : String(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {uploadData.data_quality?.total_nulls > 0 && (
            <p className="border-t border-slate-100 px-6 py-3 text-xs text-slate-500">
              {fmtNumber(uploadData.data_quality.total_nulls)} missing values detected — they'll be
              interpolated during preprocessing.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
