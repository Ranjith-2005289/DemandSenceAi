// Number / label formatting shared by the dashboard.

export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export function fmtNumber(v, { digits } = {}) {
  if (!isNum(v)) return "—";
  const abs = Math.abs(v);
  const d = digits ?? (abs >= 1000 ? 0 : abs >= 100 ? 1 : 2);
  return v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
}

export function fmtCompact(v) {
  if (!isNum(v)) return "—";
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

export function fmtPct(v, digits = 1) {
  if (!isNum(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtSeconds(v) {
  if (!isNum(v)) return "—";
  if (v < 0.01) return "<0.01s";
  if (v < 60) return `${v.toFixed(v < 10 ? 2 : 1)}s`;
  return `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`;
}

// Which family each model belongs to (display only).
export const MODEL_CATEGORY = {
  ARIMA: "Statistical",
  SARIMA: "Statistical",
  SARIMAX: "Statistical",
  HoltWinters: "Exp. smoothing",
  ExponentialSmoothing: "Exp. smoothing",
  AutoETS: "Exp. smoothing",
  Theta: "Exp. smoothing",
  MSTL: "Decomposition",
  Prophet: "Decomposition",
  MovingAverage: "Baseline",
  SeasonalNaive: "Baseline",
  XGBoost: "Machine learning",
  RandomForest: "Machine learning",
  SVM: "Machine learning",
  CrostonSBA: "Intermittent",
  Ensemble: "Ensemble",
};

export const MODEL_LABEL = {
  ExponentialSmoothing: "Exp. Smoothing",
  HoltWinters: "Holt-Winters",
  RandomForest: "Random Forest",
  MovingAverage: "Moving Average",
  SeasonalNaive: "Seasonal Naive",
  CrostonSBA: "Croston (SBA)",
  Ensemble: "Ensemble (top 3)",
};

export const modelLabel = (name) => MODEL_LABEL[name] || name;

export const MODEL_COUNT = Object.keys(MODEL_CATEGORY).filter((m) => m !== "Ensemble").length;

/** "2026-01-12" → "12 Jan 2026" (dates without a time are treated as calendar dates). */
export function fmtDate(v) {
  if (!v) return "—";
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(v) ? `${v}T00:00:00` : v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/** Signed percentage from a ratio, e.g. 0.052 → "+5.2%". */
export function fmtSignedPct(v, digits = 1) {
  if (!isNum(v)) return "—";
  const s = (v * 100).toFixed(digits);
  return `${v > 0 ? "+" : ""}${s}%`;
}

/** Backend sends "daily", "weekly (estimated)", "unknown"… */
export const freqLabel = (f) => (!f || f === "unknown" ? "—" : String(f).charAt(0).toUpperCase() + String(f).slice(1));

// Holiday calendars offered in the UI (codes from the `holidays` package).
export const COUNTRY_NAMES = {
  US: "United States", GB: "United Kingdom", IN: "India", CA: "Canada", AU: "Australia",
  DE: "Germany", FR: "France", BR: "Brazil", MX: "Mexico", JP: "Japan", CN: "China",
  SG: "Singapore", AE: "United Arab Emirates", ZA: "South Africa", NZ: "New Zealand",
};
export const countryLabel = (code) => COUNTRY_NAMES[code] || code;

/** Short axis label: "12 Jan" / "Jan 12" depending on locale. */
export function fmtShortDate(v) {
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(v) ? `${v}T00:00:00` : v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

export const fmtShare = (v) => (isNum(v) ? `${v.toFixed(1)}%` : "—");
