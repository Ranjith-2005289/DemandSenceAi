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
