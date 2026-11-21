import { fmtNumber, isNum, modelLabel } from "./format";

const sum = (a) => a.reduce((s, v) => s + (isNum(v) ? v : 0), 0);
const mean = (a) => (a.length ? sum(a) / a.length : NaN);

/** Headline numbers for a forecast result. */
export function forecastSummary(result) {
  const best = result?.best_model;
  const fc = best?.forecast || [];
  const n = fc.length;
  const hist = result?.historical?.values || [];
  const total = sum(fc);
  const prev = hist.length >= n && n > 0 ? sum(hist.slice(-n)) : NaN;
  const change = isNum(prev) && prev !== 0 ? (total - prev) / Math.abs(prev) : NaN;
  const avg = n ? total / n : NaN;

  let rangePct = NaN;
  if (Array.isArray(best?.forecast_lower) && Array.isArray(best?.forecast_upper) && n) {
    const half = mean(fc.map((_, i) => (best.forecast_upper[i] - best.forecast_lower[i]) / 2).filter(isNum));
    rangePct = avg ? half / Math.abs(avg) : NaN;
  }

  const models = result?.all_models || [];
  const ok = models.filter((m) => m.status === "success");
  const failed = models.filter((m) => m.status !== "success");
  const beatNaive = ok.filter((m) => isNum(m.mase) && m.mase < 1);

  return { n, total, prev, change, avg, rangePct, models, ok, failed, beatNaive, best };
}

/**
 * Items for the dashboard's "Needs attention" panel, most severe first.
 * Each: { tone: "bad"|"warn"|"info", title, detail, route, action }.
 */
export function attentionItems({ result, runMeta, products, report }) {
  if (!result) return [];
  const s = forecastSummary(result);
  const items = [];
  const mase = s.best?.mase;

  if (isNum(mase) && mase >= 1) {
    items.push({
      tone: "bad",
      title: "Forecast does not beat the seasonal-naive baseline",
      detail: `Selected model MASE is ${mase.toFixed(2)} (below 1.00 is better than naive). Treat this forecast as indicative.`,
      route: "models",
      action: "Compare models",
    });
  }

  const alerts = report?.stockout_alerts || [];
  if (alerts.length) {
    items.push({
      tone: "bad",
      title: `${alerts.length} possible stockout period${alerts.length > 1 ? "s" : ""} in history`,
      detail: "Sudden drops to near zero that may be lost sales rather than low demand.",
      route: "report",
      action: "Open report",
    });
  }

  const declining = (products?.groups || []).filter((g) => g.trend === "declining");
  if (declining.length) {
    const names = declining.slice(0, 3).map((g) => g.group_value).join(", ");
    items.push({
      tone: "warn",
      title: `Declining trend: ${declining.length} ${runMeta?.groupCol || "group"} value${declining.length > 1 ? "s" : ""}`,
      detail: `${names}${declining.length > 3 ? ` and ${declining.length - 3} more` : ""}.`,
      route: "products",
      action: "View breakdown",
    });
  }

  if (s.failed.length) {
    items.push({
      tone: "warn",
      title: `${s.failed.length} model${s.failed.length > 1 ? "s" : ""} failed to train`,
      detail: s.failed.map((m) => modelLabel(m.model_name)).join(", "),
      route: "models",
      action: "View models",
    });
  }

  if (isNum(s.rangePct) && s.rangePct > 0.5) {
    items.push({
      tone: "warn",
      title: "Wide forecast range",
      detail: `The 95% range is about ±${Math.round(s.rangePct * 100)}% of the forecast. Plan buffers accordingly.`,
      route: "forecast",
      action: "View forecast",
    });
  }

  for (const w of runMeta?.warnings || []) {
    items.push({ tone: "warn", title: "Data check", detail: w, route: "new", action: "Review data" });
  }

  const info = result.model_run_info;
  if (info?.training_window_truncated) {
    items.push({
      tone: "info",
      title: "Trained on recent history only",
      detail: `Models used the most recent ${fmtNumber(info.row_count)} of ${fmtNumber(info.original_row_count)} observations.`,
    });
  }

  return items;
}

/**
 * Plain-language cause when the language model behind the report/assistant
 * fails. The backend returns these failures as ordinary text, not HTTP errors.
 */
export function llmUnavailableReason(text) {
  if (/API key not valid|API_KEY_INVALID|api key/i.test(text)) return "The forecasting server's Gemini API key is missing or invalid.";
  if (/quota|429/i.test(text)) return "The Gemini API quota for the server's key is used up for today.";
  return "The language model service didn't respond.";
}
