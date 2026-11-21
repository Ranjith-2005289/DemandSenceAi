import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { generateInsightsReport, getAsyncReportStatus, getProductSummary, errorMessage } from "../api";
import { Alert, Badge, Button, Card, CardHeader, Icon, Spinner } from "./ui";
import { fmtCompact, fmtNumber } from "../lib/format";
import Markdown from "./Markdown";

const TREND_COLOR = "#4f46e5";
const CRITICAL = "#e11d48";

const SECTIONS = [
  { key: "dataset_summary", title: "Dataset overview" },
  { key: "sales_forecast_summary", title: "Forecast outlook" },
  { key: "recommendations", title: "Recommendations" },
  { key: "profit_loss_analysis", title: "Revenue & trend signal" },
  { key: "stock_alerts_summary", title: "Stock alerts" },
];

// The LLM returns plain text, sometimes with "1." numbered items. Render
// those as a list instead of one long paragraph.
function Prose({ text }) {
  if (!text) return null;
  const parts = String(text).split(/\s*(?=(?:^|\s)\d{1,2}\.\s)/).map((s) => s.trim()).filter(Boolean);
  const numbered = parts.length > 1 && parts.every((p) => /^\d{1,2}\.\s/.test(p));
  if (numbered) {
    return (
      <ol className="space-y-2">
        {parts.map((p, i) => (
          <li key={i} className="flex gap-3 text-sm leading-relaxed text-slate-700">
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-brand-50 text-[11px] font-semibold text-brand-700">{i + 1}</span>
            <span>{p.replace(/^\d+\.\s*/, "")}</span>
          </li>
        ))}
      </ol>
    );
  }
  return <div className="text-sm leading-relaxed text-slate-700"><Markdown text={text} /></div>;
}

export default function InsightsPage({ filename, date_col, target_col, group_col, forecast_result, onBack }) {
  const [report, setReport] = useState(null);
  const [productSummary, setProductSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fullReady, setFullReady] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let pollTimer = null;
    setLoading(true);
    setError(null);
    setFullReady(false);

    (async () => {
      const summary = group_col
        ? await getProductSummary(filename, date_col, target_col, group_col, 50).catch(() => null)
        : null;
      if (cancelled) return;
      setProductSummary(summary);
      try {
        const data = await generateInsightsReport(filename, date_col, target_col, group_col, forecast_result, summary);
        if (cancelled) return;
        setReport(data);
        setLoading(false);
        if (data.async_report_path) {
          let polls = 0;
          pollTimer = setInterval(async () => {
            polls += 1;
            if (polls > 60) return clearInterval(pollTimer); // give up after ~2 minutes
            try {
              const status = await getAsyncReportStatus(filename);
              if (status.ready && status.report && !cancelled) {
                // Keep the chart data from the first response — the
                // background report doesn't include it.
                setReport((prev) => ({ ...status.report, chart_data: prev?.chart_data }));
                setFullReady(true);
                clearInterval(pollTimer);
              }
            } catch {
              /* transient — keep polling */
            }
          }, 2000);
        }
      } catch (err) {
        if (!cancelled) {
          setLoading(false);
          setError(errorMessage(err, "Failed to generate the insights report."));
        }
      }
    })();

    return () => {
      cancelled = true;
      if (pollTimer) clearInterval(pollTimer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filename, date_col, target_col, group_col, attempt]);

  const header = (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <button type="button" onClick={onBack} className="mb-3 inline-flex items-center gap-1.5 text-sm font-medium text-slate-500 hover:text-slate-900">
          <Icon.ArrowLeft width={16} height={16} /> Back to forecast
        </button>
        <div className="flex items-center gap-2">
          <Badge tone="brand"><Icon.Sparkles width={12} height={12} /> AI generated</Badge>
          {fullReady && <Badge tone="success">Full report ready</Badge>}
        </div>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">Insights report</h1>
        <p className="mt-1 text-sm text-slate-500">
          A briefing on <span className="font-medium text-slate-700">{target_col}</span>, grounded in your data, the forecast and a retail knowledge base.
        </p>
      </div>
      {report && (
        <Button variant="secondary" onClick={() => window.print()}>
          <Icon.Download width={16} height={16} /> Print / PDF
        </Button>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        {header}
        <Card className="flex flex-col items-center gap-3 py-16">
          <Spinner className="h-8 w-8 text-brand-600" />
          <p className="text-sm font-medium text-slate-900">Writing your insights report…</p>
          <p className="text-xs text-slate-500">Analyzing the series and retrieving relevant planning guidance.</p>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        {header}
        <Alert title="Couldn't generate the report">
          <p>{error}</p>
          <Button variant="secondary" size="sm" className="mt-3" onClick={() => setAttempt((a) => a + 1)}>Try again</Button>
        </Alert>
      </div>
    );
  }

  if (!report) return header;

  const trendData = (report.chart_data?.dates || []).map((date, i) => ({ date, value: report.chart_data.values[i] }));
  const alerts = report.stockout_alerts || [];
  const ov = report.dataset_overview || {};
  const products = (productSummary?.groups || []).slice(0, 8).map((r) => ({ name: String(r.group_value), share: r.share_pct }));

  return (
    <div className="space-y-6 animate-fade-in">
      {header}

      {Object.keys(ov).length > 0 && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {[
            ["Period", ov.date_start && ov.date_end ? `${ov.date_start} → ${ov.date_end}` : "—"],
            ["Average", fmtNumber(ov.mean)],
            ["Volatility (CV)", ov.coefficient_of_variation != null ? ov.coefficient_of_variation.toFixed(2) : "—"],
            ["Overall trend", ov.trend ? `${ov.trend}${ov.growth_rate_pct != null ? ` (${ov.growth_rate_pct > 0 ? "+" : ""}${ov.growth_rate_pct}%)` : ""}` : "—"],
          ].map(([l, v]) => (
            <Card key={l} className="p-4">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{l}</p>
              <p className="mt-1 truncate text-base font-semibold capitalize text-slate-900">{v}</p>
            </Card>
          ))}
        </div>
      )}

      <Card>
        <CardHeader
          title="Sales history"
          subtitle={alerts.length ? `${alerts.length} possible stockout period${alerts.length > 1 ? "s" : ""} highlighted` : "No stockout-risk patterns detected"}
          actions={alerts.length > 0 && <Badge tone="danger"><Icon.Alert width={12} height={12} /> {alerts.length} alert{alerts.length > 1 ? "s" : ""}</Badge>}
        />
        <div className="h-80 px-2 py-4 sm:px-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trendData} margin={{ top: 8, right: 16, left: 4, bottom: 0 }}>
              <CartesianGrid stroke="#eef2f6" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={{ stroke: "#e2e8f0" }} minTickGap={48} />
              <YAxis tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} tickFormatter={fmtCompact} width={56} />
              <Tooltip formatter={(v) => [fmtNumber(Number(v)), target_col || "Value"]} contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12 }} />
              {alerts.map((a, i) => (
                <ReferenceArea key={i} x1={a.start_date} x2={a.end_date} fill={CRITICAL} fillOpacity={0.1} stroke={CRITICAL} strokeOpacity={0.3} />
              ))}
              <Line type="monotone" dataKey="value" stroke={TREND_COLOR} strokeWidth={1.8} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {SECTIONS.filter((s) => report[s.key]).map((s) => (
          <Card key={s.key} className={s.key === "recommendations" ? "lg:col-span-2" : ""}>
            <CardHeader title={s.title} />
            <div className="px-6 py-5"><Prose text={report[s.key]} /></div>
          </Card>
        ))}
      </div>

      {products.length > 0 && (
        <Card>
          <CardHeader title={`Top contributors by ${group_col}`} subtitle="Share of total" />
          <div className="px-4 py-4" style={{ height: products.length * 36 + 40 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={products} layout="vertical" margin={{ top: 0, right: 24, left: 8, bottom: 0 }} barSize={16}>
                <CartesianGrid stroke="#eef2f6" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} unit="%" />
                <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12, fill: "#334155" }} tickLine={false} axisLine={false} />
                <Tooltip formatter={(v) => [`${v}%`, "Share"]} cursor={{ fill: "rgba(99,102,241,0.06)" }} />
                <Bar dataKey="share" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {products.map((p, i) => <Cell key={p.name} fill={i === 0 ? "#4f46e5" : "#a5b4fc"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  );
}
