import { useEffect, useState } from "react";
import {
  Line,
  LineChart,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";
import { generateInsightsReport, getProductSummary } from "../api";

// Validated categorical palette (dataviz skill reference palette, light mode,
// first 4 slots — the only span that clears all-pairs CVD separation).
// Direct value labels are used throughout per the palette's contrast WARN
// for the magenta/yellow slots against a light surface.
const PRODUCT_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"];
const OTHER_COLOR = "#8a8a86";
const TREND_HUE = "#2a78d6"; // sequential blue, single series — no legend needed
const CRITICAL = "#d03b3b"; // status palette — stockout risk overlay

const SECTION_META = [
  { key: "dataset_summary", icon: "📄", title: "Dataset Overview" },
  { key: "sales_forecast_summary", icon: "📈", title: "Sales Forecast Outlook" },
  { key: "recommendations", icon: "💡", title: "Recommendations" },
  { key: "profit_loss_analysis", icon: "💰", title: "Profit & Loss Signal" },
  { key: "stock_alerts_summary", icon: "📦", title: "Stock Alerts" },
];

export default function InsightsPage({
  filename,
  date_col,
  target_col,
  group_col,
  forecast_result,
  onBack,
}) {
  const [report, setReport] = useState(null);
  const [productSummary, setProductSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    // If a grouping column was used, fetch the per-product ranking first
    // (same data ProductSummaryTable/ChatBot already use) so the report and
    // its product chart can be grounded in real per-product numbers.
    const productPromise = group_col
      ? getProductSummary(filename, date_col, target_col, group_col, 50).catch(() => null)
      : Promise.resolve(null);

    productPromise.then((summary) => {
      if (cancelled) return;
      setProductSummary(summary);
      generateInsightsReport(filename, date_col, target_col, group_col, forecast_result, summary)
        .then((data) => {
          if (!cancelled) setReport(data);
        })
        .catch((err) => {
          if (!cancelled) {
            setError(err?.response?.data?.detail || err?.message || "Failed to generate the insights report.");
          }
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filename, date_col, target_col, group_col]);

  const backButton = (
    <button
      onClick={onBack}
      className="mb-6 flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-blue-600"
    >
      ← Back to Results
    </button>
  );

  if (loading) {
    return (
      <div className="space-y-6">
        {backButton}
        <div className="flex flex-col items-center gap-3 rounded-xl border border-slate-200 bg-white py-16 shadow-sm">
          <svg className="h-10 w-10 animate-spin text-blue-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="text-sm font-medium text-slate-900">Generating your AI insights report...</p>
          <p className="text-xs text-slate-600">Analyzing your dataset and retrieving relevant retail knowledge</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        {backButton}
        <div className="rounded-xl border border-red-300 bg-red-50 p-6 text-sm text-red-800 shadow-sm">
          <p className="font-semibold">Couldn't generate the report</p>
          <p className="mt-1">{error}</p>
        </div>
      </div>
    );
  }

  if (!report) return <div className="space-y-6">{backButton}</div>;

  const trendData = (report.chart_data?.dates || []).map((date, i) => ({
    date,
    value: report.chart_data.values[i],
  }));

  const alerts = report.stockout_alerts || [];

  const productRows = (productSummary?.groups || []).slice(0, 6);
  const topProducts = productRows.slice(0, 4);
  const otherShare = productRows.slice(4).reduce((sum, r) => sum + (r.share_pct || 0), 0);
  const productChartData = [
    ...topProducts.map((r) => ({ name: r.group_value, share: r.share_pct })),
    ...(otherShare > 0 ? [{ name: "Other", share: Math.round(otherShare * 100) / 100 }] : []),
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {backButton}

      <div className="rounded-xl border-l-4 border-l-blue-500 bg-gradient-to-r from-blue-50 to-indigo-50 p-6 shadow-sm">
        <h2 className="text-2xl font-bold text-blue-900">📋 AI Insights Report</h2>
        <p className="mt-1 text-sm text-blue-800">
          A proactive briefing generated from your dataset, forecast, and retrieved retail domain knowledge.
        </p>
      </div>

      {/* Historical Sales Trend */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900">Historical Sales Trend</h3>
        <p className="mt-1 text-sm text-slate-600">
          {alerts.length > 0
            ? `${alerts.length} possible stockout-risk period${alerts.length > 1 ? "s" : ""} highlighted below.`
            : "No stockout-risk patterns detected in this history."}
        </p>
        <div className="mt-4 h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trendData} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} angle={-45} textAnchor="end" height={70} minTickGap={30} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#f8fafc", border: "1px solid #cbd5e1", borderRadius: "8px", padding: "10px" }}
                formatter={(value) => [Number(value).toFixed(2), target_col || "Value"]}
                labelFormatter={(label) => `Date: ${label}`}
              />
              {alerts.map((a, i) => (
                <ReferenceArea
                  key={i}
                  x1={a.start_date}
                  x2={a.end_date}
                  fill={CRITICAL}
                  fillOpacity={0.12}
                  stroke={CRITICAL}
                  strokeOpacity={0.4}
                  label={{ value: "Possible stockout", position: "insideTop", fontSize: 10, fill: CRITICAL }}
                />
              ))}
              <Line type="monotone" dataKey="value" stroke={TREND_HUE} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sales by Product (only if a grouping column was used) */}
      {productChartData.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900">Sales Share by Product</h3>
          <p className="mt-1 text-sm text-slate-600">Percent of total sales, top products individually and the rest folded into "Other".</p>
          <div className="mt-4 h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={productChartData} layout="vertical" margin={{ top: 5, right: 40, left: 10, bottom: 5 }} barSize={22}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} unit="%" />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={100} />
                <Tooltip formatter={(value) => [`${value}%`, "Share"]} />
                <Bar dataKey="share" radius={[0, 4, 4, 0]}>
                  {productChartData.map((row, i) => (
                    <Cell key={row.name} fill={row.name === "Other" ? OTHER_COLOR : PRODUCT_COLORS[i % PRODUCT_COLORS.length]} />
                  ))}
                  <LabelList dataKey="share" position="right" formatter={(v) => `${v}%`} style={{ fill: "#334155", fontSize: 12 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Report sections */}
      <div className="grid gap-6 md:grid-cols-2">
        {SECTION_META.map(({ key, icon, title }) => (
          <div key={key} className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <h3 className="flex items-center gap-2 text-base font-semibold text-slate-900">
              <span>{icon}</span> {title}
            </h3>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
              {report[key] || "No information available for this section."}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
