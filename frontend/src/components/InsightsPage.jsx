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
import { generateInsightsReport, getAsyncReportStatus, getProductSummary, errorMessage } from "../api";
import Markdown from "./Markdown";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft, AlertCircle, RefreshCw } from "lucide-react";

const PRODUCT_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"];
const OTHER_COLOR = "#8a8a86";
const TREND_HUE = "#2a78d6";
const CRITICAL = "#d03b3b";

const SECTION_META = [
  { key: "dataset_summary", title: "Dataset Overview" },
  { key: "sales_forecast_summary", title: "Sales Forecast Outlook" },
  { key: "recommendations", title: "Recommendations" },
  { key: "profit_loss_analysis", title: "Profit & Loss Signal" },
  { key: "stock_alerts_summary", title: "Stock Alerts" },
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
        // The backend also writes a fuller report in the background (longer
        // LLM retries). Poll for it and swap it in when it lands.
        if (data.async_report_path) {
          let polls = 0;
          pollTimer = setInterval(async () => {
            polls += 1;
            if (polls > 60) return clearInterval(pollTimer); // give up after ~2 minutes
            try {
              const status = await getAsyncReportStatus(filename);
              if (status.ready && status.report && !cancelled) {
                // The background report has no chart data; keep the first one's.
                setReport((prev) => ({ ...status.report, chart_data: prev?.chart_data, stockout_alerts: status.report.stockout_alerts ?? prev?.stockout_alerts }));
                setFullReady(true);
                clearInterval(pollTimer);
              }
            } catch {
              /* transient: keep polling */
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

  const backButton = (
    <Button
      variant="ghost"
      onClick={onBack}
      className="mb-6 -ml-4 text-slate-600 hover:text-blue-600"
    >
      <ArrowLeft className="mr-2 h-4 w-4" /> Back to Results
    </Button>
  );

  if (loading) {
    return (
      <div className="space-y-6">
        {backButton}
        <Card className="shadow-sm border-slate-200">
          <CardContent className="flex flex-col items-center gap-3 py-16">
            <Loader2 className="h-10 w-10 animate-spin text-blue-600" />
            <p className="text-sm font-medium text-slate-900">Generating your AI insights report...</p>
            <p className="text-xs text-slate-500">Analyzing your dataset and retrieving relevant retail knowledge</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        {backButton}
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Couldn't generate the report</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button variant="outline" onClick={() => setAttempt((a) => a + 1)} className="gap-2">
          <RefreshCw className="h-4 w-4" /> Try again
        </Button>
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

      <Alert className="border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50 shadow-sm border-l-4 border-l-blue-500">
        <AlertTitle className="text-2xl font-bold text-blue-900">AI Insights Report</AlertTitle>
        <AlertDescription className="mt-1 text-sm text-blue-800 font-medium">
          A proactive briefing generated from your dataset, forecast, and retrieved retail domain knowledge.
          {!fullReady && report?.async_report_path && (
            <span className="ml-2 inline-flex items-center gap-1 text-xs text-blue-700">
              <Loader2 className="h-3 w-3 animate-spin" /> Refining with a fuller analysis…
            </span>
          )}
        </AlertDescription>
      </Alert>

      {/* Historical Sales Trend */}
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-slate-900">Historical Sales Trend</CardTitle>
          <p className="text-sm text-slate-500">
            {alerts.length > 0
              ? `${alerts.length} possible stockout-risk period${alerts.length > 1 ? "s" : ""} highlighted below.`
              : "No stockout-risk patterns detected in this history."}
          </p>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full">
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
        </CardContent>
      </Card>

      {/* Sales by Product */}
      {productChartData.length > 0 && (
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-slate-900">Sales Share by Product</CardTitle>
            <p className="text-sm text-slate-500">Percent of total sales, top products individually and the rest folded into "Other".</p>
          </CardHeader>
          <CardContent>
            <div className="h-72 w-full">
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
          </CardContent>
        </Card>
      )}

      {/* Report sections */}
      <div className="grid gap-6 md:grid-cols-2">
        {SECTION_META.filter(({ key }) => report[key]).map(({ key, title }) => (
          <Card key={key} className="shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-base font-semibold text-slate-900">{title}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm leading-relaxed text-slate-700">
                <Markdown text={report[key]} />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
