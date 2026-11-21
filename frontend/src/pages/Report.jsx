import { useEffect } from "react";
import { Loader2, Printer, RefreshCw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import PageHeader, { PageBody } from "@/components/shell/PageHeader";
import Markdown from "@/components/Markdown";
import { Banner, Button, Chip, Loading, Panel, PanelBody, PanelHeader } from "@/components/ui";
import { navigate } from "@/lib/router";
import { useChartPalette } from "@/lib/theme";
import { useAppState } from "@/state/AppState";
import { llmUnavailableReason } from "@/lib/insights";
import { fmtCompact, fmtDate, fmtNumber, fmtShortDate } from "@/lib/format";

const SECTIONS = [
  { key: "dataset_summary", title: "Dataset overview" },
  { key: "sales_forecast_summary", title: "Forecast outlook" },
  { key: "recommendations", title: "Recommendations", wide: true },
  { key: "profit_loss_analysis", title: "Revenue and trend signal" },
  { key: "stock_alerts_summary", title: "Stock alerts" },
];

function TrendPanel({ report, targetCol }) {
  const palette = useChartPalette();
  const alerts = report.stockout_alerts || [];
  const data = (report.chart_data?.dates || []).map((date, i) => ({ date, value: report.chart_data.values[i] }));
  if (!data.length) return null;
  const tick = { fontSize: 11, fill: palette.axis };
  return (
    <Panel className="xl:col-span-2">
      <PanelHeader
        title="Historical trend"
        subtitle={
          alerts.length
            ? `${alerts.length} possible stockout period${alerts.length > 1 ? "s" : ""} highlighted`
            : "No stockout-risk patterns detected in this history"
        }
        actions={alerts.length > 0 && <Chip tone="bad">{alerts.length} alerts</Chip>}
      />
      <PanelBody>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={palette.grid} vertical={false} />
              <XAxis dataKey="date" tick={tick} tickLine={false} axisLine={{ stroke: palette.line }} minTickGap={56} tickFormatter={fmtShortDate} />
              <YAxis tick={tick} tickLine={false} axisLine={false} tickFormatter={fmtCompact} width={52} />
              <Tooltip
                formatter={(v) => [fmtNumber(Number(v)), targetCol || "Value"]}
                labelFormatter={fmtDate}
                contentStyle={{ background: palette.surface, border: `1px solid ${palette.line}`, borderRadius: 4, fontSize: 12 }}
              />
              {alerts.map((a, i) => (
                <ReferenceArea key={i} x1={a.start_date} x2={a.end_date} fill={palette.alert} fillOpacity={0.12} stroke={palette.alert} strokeOpacity={0.35} />
              ))}
              <Line type="monotone" dataKey="value" stroke={palette.forecast} strokeWidth={1.75} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </PanelBody>
    </Panel>
  );
}

function SharePanel({ groups, groupCol }) {
  const palette = useChartPalette();
  const rows = (groups || []).slice(0, 6);
  if (!rows.length) return null;
  const top = rows.slice(0, 4);
  const other = rows.slice(4).reduce((s, r) => s + (r.share_pct || 0), 0);
  const data = [...top.map((r) => ({ name: r.group_value, share: r.share_pct })), ...(other > 0 ? [{ name: "Other", share: Math.round(other * 100) / 100 }] : [])];
  const colors = [palette.c1, palette.c2, palette.c3, palette.c4];
  return (
    <Panel>
      <PanelHeader title={`Share by ${groupCol}`} subtitle="Percent of total, top four shown individually" />
      <PanelBody>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, left: 0, bottom: 0 }} barSize={16}>
              <CartesianGrid stroke={palette.grid} horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: palette.axis }} tickLine={false} axisLine={false} unit="%" />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fill: palette.ink2 }} tickLine={false} axisLine={false} width={110} />
              <Tooltip
                formatter={(v) => [`${v}%`, "Share"]}
                cursor={{ fill: palette.grid }}
                contentStyle={{ background: palette.surface, border: `1px solid ${palette.line}`, borderRadius: 4, fontSize: 12 }}
              />
              <Bar dataKey="share" radius={[0, 2, 2, 0]} isAnimationActive={false}>
                {data.map((d, i) => (
                  <Cell key={d.name} fill={d.name === "Other" ? palette.other : colors[i % colors.length]} />
                ))}
                <LabelList dataKey="share" position="right" formatter={(v) => `${v}%`} style={{ fill: palette.ink2, fontSize: 11 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </PanelBody>
    </Panel>
  );
}

// When the language model is unavailable the backend still returns 200, with
// fallback text in the narrative sections. Detect that so we can show one
// clear notice instead of raw provider errors.
const FALLBACK_MARKER = "AI insights temporarily unavailable";
const NARRATIVE_KEYS = ["dataset_summary", "sales_forecast_summary", "recommendations"];


export default function ReportPage() {
  const { report, loadReport, runMeta, products } = useAppState();

  useEffect(() => {
    loadReport();
  }, [loadReport]);

  const data = report.data;
  const keyMissing = /api key|GEMINI/i.test(report.error || "");
  const degraded = String(data?.dataset_summary || "").startsWith(FALLBACK_MARKER);
  const sectionText = (key) => {
    const t = data?.[key];
    if (degraded && key === "profit_loss_analysis") return String(t).replace(/^.*Deterministic trend is:\s*/i, "Overall trend: ");
    return t;
  };
  const sections = SECTIONS.filter((s) => data?.[s.key] && !(degraded && NARRATIVE_KEYS.includes(s.key)));

  return (
    <>
      <PageHeader
        title="Insights report"
        description="A briefing generated from this run's data and forecast, grounded in the retail knowledge base."
        meta={data && [["Dataset", runMeta?.fileName], ["Target", runMeta?.targetCol], ["Generated", report.at ? new Date(report.at).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "—"]]}
        actions={
          <>
            {report.refining && (
              <span className="inline-flex items-center gap-1.5 text-xs text-ink-3">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Refining with a fuller analysis
              </span>
            )}
            <Button icon={RefreshCw} disabled={report.loading} onClick={() => loadReport({ force: true })}>
              Regenerate
            </Button>
            <Button icon={Printer} disabled={!data} onClick={() => window.print()}>
              Print / PDF
            </Button>
          </>
        }
      />
      <PageBody>
        {report.loading && (
          <Panel>
            <Loading label="Generating report. This usually takes 10–30 seconds…" className="py-16" />
          </Panel>
        )}

        {report.error && (
          <Banner
            tone="bad"
            title="Couldn't generate the report"
            action={
              <Button size="sm" onClick={() => (keyMissing ? navigate("settings") : loadReport({ force: true }))}>
                {keyMissing ? "Open settings" : "Try again"}
              </Button>
            }
          >
            {report.error}
            {keyMissing && <p className="mt-1">The report and Assistant need a Gemini API key configured on the forecasting server.</p>}
          </Banner>
        )}

        {data && degraded && (
          <Banner
            tone="warn"
            title="Written analysis unavailable"
            action={
              <Button size="sm" onClick={() => navigate("settings")}>
                Open settings
              </Button>
            }
          >
            {llmUnavailableReason(data.dataset_summary)} The charts, trend and stock alerts below are computed directly from your data.
          </Banner>
        )}

        {data && (
          <>
            <div className="grid gap-4 xl:grid-cols-3">
              <TrendPanel report={data} targetCol={runMeta?.targetCol} />
              <SharePanel groups={products.data?.groups} groupCol={runMeta?.groupCol} />
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              {sections.map((s) => (
                <Panel key={s.key} className={s.wide ? "lg:col-span-2" : ""}>
                  <PanelHeader title={s.title} />
                  <PanelBody className="text-[13px] leading-relaxed text-ink-2">
                    <Markdown text={sectionText(s.key)} />
                  </PanelBody>
                </Panel>
              ))}
            </div>
          </>
        )}
      </PageBody>
    </>
  );
}
