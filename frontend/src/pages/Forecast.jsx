import { useState } from "react";
import { ArrowLeft, Download, FileJson } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import PageHeader, { PageBody } from "@/components/shell/PageHeader";
import ForecastChart from "@/components/ForecastChart";
import { Banner, Button, Panel, PanelBody, PanelHeader, Segmented } from "@/components/ui";
import { useAppState } from "@/state/AppState";
import { useChartPalette } from "@/lib/theme";
import { forecastSummary } from "@/lib/insights";
import { fmtCompact, fmtDate, fmtNumber, fmtSignedPct, isNum, modelLabel } from "@/lib/format";

const RANGES = [
  { value: 90, label: "90" },
  { value: 180, label: "180" },
  { value: 365, label: "365" },
];

function download(name, type, content) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function exportCsv(result) {
  const b = result.best_model;
  const rows = [["date", "forecast", "lower_95", "upper_95"]];
  result.forecast_dates.forEach((d, i) => rows.push([d, b.forecast[i], b.forecast_lower?.[i] ?? "", b.forecast_upper?.[i] ?? ""]));
  const suffix = result.group ? `_${String(result.group.group_value).replace(/[^\w-]+/g, "_")}` : "";
  download(`forecast_${result.target_col || "series"}${suffix}.csv`, "text/csv", rows.map((r) => r.join(",")).join("\n"));
}

function Drivers({ best }) {
  const palette = useChartPalette();
  const data = Object.entries(best?.winner_details?.feature_importance || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([feature, value]) => ({ feature: feature.replaceAll("_", " "), value: Number(value) }));
  if (!data.length) return null;
  return (
    <Panel>
      <PanelHeader title="Forecast drivers" subtitle={`Feature importance in ${modelLabel(best.best_model_name)}`} />
      <PanelBody>
        <div style={{ height: data.length * 30 + 24 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, left: 0, bottom: 0 }} barSize={12}>
              <CartesianGrid stroke={palette.grid} horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: palette.axis }} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="feature" width={120} tick={{ fontSize: 12, fill: palette.ink2 }} tickLine={false} axisLine={false} />
              <Tooltip
                formatter={(v) => [v.toFixed(4), "Importance"]}
                cursor={{ fill: palette.grid }}
                contentStyle={{ background: palette.surface, border: `1px solid ${palette.line}`, borderRadius: 4, fontSize: 12 }}
              />
              <Bar dataKey="value" fill={palette.forecast} radius={[0, 2, 2, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </PanelBody>
    </Panel>
  );
}

export default function ForecastPage() {
  const { view, runMeta, groupResult, clearGroup } = useAppState();
  const [range, setRange] = useState(180);
  if (!view) return null;

  const s = forecastSummary(view);
  const best = view.best_model;
  const hasDrivers = Object.keys(best?.winner_details?.feature_importance || {}).length > 0;

  return (
    <>
      <PageHeader
        title={view.group ? `Forecast: ${view.group.group_value}` : "Forecast"}
        description={`${view.target_col || runMeta.targetCol} · ${s.n} periods ahead`}
        meta={[
          ["Selected model", modelLabel(best?.best_model_name)],
          ["MASE", isNum(best?.mase) ? best.mase.toFixed(3) : "—"],
          ["Forecast total", fmtCompact(s.total)],
          ["vs previous period", fmtSignedPct(s.change)],
          ["Period", `${fmtDate(view.forecast_dates[0])} – ${fmtDate(view.forecast_dates.at(-1))}`],
        ]}
        actions={
          <>
            <Button icon={FileJson} onClick={() => download("forecast_results.json", "application/json", JSON.stringify(view, null, 2))}>
              Export JSON
            </Button>
            <Button variant="primary" icon={Download} onClick={() => exportCsv(view)}>
              Download CSV
            </Button>
          </>
        }
      />
      <PageBody>
        {groupResult && (
          <Banner
            tone="info"
            title={`Showing ${groupResult.group.group_col} = ${groupResult.group.group_value}`}
            action={
              <Button size="sm" icon={ArrowLeft} onClick={clearGroup}>
                Back to all data
              </Button>
            }
          >
            This forecast was trained on this group only. The dashboard and report still use all data.
          </Banner>
        )}

        <Panel>
          <PanelHeader
            title="Actual vs forecast"
            subtitle="Shaded area is the 95% range, calibrated on the model's own backtest errors."
            actions={
              <>
                <span className="hidden text-2xs text-ink-3 sm:inline">Periods of history</span>
                <Segmented value={range} onChange={setRange} options={RANGES} />
              </>
            }
          />
          <PanelBody className="pt-3">
            <ForecastChart result={view} range={range} height={380} />
          </PanelBody>
        </Panel>

        <div className={`grid gap-4 ${hasDrivers ? "xl:grid-cols-2" : ""}`}>
          <Panel>
            <PanelHeader title="Forecast values" subtitle={`${s.n} periods`} />
            <div className="scroll-thin max-h-[420px] overflow-y-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th className="num">Forecast</th>
                    <th className="num">Lower 95%</th>
                    <th className="num">Upper 95%</th>
                  </tr>
                </thead>
                <tbody>
                  {view.forecast_dates.map((d, i) => (
                    <tr key={d}>
                      <td>{fmtDate(d)}</td>
                      <td className="num font-medium">{fmtNumber(best.forecast[i])}</td>
                      <td className="num text-ink-2">{fmtNumber(best.forecast_lower?.[i])}</td>
                      <td className="num text-ink-2">{fmtNumber(best.forecast_upper?.[i])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          {hasDrivers && <Drivers best={best} />}
        </div>
      </PageBody>
    </>
  );
}
