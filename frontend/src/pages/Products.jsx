import { useMemo, useState } from "react";
import { Minus, Play, Search, TrendingDown, TrendingUp } from "lucide-react";
import PageHeader, { PageBody } from "@/components/shell/PageHeader";
import { Banner, Button, Chip, Loading, Panel, PanelHeader, Segmented, TextInput } from "@/components/ui";
import { navigate } from "@/lib/router";
import { useAppState } from "@/state/AppState";
import { fmtNumber, fmtShare } from "@/lib/format";

const TREND = {
  growing: { tone: "ok", label: "Growing", icon: TrendingUp },
  declining: { tone: "bad", label: "Declining", icon: TrendingDown },
  flat: { tone: "neutral", label: "Flat", icon: Minus },
  insufficient_data: { tone: "neutral", label: "Not enough data", icon: Minus },
};

export function TrendChip({ trend }) {
  const t = TREND[trend] || TREND.flat;
  return (
    <Chip tone={t.tone} icon={t.icon}>
      {t.label}
    </Chip>
  );
}

export default function ProductsPage() {
  const { runMeta, products, forecastGroup, groupRunning, groupError } = useAppState();
  const [query, setQuery] = useState("");
  const [trend, setTrend] = useState("all");
  const groups = products.data?.groups || [];
  const col = runMeta?.groupCol;

  const counts = useMemo(() => {
    const c = { growing: 0, declining: 0, flat: 0 };
    groups.forEach((g) => (c[g.trend] = (c[g.trend] || 0) + 1));
    return c;
  }, [groups]);

  const rows = groups.filter(
    (g) => (trend === "all" || g.trend === trend) && String(g.group_value).toLowerCase().includes(query.trim().toLowerCase())
  );
  const maxShare = Math.max(...groups.map((g) => g.share_pct || 0), 1);

  const runGroup = async (value) => {
    if (await forecastGroup(value)) navigate("forecast");
  };

  return (
    <>
      <PageHeader
        title="Product breakdown"
        description={`${runMeta?.targetCol} by ${col}, ranked by total. Trend compares recent to earlier periods (±5% threshold).`}
        meta={
          products.data && [
            [`${col} values`, fmtNumber(products.data.total_groups)],
            ["Growing", counts.growing],
            ["Flat", counts.flat],
            ["Declining", counts.declining],
          ]
        }
      />
      <PageBody>
        {groupError && <Banner tone="bad" title="Group forecast failed">{groupError}</Banner>}
        {products.error && <Banner tone="bad" title="Couldn't load the breakdown">{products.error}</Banner>}
        <Panel>
          <PanelHeader
            title={`Sales by ${col}`}
            subtitle={products.data?.truncated ? `Showing the top ${groups.length} of ${products.data.total_groups}` : undefined}
            actions={
              <>
                <div className="relative">
                  <Search className="pointer-events-none absolute left-2 top-2 h-4 w-4 text-ink-3" />
                  <TextInput value={query} onChange={(e) => setQuery(e.target.value)} placeholder={`Search ${col}`} className="w-48 pl-7" />
                </div>
                <Segmented
                  value={trend}
                  onChange={setTrend}
                  options={[
                    { value: "all", label: "All" },
                    { value: "growing", label: "Growing" },
                    { value: "flat", label: "Flat" },
                    { value: "declining", label: "Declining" },
                  ]}
                />
              </>
            }
          />
          {products.loading ? (
            <Loading label={`Loading ${col} breakdown…`} />
          ) : (
            <div className="scroll-thin overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th className="w-12">Rank</th>
                    <th>{col}</th>
                    <th className="num">Total</th>
                    <th className="num">Share</th>
                    <th>Trend</th>
                    <th className="num">Growth</th>
                    <th className="w-28" />
                  </tr>
                </thead>
                <tbody>
                  {rows.map((g) => (
                    <tr key={g.group_value}>
                      <td className="text-ink-3">{g.rank}</td>
                      <td className="font-medium">{g.group_value}</td>
                      <td className="num">{fmtNumber(g.total)}</td>
                      <td className="num">
                        <div className="flex items-center justify-end gap-2">
                          <div className="hidden h-1.5 w-20 rounded-full bg-surface-muted md:block">
                            <div className="h-full rounded-full bg-brand" style={{ width: `${(g.share_pct / maxShare) * 100}%` }} />
                          </div>
                          <span className="w-12">{fmtShare(g.share_pct)}</span>
                        </div>
                      </td>
                      <td>
                        <TrendChip trend={g.trend} />
                      </td>
                      <td className={`num ${g.growth_rate_pct > 0 ? "text-ok" : g.growth_rate_pct < 0 ? "text-bad" : "text-ink-2"}`}>
                        {g.growth_rate_pct == null ? "—" : `${g.growth_rate_pct > 0 ? "+" : ""}${Number(g.growth_rate_pct).toFixed(1)}%`}
                      </td>
                      <td className="text-right">
                        <Button
                          size="sm"
                          icon={Play}
                          loading={groupRunning === g.group_value}
                          disabled={groupRunning !== null || g.trend === "insufficient_data"}
                          onClick={() => runGroup(g.group_value)}
                        >
                          {groupRunning === g.group_value ? "Running" : "Forecast"}
                        </Button>
                      </td>
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={7} className="py-8 text-center text-ink-3">
                        No {col} values match.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </PageBody>
    </>
  );
}
