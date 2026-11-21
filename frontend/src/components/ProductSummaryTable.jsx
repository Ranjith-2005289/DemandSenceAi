import { useEffect, useState } from "react";
import { getProductSummary, runForecast } from "../api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { TrendingUp, TrendingDown, Minus, AlertCircle, Loader2, Play } from "lucide-react";

export default function ProductSummaryTable({
  filename,
  date_col,
  target_col,
  group_col,
  forecast_horizon,
  country,
  sessionId,
  onForecastGroup,
}) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forecastingGroup, setForecastingGroup] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getProductSummary(filename, date_col, target_col, group_col, 50)
      .then((data) => {
        if (!cancelled) setSummary(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err?.response?.data?.detail || err?.message || "Failed to load product summary.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [filename, date_col, target_col, group_col]);

  const handleForecastGroup = async (groupValue) => {
    setForecastingGroup(groupValue);
    try {
      const result = await runForecast(
        filename,
        date_col,
        target_col,
        forecast_horizon,
        group_col,
        groupValue,
        country || null,
        sessionId || null
      );
      onForecastGroup?.(result);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Forecast failed for this group.");
    } finally {
      setForecastingGroup(null);
    }
  };

  const getTrendStyle = (trend) => {
    switch (trend) {
      case "growing":
        return { label: "Growing", icon: <TrendingUp className="mr-1 h-3 w-3" />, className: "bg-green-100 text-green-800 border-green-200" };
      case "declining":
        return { label: "Declining", icon: <TrendingDown className="mr-1 h-3 w-3" />, className: "bg-red-100 text-red-800 border-red-200" };
      case "insufficient_data":
        return { label: "Not enough data", icon: <Minus className="mr-1 h-3 w-3" />, className: "bg-slate-100 text-slate-500 border-slate-200" };
      default:
        return { label: "Flat", icon: <Minus className="mr-1 h-3 w-3" />, className: "bg-slate-100 text-slate-700 border-slate-200" };
    }
  };

  if (loading) {
    return (
      <Card className="shadow-sm">
        <CardContent className="flex items-center justify-center p-6 text-sm text-slate-600">
          <Loader2 className="mr-2 h-4 w-4 animate-spin text-blue-500" /> Loading {group_col} breakdown...
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="mb-4">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!summary?.groups?.length) return null;

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold text-slate-900">
          Sales by {group_col}
        </CardTitle>
        <p className="text-xs text-slate-600">
          {summary.total_groups} total
          {summary.truncated ? ` (showing top ${summary.groups.length})` : ""}
        </p>
      </CardHeader>

      <CardContent>
        <div className="rounded-md border border-slate-200">
          <Table>
            <TableHeader className="bg-slate-50">
              <TableRow>
                <TableHead className="w-16 text-center font-semibold text-slate-700">Rank</TableHead>
                <TableHead className="font-semibold text-slate-700">{group_col}</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">Total</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">Share</TableHead>
                <TableHead className="font-semibold text-slate-700">Trend</TableHead>
                <TableHead className="text-right font-semibold text-slate-700">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {summary.groups.map((row) => {
                const trend = getTrendStyle(row.trend);
                const isForecasting = forecastingGroup === row.group_value;

                return (
                  <TableRow key={row.group_value}>
                    <TableCell className="text-center font-mono text-slate-500">{row.rank}</TableCell>
                    <TableCell className="font-medium text-slate-900">{row.group_value}</TableCell>
                    <TableCell className="text-right font-mono text-slate-900">
                      {row.total.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </TableCell>
                    <TableCell className="text-right font-mono text-slate-700">{row.share_pct}%</TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        <Badge variant="outline" className={trend.className}>
                          {trend.icon} {trend.label}
                        </Badge>
                        {row.growth_rate_pct != null && (
                          <span className="ml-2 text-xs font-mono text-slate-500">
                            {row.growth_rate_pct > 0 ? "+" : ""}
                            {row.growth_rate_pct}%
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => handleForecastGroup(row.group_value)}
                        disabled={forecastingGroup !== null || row.trend === "insufficient_data"}
                        className="bg-blue-50 text-blue-900 hover:bg-blue-100 border-blue-200 h-8"
                      >
                        {isForecasting ? (
                          <>
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                            Running
                          </>
                        ) : (
                          <>
                            <Play className="mr-1 h-3 w-3" />
                            Forecast
                          </>
                        )}
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
