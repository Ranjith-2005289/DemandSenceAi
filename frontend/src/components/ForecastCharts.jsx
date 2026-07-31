import { useState } from "react";
import {
  Line,
  LineChart,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function ForecastCharts({ results }) {
  const [activeChart, setActiveChart] = useState("line");

  if (!results?.best_model?.forecast?.length || !results?.forecast_dates?.length) {
    return null;
  }

  // ── Chart 1: Line Chart Data (Historical + Forecast) ────────────────────────
  // Built entirely from real data returned by the backend — the previous
  // version fabricated 30 "historical" points with Math.random().
  const buildLineChartData = () => {
    const historicalDates = results.historical?.dates || [];
    const historicalValues = results.historical?.values || [];

    const historical = historicalDates.map((date, idx) => ({
      date,
      historical: historicalValues[idx],
      forecast: null,
    }));

    const forecast = results.forecast_dates.map((date, idx) => ({
      date,
      historical: null,
      forecast: results.best_model.forecast[idx],
    }));

    return [...historical, ...forecast];
  };

  // ── Chart 2: RMSE Comparison Data ──────────────────────────────────────────
  // Comprehensive color palette for 15+ models
  const modelColors = {
    // Statistical Models (6 colors)
    "ARIMA": "#8b5cf6",         // Purple
    "SARIMA": "#7c3aed",        // Violet
    "SARIMAX": "#a855f7",       // Fuchsia
    "HoltWinters": "#d946ef",   // Pink
    "ExponentialSmoothing": "#ec4899", // Rose
    "MovingAverage": "#f472b6", // Lighter Pink
    
    // ML Models (4 colors)
    "XGBoost": "#ef4444",       // Red
    "RandomForest": "#3b82f6",  // Blue
    "KNN": "#14b8a6",           // Teal
    "SVM": "#06b6d4",           // Cyan
    
    // DL Models (7 colors)
    "RNN": "#f59e0b",           // Amber
    "LSTM": "#f97316",          // Orange
    "GRU": "#ea580c",           // Dark Orange
    "BiLSTM": "#dc2626",        // Dark Red
    "CNN1D": "#10b981",         // Emerald
    "TCN": "#059669",           // Teal-Green
    "Transformer": "#0d9488",   // Dark Teal
    
    // Prophet (legacy)
    "Prophet": "#14b8a6",       // Teal
  };
  
  // Generate dynamic colors if a model isn't in our palette
  const getModelColor = (modelName) => {
    if (modelColors[modelName]) {
      return modelColors[modelName];
    }
    
    // Create a consistent color from model name hash
    let hash = 0;
    for (let i = 0; i < modelName.length; i++) {
      hash = ((hash << 5) - hash) + modelName.charCodeAt(i);
      hash = hash & hash; // Convert to 32bit integer
    }
    
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 70%, 50%)`;
  };

  const rmseData = (results.all_models || [])
    .filter((m) => m.status === "success" && Number.isFinite(m.rmse))
    .sort((a, b) => a.rmse - b.rmse)
    .map((m) => ({
      name: m.model_name,
      // RMSE can span several orders of magnitude across models (a badly-fit
      // statistical model can be 1000x worse than a good one) — floor at a
      // small positive value so a log-scale axis never has to plot zero.
      rmse: Math.max(parseFloat(m.rmse.toFixed(4)), 0.0001),
      isBest: m.model_name === results.best_model?.best_model_name,
      fill: getModelColor(m.model_name),
    }));

  // ── Chart 3: Confidence Band Data (real, model-specific interval) ──────────
  // No fabricated fallback — if the winning model genuinely has no interval
  // (e.g. its RMSE was non-finite), the band is simply absent rather than
  // invented.
  const hasRealInterval =
    Array.isArray(results.best_model.forecast_lower) &&
    Array.isArray(results.best_model.forecast_upper);

  const confidenceData = hasRealInterval
    ? results.forecast_dates.map((date, idx) => ({
        date,
        forecast: results.best_model.forecast[idx],
        upper: results.best_model.forecast_upper[idx],
        lower: results.best_model.forecast_lower[idx],
      }))
    : [];

  // ── Chart 4: Feature Importance (only for models that compute it) ─────────
  const featureImportance = results.best_model?.winner_details?.feature_importance;
  const hasFeatureImportance =
    featureImportance && Object.keys(featureImportance).length > 0;

  const featureImportanceData = hasFeatureImportance
    ? Object.entries(featureImportance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([feature, importance]) => ({
          feature,
          importance: Number(importance),
        }))
        .reverse() // reverse so the highest bar renders at the top in a vertical-layout BarChart
    : [];

  const lineChartData = buildLineChartData();

  return (
    <section className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-slate-200">
        {[
          { id: "line", label: "📈 Historical & Forecast" },
          { id: "bar", label: "📊 Model Comparison" },
          { id: "area", label: "📉 Confidence Band" },
          ...(hasFeatureImportance
            ? [{ id: "importance", label: "🔑 Feature Importance" }]
            : []),
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveChart(tab.id)}
            className={`px-4 py-2 text-sm font-medium transition-all ${
              activeChart === tab.id
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Chart 1: Line Chart */}
      {activeChart === "line" && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">
            Historical Data & Forecast
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Blue line: Historical data | Orange dashed: Forecast ({results.best_model.best_model_name})
          </p>
          <div className="mt-4 h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={lineChartData}
                margin={{ top: 5, right: 30, left: 0, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#f1f5f9",
                    border: "1px solid #cbd5e1",
                    borderRadius: "8px",
                    padding: "10px",
                  }}
                  formatter={(value) => value?.toFixed(2)}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="historical"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  dot={false}
                  name="Historical"
                  isAnimationActive={true}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#f97316"
                  strokeWidth={2.5}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Forecast"
                  isAnimationActive={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Chart 2: Bar Chart (RMSE Comparison) */}
      {activeChart === "bar" && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">
            Model Performance (RMSE)
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Lower RMSE indicates better forecast accuracy. Each color represents a different model.
            Log scale — RMSE often spans several orders of magnitude between models.
          </p>
          <div className="mt-4 h-96 w-full overflow-x-auto">
            <ResponsiveContainer width={Math.max(1000, rmseData.length * 80)} height="100%">
              <BarChart
                data={rmseData}
                margin={{ top: 20, right: 30, left: 0, bottom: 80 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11 }}
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis
                  scale="log"
                  domain={["auto", "auto"]}
                  allowDataOverflow
                  label={{ value: "RMSE (log scale)", angle: -90, position: "insideLeft", offset: 10 }}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#f1f5f9",
                    border: "2px solid #3b82f6",
                    borderRadius: "8px",
                    padding: "12px",
                    fontSize: "14px",
                  }}
                  formatter={(value) => [
                    value?.toFixed(4),
                    "RMSE"
                  ]}
                  labelFormatter={(label) => `Model: ${label}`}
                  cursor={{ fill: "rgba(59, 130, 246, 0.1)" }}
                />
                <Bar
                  dataKey="rmse"
                  radius={[8, 8, 0, 0]}
                  shape={(props) => {
                    const { x, y, width, height, fill } = props;
                    const data = rmseData[props.index];
                    return (
                      <rect
                        x={x}
                        y={y}
                        width={width}
                        height={height}
                        fill={data?.fill || fill}
                        rx={8}
                        ry={8}
                      />
                    );
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 space-y-2">
            {rmseData.map((model, idx) => (
              <div key={idx} className="rounded-lg border border-slate-200 bg-gradient-to-r from-slate-50 to-white p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div
                      className="h-4 w-4 rounded"
                      style={{ backgroundColor: model.fill }}
                    ></div>
                    <span className="font-semibold text-slate-900">{model.name}</span>
                    {model.isBest && (
                      <span className="ml-2 rounded-full bg-green-100 px-2 py-1 text-xs font-bold text-green-700">
                        🏆 BEST
                      </span>
                    )}
                  </div>
                  <span className="font-mono text-sm font-semibold text-slate-700">
                    RMSE: {model.rmse.toFixed(4)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Chart 3: Area Chart (Confidence Band) */}
      {activeChart === "area" && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">
            Forecast with Confidence Band
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            {hasRealInterval
              ? `Shaded area represents a 95% interval, derived from ${results.best_model.best_model_name}'s own ${
                  results.best_model.interval_method === "tree_variance"
                    ? "tree-variance"
                    : "historical cross-validation error"
                } — not a fixed percentage.`
              : `No interval data is available for ${results.best_model.best_model_name} on this run.`}
          </p>
          {hasRealInterval ? (
            <div className="mt-4 h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={confidenceData}
                  margin={{ top: 5, right: 30, left: 0, bottom: 60 }}
                >
                  <defs>
                    <linearGradient id="confidenceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#fbbf24" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11 }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#f1f5f9",
                      border: "1px solid #cbd5e1",
                      borderRadius: "8px",
                      padding: "10px",
                    }}
                    formatter={(value) => value?.toFixed(2)}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="upper"
                    stroke="none"
                    fill="url(#confidenceGrad)"
                    name="Upper Bound"
                  />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    stroke="none"
                    fill="white"
                    name="Lower Bound"
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="#f59e0b"
                    strokeWidth={3}
                    dot={false}
                    name="Forecast"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="mt-4 flex h-96 w-full items-center justify-center text-sm text-slate-500">
              This model's RMSE could not be computed, so no interval can be derived.
            </div>
          )}
          <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-2">
              <p className="text-slate-600">Min Forecast</p>
              <p className="font-semibold text-slate-900">
                {Math.min(...results.best_model.forecast).toFixed(2)}
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-2">
              <p className="text-slate-600">Avg Forecast</p>
              <p className="font-semibold text-slate-900">
                {(
                  results.best_model.forecast.reduce((a, b) => a + b, 0) /
                  results.best_model.forecast.length
                ).toFixed(2)}
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-2">
              <p className="text-slate-600">Max Forecast</p>
              <p className="font-semibold text-slate-900">
                {Math.max(...results.best_model.forecast).toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Chart 4: Feature Importance */}
      {activeChart === "importance" && hasFeatureImportance && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">
            Feature Importance — {results.best_model.best_model_name}
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Top {featureImportanceData.length} features ranked by contribution to{" "}
            {results.best_model.best_model_name}'s predictions. A single measure across
            a fixed set of features, so all bars share one color.
          </p>
          <div className="mt-4 w-full" style={{ height: Math.max(320, featureImportanceData.length * 40) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureImportanceData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
                barSize={20}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 12 }}
                  width={160}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#f1f5f9",
                    border: "1px solid #cbd5e1",
                    borderRadius: "8px",
                    padding: "10px",
                  }}
                  formatter={(value) => [value?.toFixed(4), "Importance"]}
                  labelFormatter={(label) => `Feature: ${label}`}
                  cursor={{ fill: "rgba(42, 120, 214, 0.08)" }}
                />
                <Bar dataKey="importance" fill="#2a78d6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </section>
  );
}
