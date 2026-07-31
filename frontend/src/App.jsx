import { useState } from "react";
import UploadSection from "./components/UploadSection";
import StatusTracker from "./components/StatusTracker";
import ModelResults from "./components/ModelResults";
import ForecastCharts from "./components/ForecastCharts";
import ChatBot from "./components/ChatBot";
import ProductSummaryTable from "./components/ProductSummaryTable";
import InsightsPage from "./components/InsightsPage";

export default function App() {
  // One session id per app load, so concurrent tabs/users don't collide on
  // the backend's in-memory forecast cache (mirrors the chatbot's existing
  // per-session pattern).
  const [sessionId] = useState(() => crypto.randomUUID());
  const [uploadResult, setUploadResult] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);
  const [aggregateForecastResult, setAggregateForecastResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState("upload"); // upload, forecasting, results
  const [isChatOpen, setIsChatOpen] = useState(false);

  const handleUploadStart = (meta) => {
    console.log("\n🎯 App.jsx: handleUploadStart callback");
    console.log("  📋 Metadata received:", meta);
    setUploadResult(meta);
    console.log("  ✓ uploadResult state updated");
  };

  const handleForecastStart = () => {
    console.log("\n🎯 App.jsx: handleForecastStart callback");
    console.log("  ⚙ Setting currentStep to 'forecasting'");
    setCurrentStep("forecasting");
    console.log("  ⚙ Setting isLoading to true");
    setIsLoading(true);
    console.log("  ✓ State updated, StatusTracker should now display");
  };

  const handleForecastComplete = (results) => {
    console.log("\n🎯 App.jsx: handleForecastComplete callback");
    console.log("  📊 Results received:", results);
    console.log("  📊 Result keys:", Object.keys(results || {}));
    console.log("  📊 Best model:", results?.best_model?.model_name);
    setForecastResult(results);
    setAggregateForecastResult(results);
    console.log("  ✓ forecastResult state updated");
    console.log("  ⚙ Setting currentStep to 'results'");
    setCurrentStep("results");
    console.log("  ⚙ Setting isLoading to false");
    setIsLoading(false);
    console.log("  ✓ State fully updated, results should display");
  };

  const handleGroupForecast = (results) => {
    setForecastResult(results);
  };

  const handleReset = () => {
    setUploadResult(null);
    setForecastResult(null);
    setAggregateForecastResult(null);
    setIsLoading(false);
    setCurrentStep("upload");
  };

  return (
    <div className="flex h-screen flex-col bg-slate-50">
      {/* ─── Navbar ─────────────────────────────────────────────────────────── */}
      <nav className="border-b border-slate-200 bg-gradient-to-r from-slate-900 to-slate-800 shadow-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="space-y-1">
            <h1 className="text-3xl font-bold text-white">
              📊 DemandSense AI
            </h1>
            <p className="text-sm text-slate-300">
              Intelligent Demand Forecasting Engine
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm font-medium text-slate-200">
              {currentStep === "upload" && "Ready to upload"}
              {currentStep === "forecasting" && "🔄 Processing..."}
              {currentStep === "results" && "✅ Complete"}
              {currentStep === "insights" && "📋 AI Insights Report"}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Real-time ML Pipeline
            </p>
          </div>
        </div>
      </nav>

      {/* ─── Main Content ────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-6 py-8">
          {/* ─ Upload Phase ─ */}
          {currentStep === "upload" && !uploadResult && (
            <div className="space-y-6">
              <div className="rounded-xl border-2 border-dashed border-blue-300 bg-blue-50 p-8 text-center">
                <h2 className="text-2xl font-semibold text-blue-900">
                  Step 1: Upload Your Data
                </h2>
                <p className="mt-2 text-blue-700">
                  CSV, Excel, or JSON files with time-series data (any retail, stock, energy, or IoT dataset)
                </p>
              </div>
              <UploadSection
                onUploadStart={handleUploadStart}
                onForecastStart={handleForecastStart}
                onForecastComplete={handleForecastComplete}
                sessionId={sessionId}
              />
            </div>
          )}

          {/* ─ Forecasting Phase ─ */}
          {currentStep === "forecasting" && uploadResult && (
            <div className="space-y-6 animate-fade-in">
              {/* Upload Summary */}
              <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                  <div>
                    <p className="text-xs font-semibold uppercase text-slate-600">
                      File
                    </p>
                    <p className="mt-1 truncate font-mono text-sm text-slate-900">
                      {uploadResult.filename}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase text-slate-600">
                      Rows
                    </p>
                    <p className="mt-1 text-xl font-bold text-slate-900">
                      {uploadResult.num_rows}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase text-slate-600">
                      Date Column
                    </p>
                    <p className="mt-1 text-sm text-slate-900">
                      {uploadResult.selected_date_col}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase text-slate-600">
                      Horizon
                    </p>
                    <p className="mt-1 text-xl font-bold text-blue-600">
                      {uploadResult.forecast_horizon} days
                    </p>
                  </div>
                </div>
              </div>

              {/* Status Tracker */}
              <StatusTracker results={forecastResult} />
            </div>
          )}

          {/* ─ Results Phase ─ */}
          {currentStep === "results" && forecastResult && (
            <div className="space-y-8 animate-fade-in">
              {/* Success Banner */}
              <div className="rounded-xl border-l-4 border-l-green-500 bg-gradient-to-r from-green-50 to-emerald-50 p-6 shadow-sm">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="flex items-center text-2xl font-bold text-green-900">
                      ✅ Forecast Complete
                    </h2>
                    {forecastResult.group && (
                      <p className="mt-1 text-sm font-medium text-green-800">
                        Showing: {forecastResult.group.group_col} = "{forecastResult.group.group_value}"
                      </p>
                    )}
                    <p className="mt-3 space-x-6 text-sm text-green-800">
                      <span>
                        <strong>Best Model:</strong>{" "}
                        <span className="inline-block rounded bg-green-200 px-2 py-1 font-semibold">
                          {forecastResult.best_model?.best_model_name}
                        </span>
                      </span>
                      <span>
                        <strong>RMSE:</strong>{" "}
                        <span className="font-mono font-bold text-green-700">
                          {forecastResult.best_model?.rmse?.toFixed(4)}
                        </span>
                      </span>
                      <span>
                        <strong>Horizon:</strong> {uploadResult?.forecast_horizon} days
                      </span>
                    </p>
                    {forecastResult.group && aggregateForecastResult && (
                      <button
                        onClick={() => setForecastResult(aggregateForecastResult)}
                        className="mt-3 text-sm font-medium text-green-900 underline hover:text-green-700"
                      >
                        ← Back to overall forecast
                      </button>
                    )}
                    {forecastResult.model_run_info?.dl_skipped && (
                      <p className="mt-3 rounded-md bg-amber-100 px-3 py-2 text-xs text-amber-900">
                        ℹ️ Deep learning models were skipped — this dataset has{" "}
                        {forecastResult.model_run_info.row_count} rows, below the{" "}
                        {forecastResult.model_run_info.min_rows_for_dl}-row minimum recommended
                        for reliable deep learning forecasts.
                      </p>
                    )}
                  </div>
                  <div className="text-4xl">🎯</div>
                </div>
              </div>

              {/* Model Results Table */}
              <div>
                <h3 className="mb-4 text-xl font-semibold text-slate-900">
                  Model Performance Ranking
                </h3>
                <ModelResults
                  all_models={forecastResult.all_models}
                  best_model={forecastResult.best_model}
                />
              </div>

              {/* Forecast Charts */}
              <div>
                <h3 className="mb-4 text-xl font-semibold text-slate-900">
                  Detailed Analysis
                </h3>
                <ForecastCharts results={forecastResult} />
              </div>

              {/* Per-group sales ranking (only if a grouping column was selected at upload) */}
              {uploadResult?.group_col && (
                <ProductSummaryTable
                  filename={uploadResult.saved_as}
                  date_col={uploadResult.selected_date_col}
                  target_col={uploadResult.selected_target_col}
                  group_col={uploadResult.group_col}
                  forecast_horizon={uploadResult.forecast_horizon}
                  country={uploadResult.country}
                  sessionId={sessionId}
                  onForecastGroup={handleGroupForecast}
                />
              )}

              {/* Action Buttons */}
              <div className="flex justify-center gap-4 pb-8 pt-4">
                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 rounded-lg border-2 border-slate-300 bg-white px-6 py-3 font-semibold text-slate-900 transition-all hover:border-slate-400 hover:bg-slate-50 active:scale-95"
                >
                  <span>↻</span>
                  Upload New File
                </button>
                <button
                  onClick={() => setIsChatOpen(true)}
                  className="flex items-center gap-2 rounded-lg border-2 border-purple-300 bg-purple-50 px-6 py-3 font-semibold text-purple-900 transition-all hover:border-purple-400 hover:bg-purple-100 active:scale-95"
                >
                  <span>🤖</span>
                  Ask AI Assistant
                </button>
                <button
                  onClick={() => setCurrentStep("insights")}
                  className="flex items-center gap-2 rounded-lg border-2 border-indigo-300 bg-indigo-50 px-6 py-3 font-semibold text-indigo-900 transition-all hover:border-indigo-400 hover:bg-indigo-100 active:scale-95"
                >
                  <span>📋</span>
                  AI Insights Report
                </button>
                <button
                  onClick={() => {
                    const element = document.createElement("a");
                    element.setAttribute(
                      "href",
                      "data:text/json;charset=utf-8," +
                      encodeURIComponent(JSON.stringify(forecastResult, null, 2))
                    );
                    element.setAttribute("download", "forecast_results.json");
                    element.style.display = "none";
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                  }}
                  className="flex items-center gap-2 rounded-lg border-2 border-blue-300 bg-blue-50 px-6 py-3 font-semibold text-blue-900 transition-all hover:border-blue-400 hover:bg-blue-100 active:scale-95"
                >
                  <span>📥</span>
                  Export Results
                </button>
              </div>
            </div>
          )}

          {/* ─ AI Insights Report Page ─ */}
          {currentStep === "insights" && uploadResult && aggregateForecastResult && (
            <InsightsPage
              filename={uploadResult.saved_as}
              date_col={uploadResult.selected_date_col}
              target_col={uploadResult.selected_target_col}
              group_col={uploadResult.group_col}
              forecast_result={aggregateForecastResult}
              onBack={() => setCurrentStep("results")}
            />
          )}

          {/* ─ Empty State ─ */}
          {currentStep === "upload" && !uploadResult && !isLoading && (
            <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-12 text-center">
              <p className="text-slate-600">
                Start by uploading a CSV file to begin forecasting
              </p>
            </div>
          )}
        </div>
      </main>

      {/* ─── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-200 bg-slate-900 py-4 text-center text-sm text-slate-400">
        <p>DemandSense AI © 2026 | Powered by ARIMA, Prophet, XGBoost, RandomForest And More.. </p>
      </footer>

      {/* ─── Fade Animation ─────────────────────────────────────────────────── */}
      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fadeIn 0.3s ease-in-out;
        }
      `}</style>

      {/* ─── ChatBot Component ─────────────────────────────────────────────── */}
      <ChatBot
        forecastResult={forecastResult}
        isVisible={isChatOpen && currentStep === "results"}
        onClose={() => setIsChatOpen(false)}
        groupInfo={
          uploadResult?.group_col
            ? {
                filename: uploadResult.saved_as,
                date_col: uploadResult.selected_date_col,
                target_col: uploadResult.selected_target_col,
                group_col: uploadResult.group_col,
              }
            : null
        }
      />
    </div>
  );
}
