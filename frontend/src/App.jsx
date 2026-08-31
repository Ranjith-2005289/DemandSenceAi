import { useState } from "react";
import UploadSection from "./components/UploadSection";
import StatusTracker from "./components/StatusTracker";
import ModelResults from "./components/ModelResults";
import ForecastCharts from "./components/ForecastCharts";
import ChatBot from "./components/ChatBot";
import ProductSummaryTable from "./components/ProductSummaryTable";
import InsightsPage from "./components/InsightsPage";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent } from "@/components/ui/card";
import { CheckCircle2, RefreshCw, Bot, FileText, Download, BarChart2 } from "lucide-react";

export default function App() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [uploadResult, setUploadResult] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);
  const [aggregateForecastResult, setAggregateForecastResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState("upload");
  const [isChatOpen, setIsChatOpen] = useState(false);

  const handleUploadStart = (meta) => {
    setUploadResult(meta);
  };

  const handleForecastStart = () => {
    setCurrentStep("forecasting");
    setIsLoading(true);
  };

  const handleForecastComplete = (results) => {
    setForecastResult(results);
    setAggregateForecastResult(results);
    setCurrentStep("results");
    setIsLoading(false);
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
            <h1 className="text-3xl font-bold text-white flex items-center gap-2">
              <BarChart2 className="h-8 w-8 text-blue-400" />
              DemandSense AI
            </h1>
            <p className="text-sm text-slate-300">
              Intelligent Demand Forecasting Engine
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm font-medium text-slate-200">
              {currentStep === "upload" && "Ready to upload"}
              {currentStep === "forecasting" && "Processing..."}
              {currentStep === "results" && "Complete"}
              {currentStep === "insights" && "AI Insights Report"}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Real-time ML Pipeline
            </p>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-6 py-8 flex flex-col min-h-full">
          {currentStep === "upload" && !uploadResult && (
            <div className="flex flex-col flex-1">

              {/* ── Hero ── */}
              <div className="py-10 text-center">
                <span className="inline-block rounded-full bg-blue-100 px-4 py-1.5 text-xs font-semibold uppercase tracking-widest text-blue-700 mb-4">
                  Powered by ARIMA · Prophet · XGBoost · RandomForest
                </span>
                <h2 className="text-4xl font-bold text-slate-900 tracking-tight">
                  Forecast demand with <span className="text-blue-600">machine learning</span>
                </h2>
                <p className="mt-3 text-base text-slate-500 max-w-xl mx-auto leading-relaxed">
                  Upload your dataset, configure two columns, and get a ranked multi-model forecast — no code, no setup.
                </p>
              </div>

              {/* ── Split: Upload LEFT (primary) · Context RIGHT (supporting) ── */}
              <div className="flex flex-col lg:flex-row gap-6 pb-8 items-start">

                {/* PRIMARY — Upload (left, wider) */}
                <div className="w-full lg:w-[55%] flex-shrink-0">
                  <UploadSection
                    onUploadStart={handleUploadStart}
                    onForecastStart={handleForecastStart}
                    onForecastComplete={handleForecastComplete}
                    sessionId={sessionId}
                  />
                </div>

                {/* SUPPORTING — Context (right, stacked) */}
                <div className="flex-1 min-w-0 flex flex-col gap-4">

                  {/* 1. How it works — compact, scannable before acting */}
                  <div className="rounded-2xl border border-slate-200 bg-white px-6 py-5 shadow-sm">
                    <p className="text-xs font-bold uppercase tracking-widest text-blue-600 mb-4">How it works</p>
                    <div className="space-y-4">
                      {[
                        { step: "01", title: "Upload your dataset", body: "Drop a CSV, Excel, or JSON file with a date column and a numeric target." },
                        { step: "02", title: "Map columns & horizon", body: "Pick your date, target, forecast window, and optional product grouping." },
                        { step: "03", title: "Models train in parallel", body: "ARIMA, Prophet, XGBoost & Random Forest race — each scored on RMSE, MAE, MAPE, R²." },
                        { step: "04", title: "Get results & insights", body: "Review the ranked table, charts, per-product breakdown, and AI report." },
                      ].map(({ step, title, body }) => (
                        <div key={step} className="flex gap-3">
                          <div className="flex-shrink-0 flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600 text-white text-xs font-bold">
                            {step}
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-slate-800 leading-snug">{title}</p>
                            <p className="mt-0.5 text-xs text-slate-500 leading-relaxed">{body}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 2. Formats + Use cases — 2-col quick reference */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                      <p className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-3">Supported formats</p>
                      <ul className="space-y-2.5">
                        {[
                          { fmt: "CSV", desc: "comma-separated" },
                          { fmt: "Excel", desc: ".xlsx / .xlsm" },
                          { fmt: "JSON", desc: "array of records" },
                        ].map(({ fmt, desc }) => (
                          <li key={fmt} className="flex items-center gap-2.5">
                            <span className="flex-shrink-0 rounded-md bg-slate-900 px-1.5 py-0.5 font-mono text-xs font-bold text-white">{fmt}</span>
                            <span className="text-xs text-slate-500">{desc}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                      <p className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-3">Works great for</p>
                      <ul className="space-y-2">
                        {[
                          "Retail & e-commerce",
                          "Inventory & stock",
                          "Energy consumption",
                          "Financial data",
                          "IoT sensor readings",
                        ].map((uc) => (
                          <li key={uc} className="flex items-center gap-2 text-xs text-slate-600">
                            <span className="h-1.5 w-1.5 flex-shrink-0 rounded-full bg-blue-500" />
                            {uc}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* 3. Stats — social proof, last */}
                  <div className="rounded-2xl border border-blue-100 bg-gradient-to-r from-blue-50 to-indigo-50 px-5 py-4 grid grid-cols-3 divide-x divide-blue-100">
                    {[
                      { label: "Models", value: "5+" },
                      { label: "Metrics", value: "RMSE · MAE · MAPE · R²" },
                      { label: "Holiday calendars", value: "15 countries" },
                    ].map(({ label, value }) => (
                      <div key={label} className="px-4 first:pl-0 last:pr-0 text-center">
                        <p className="text-sm font-bold text-blue-900">{value}</p>
                        <p className="text-xs text-blue-600 mt-0.5">{label}</p>
                      </div>
                    ))}
                  </div>

                </div>
              </div>
              {/* Glassy footer strip — inside landing page only */}
              <div className="mt-auto pb-6 pt-12">
                <p className="text-center text-xs text-slate-300 tracking-wide select-none">
                  DemandSense AI &copy; 2026 &nbsp;&middot;&nbsp; ARIMA &nbsp;&middot;&nbsp; Prophet &nbsp;&middot;&nbsp; XGBoost &nbsp;&middot;&nbsp; RandomForest
                </p>
              </div>
            </div>
          )}


          {/* ─ Forecasting Phase ─ */}
          {currentStep === "forecasting" && uploadResult && (
            <div className="space-y-6 animate-fade-in">
              <Card className="shadow-sm border border-slate-200">
                <CardContent className="px-7 py-6">
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                    <div>
                      <p className="text-xs font-semibold uppercase text-slate-600">File</p>
                      <p className="mt-1 truncate font-mono text-sm text-slate-900">{uploadResult.filename}</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold uppercase text-slate-600">Rows</p>
                      <p className="mt-1 text-xl font-bold text-slate-900">{uploadResult.num_rows}</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold uppercase text-slate-600">Date Column</p>
                      <p className="mt-1 text-sm text-slate-900">{uploadResult.selected_date_col}</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold uppercase text-slate-600">Horizon</p>
                      <p className="mt-1 text-xl font-bold text-blue-600">{uploadResult.forecast_horizon} days</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <StatusTracker results={forecastResult} />
            </div>
          )}

          {/* ─ Results Phase ─ */}
          {currentStep === "results" && forecastResult && (
            <div className="space-y-8 animate-fade-in">
              <Alert className="border-green-500 bg-green-50/50 shadow-sm">
                <CheckCircle2 className="h-6 w-6 text-green-600 mt-0.5" />
                <AlertTitle className="text-green-900 text-xl font-bold ml-2">Forecast Complete</AlertTitle>
                <AlertDescription className="text-green-800 ml-2 mt-2">
                  {forecastResult.group && (
                    <p className="text-sm font-medium mb-3">
                      Showing: {forecastResult.group.group_col} = "{forecastResult.group.group_value}"
                    </p>
                  )}
                  <div className="flex flex-wrap gap-6 items-center">
                    <span>
                      <strong>Best Model:</strong>{" "}
                      <span className="inline-block rounded-md bg-green-200 px-2.5 py-1 text-xs font-bold text-green-900">
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
                  </div>
                  
                  {forecastResult.group && aggregateForecastResult && (
                    <div className="mt-4">
                      <Button
                        variant="link"
                        onClick={() => setForecastResult(aggregateForecastResult)}
                        className="h-auto p-0 text-green-700 font-semibold"
                      >
                        &larr; Back to overall forecast
                      </Button>
                    </div>
                  )}

                  {forecastResult.model_run_info?.dl_skipped && (
                    <div className="mt-4 rounded-md bg-amber-100/80 border border-amber-200 px-4 py-3 text-sm text-amber-900">
                      <strong>Note:</strong> Deep learning models were skipped — this dataset has{" "}
                      {forecastResult.model_run_info.row_count} rows, below the{" "}
                      {forecastResult.model_run_info.min_rows_for_dl}-row minimum recommended
                      for reliable deep learning forecasts.
                    </div>
                  )}
                </AlertDescription>
              </Alert>

              <div>
                <h3 className="mb-4 text-xl font-semibold text-slate-900">Model Performance Ranking</h3>
                <ModelResults all_models={forecastResult.all_models} best_model={forecastResult.best_model} />
              </div>

              <div>
                <h3 className="mb-4 text-xl font-semibold text-slate-900">Detailed Analysis</h3>
                <ForecastCharts results={forecastResult} />
              </div>

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

              <div className="flex flex-wrap justify-center gap-4 pb-8 pt-4">
                <Button variant="outline" size="lg" onClick={handleReset} className="gap-2 border-slate-300">
                  <RefreshCw className="h-4 w-4" />
                  Upload New File
                </Button>
                <Button 
                  variant="outline" 
                  size="lg" 
                  onClick={() => setIsChatOpen(true)} 
                  className="gap-2 bg-purple-50 text-purple-900 border-purple-300 hover:bg-purple-100 hover:text-purple-950"
                >
                  <Bot className="h-4 w-4" />
                  Ask AI Assistant
                </Button>
                <Button 
                  variant="outline" 
                  size="lg" 
                  onClick={() => setCurrentStep("insights")} 
                  className="gap-2 bg-indigo-50 text-indigo-900 border-indigo-300 hover:bg-indigo-100 hover:text-indigo-950"
                >
                  <FileText className="h-4 w-4" />
                  AI Insights Report
                </Button>
                <Button 
                  variant="outline" 
                  size="lg" 
                  onClick={() => {
                    const element = document.createElement("a");
                    element.setAttribute("href", "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(forecastResult, null, 2)));
                    element.setAttribute("download", "forecast_results.json");
                    element.style.display = "none";
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                  }} 
                  className="gap-2 bg-blue-50 text-blue-900 border-blue-300 hover:bg-blue-100 hover:text-blue-950"
                >
                  <Download className="h-4 w-4" />
                  Export Results
                </Button>
              </div>
            </div>
          )}

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
        </div>
      </main>



      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.3s ease-in-out;
        }
      `}</style>

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
