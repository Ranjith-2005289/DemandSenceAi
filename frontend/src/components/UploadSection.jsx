import { useState } from "react";
import { uploadCsv, runForecast } from "../api";

const SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xlsm", ".json"];

// Curated list of common countries the `holidays` package supports —
// friendlier than asking users to know ISO codes themselves.
const COUNTRY_OPTIONS = [
  { code: "US", label: "United States" },
  { code: "GB", label: "United Kingdom" },
  { code: "IN", label: "India" },
  { code: "CA", label: "Canada" },
  { code: "AU", label: "Australia" },
  { code: "DE", label: "Germany" },
  { code: "FR", label: "France" },
  { code: "BR", label: "Brazil" },
  { code: "MX", label: "Mexico" },
  { code: "JP", label: "Japan" },
  { code: "CN", label: "China" },
  { code: "SG", label: "Singapore" },
  { code: "AE", label: "United Arab Emirates" },
  { code: "ZA", label: "South Africa" },
  { code: "NZ", label: "New Zealand" },
];

export default function UploadSection({ onUploadStart, onForecastStart, onForecastComplete, onStatus, sessionId }) {
  const [file, setFile] = useState(null);
  const [uploadData, setUploadData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState("upload"); // "upload", "config", "running"
  const [error, setError] = useState(null); // Track errors for display

  // Column selection state
  const [selectedDateCol, setSelectedDateCol] = useState(null);
  const [selectedTargetCol, setSelectedTargetCol] = useState(null);
  const [selectedGroupCol, setSelectedGroupCol] = useState("");
  const [selectedCountry, setSelectedCountry] = useState("");
  const [forecastHorizon, setForecastHorizon] = useState(30);

  // Drag-and-drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      const f = droppedFiles[0];
      if (SUPPORTED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))) {
        setFile(f);
      } else {
        onStatus?.(`Please drop a supported file (${SUPPORTED_EXTENSIONS.join(", ")}).`);
      }
    }
  };

  const handleFileInput = (e) => {
    const f = e.target.files?.[0] || null;
    if (f) {
      setFile(f);
    }
  };

  // Upload CSV
  const handleUpload = async () => {
    console.log("\n═══════════════════════════════════════════════════════════════");
    console.log("📤 UPLOAD PIPELINE STARTED");
    console.log("═══════════════════════════════════════════════════════════════");
    
    // Step 1: Validation
    console.log("\n📋 Step 1: Validating file selection");
    if (!file) {
      console.error("❌ Validation failed: No file selected");
      setError("Select a CSV file first.");
      onStatus?.("Select a CSV file first.");
      return;
    }
    console.log("  ✓ File selected:", file.name, `(${file.size} bytes)`);

    try {
      // Step 2: Setup
      console.log("\n📋 Step 2: Setting up upload state");
      setLoading(true);
      setError(null);
      // Clear any column selections left over from a previously uploaded
      // file — otherwise a stale "date" (from an earlier file that had one)
      // can silently survive into a new file that has no date column at
      // all, and get submitted anyway since the Run Forecast button only
      // checks truthiness, not whether the value is valid for this file.
      setSelectedDateCol(null);
      setSelectedTargetCol(null);
      setSelectedGroupCol("");
      setSelectedCountry("");
      onStatus?.("Uploading and analyzing...");
      console.log("  ✓ Loading state set to true");
      console.log("  ✓ Error state cleared");

      // Step 3: Prepare data
      console.log("\n📋 Step 3: Preparing FormData");
      const formData = new FormData();
      formData.append("file", file);
      console.log("  ✓ FormData created with file");

      // Step 4: Upload
      console.log("\n📋 Step 4: Calling uploadCsv API");
      const result = await uploadCsv(formData);

      // Step 5: Process response
      console.log("\n📋 Step 5: Processing upload response");
      console.log("  ✅ Upload successful");
      console.log("  📊 Response status:", result?.status);
      console.log("  📊 File shape:", result?.shape);
      console.log("  📊 Detected columns:", {
        dates: result?.detected?.date_columns,
        targets: result?.detected?.numeric_columns,
      });
      console.log("  📊 Suggestions:", result?.suggestions);
      console.log("  📊 Full result:", result);
      setUploadData(result);
      console.log("  ✓ uploadData state updated");

      // Step 6: Auto-select columns
      console.log("\n📋 Step 6: Auto-selecting suggested columns");
      if (result?.suggestions?.date_col) {
        console.log("  ✓ Setting date column to:", result.suggestions.date_col);
        setSelectedDateCol(result.suggestions.date_col);
      }
      if (result?.suggestions?.target_col) {
        console.log("  ✓ Setting target column to:", result.suggestions.target_col);
        setSelectedTargetCol(result.suggestions.target_col);
      }

      // Step 7: Update UI
      console.log("\n📋 Step 7: Updating UI state");
      setStep("config");
      console.log("  ✓ Step set to 'config'");
      onStatus?.("File uploaded. Please configure columns and forecast horizon.");
      console.log("  ✓ Status message displayed");

      console.log("\n✅ UPLOAD PIPELINE COMPLETED SUCCESSFULLY");
      console.log("═══════════════════════════════════════════════════════════════\n");
    } catch (error) {
      console.error("\n❌ UPLOAD PIPELINE FAILED");
      console.error("═══════════════════════════════════════════════════════════════");
      console.error("  Error type:", error?.constructor?.name);
      console.error("  Error message:", error?.message);
      console.error("  Response status:", error?.response?.status);
      console.error("  Response data:", error?.response?.data);
      console.error("  Full error:", error);
      console.error("═══════════════════════════════════════════════════════════════\n");

      const errorMsg = error?.response?.data?.detail || error?.message || "Upload failed.";
      setError(errorMsg);
      onStatus?.(errorMsg);
      setStep("upload");
    } finally {
      setLoading(false);
      console.log("📋 Finally block: Loading set to false");
    }
  };

  // Run forecast
  const handleRunForecast = async () => {
    console.log("═══════════════════════════════════════════════════════════════");
    console.log("🚀 FORECAST PIPELINE STARTED");
    console.log("═══════════════════════════════════════════════════════════════");
    
    // Step 1: Validation
    console.log("📋 Step 1: Validating inputs");
    console.log("  ✓ selectedDateCol:", selectedDateCol);
    console.log("  ✓ selectedTargetCol:", selectedTargetCol);
    console.log("  ✓ forecastHorizon:", forecastHorizon);
    
    if (!selectedDateCol || !selectedTargetCol) {
      console.error("❌ Validation failed: Missing columns");
      onStatus?.("Please select both date and target columns.");
      return;
    }

    console.log("  ✓ Columns selected OK");
    console.log("  ✓ uploadData:", uploadData);
    console.log("  ✓ uploadData.file:", uploadData?.file);
    console.log("  ✓ uploadData.file.saved_as:", uploadData?.file?.saved_as);

    if (!uploadData?.file?.saved_as) {
      console.error("❌ Validation failed: No saved file");
      onStatus?.("Upload data first.");
      return;
    }

    console.log("✅ All validations passed");

    try {
      // Step 2: State setup
      console.log("\n📋 Step 2: Setting up state");
      setLoading(true);
      console.log("  ✓ Loading set to true");
      
      // Step 3: Notify callbacks
      console.log("\n📋 Step 3: Notifying parent callbacks");
      onForecastStart?.();
      console.log("  ✓ onForecastStart called");
      
      const uploadMetadata = {
        filename: file.name,
        num_rows: uploadData.shape?.rows,
        num_cols: uploadData.shape?.cols,
        selected_date_col: selectedDateCol,
        selected_target_col: selectedTargetCol,
        forecast_horizon: forecastHorizon,
        saved_as: uploadData.file.saved_as,
        group_col: selectedGroupCol || null,
        country: selectedCountry || null,
      };
      console.log("  📤 Upload metadata:", uploadMetadata);
      onUploadStart?.(uploadMetadata);
      console.log("  ✓ onUploadStart called");

      // Step 4: Update UI
      console.log("\n📋 Step 4: Updating UI state");
      setStep("running");
      console.log("  ✓ Step set to 'running'");
      onStatus?.("Training models and generating forecast...");
      console.log("  ✓ Status updated");

      // Step 5: Prepare API call
      console.log("\n📋 Step 5: Preparing forecast API call");
      const forecastPayload = {
        filename: uploadData.file.saved_as,
        date_col: selectedDateCol,
        target_col: selectedTargetCol,
        forecast_horizon: forecastHorizon,
        country: selectedCountry || null,
      };
      console.log("  📤 Forecast request payload:", forecastPayload);

      // Step 6: Call API
      console.log("\n📋 Step 6: Calling forecast API at POST /forecast");
      console.log("  🌐 Sending request...");
      const result = await runForecast(
        uploadData.file.saved_as,
        selectedDateCol,
        selectedTargetCol,
        forecastHorizon,
        null,
        null,
        selectedCountry || null,
        sessionId || null
      );

      // Step 7: Handle result
      console.log("\n📋 Step 7: Processing API response");
      console.log("  ✅ API returned successfully");
      console.log("  📊 Result status:", result?.status);
      console.log("  📊 Result keys:", Object.keys(result || {}));
      console.log("  📊 Full result:", result);

      // Step 8: Propagate to parent
      console.log("\n📋 Step 8: Propagating results to parent component");
      onForecastComplete?.(result);
      console.log("  ✓ onForecastComplete called");
      onStatus?.("Forecast completed!");
      console.log("  ✓ Status updated to 'Forecast completed!'");

      console.log("\n✅ FORECAST PIPELINE COMPLETED SUCCESSFULLY");
      console.log("═══════════════════════════════════════════════════════════════\n");
    } catch (error) {
      console.error("\n❌ FORECAST PIPELINE FAILED");
      console.error("═══════════════════════════════════════════════════════════════");
      console.error("  Error type:", error?.constructor?.name);
      console.error("  Error message:", error?.message);
      console.error("  Response status:", error?.response?.status);
      console.error("  Response data:", error?.response?.data);
      console.error("  Full error:", error);
      console.error("═══════════════════════════════════════════════════════════════\n");

      const errorMsg = error?.response?.data?.detail || error?.message || "Forecast failed.";
      console.error("💬 Setting error message:", errorMsg);
      setError(errorMsg);
      onStatus?.(errorMsg);
      setStep("config");
    } finally {
      setLoading(false);
      console.log("📋 Finally block: Loading set to false");
    }
  };

  const fileSize = file ? (file.size / 1024).toFixed(2) : 0;
  const dateColumns = uploadData?.detected?.date_columns || [];
  const numericColumns = uploadData?.detected?.numeric_columns || [];
  const groupingColumns = uploadData?.detected?.grouping_columns || [];

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">1) Upload & Configure</h2>

      {/* Error Alert */}
      {error && (
        <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-4">
          <p className="text-sm font-semibold text-red-900">❌ Error</p>
          <p className="mt-1 text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Step 1: Upload */}
      {step === "upload" && (
        <div className="mt-4 space-y-3">
          {/* Drag and drop zone */}
          <input
            type="file"
            accept={SUPPORTED_EXTENSIONS.join(",")}
            onChange={handleFileInput}
            className="hidden"
            id="csv-input"
          />
          <label
            htmlFor="csv-input"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 p-8 text-center transition hover:border-slate-400 hover:bg-slate-100 cursor-pointer block"
          >
            <div className="flex flex-col items-center gap-2">
              <svg
                className="h-10 w-10 text-slate-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-sm font-medium text-slate-900">
                Drag and drop your file here
              </p>
              <p className="text-xs text-slate-600">CSV, Excel (.xlsx), or JSON — or click to select</p>
            </div>
            <div className="mt-3 inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700">
              Select File
            </div>
          </label>

          {/* File preview */}
          {file && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs font-semibold text-slate-600 uppercase">Selected File</p>
              <div className="mt-2 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-900">{file.name}</p>
                  <p className="text-xs text-slate-600">{fileSize} KB</p>
                </div>
                <button
                  onClick={() => setFile(null)}
                  className="text-xs text-slate-500 hover:text-slate-700"
                >
                  Change
                </button>
              </div>
            </div>
          )}

          {/* Upload button */}
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Uploading...
              </span>
            ) : (
              "Upload CSV"
            )}
          </button>
        </div>
      )}

      {/* Step 2: Configuration */}
      {step === "config" && uploadData && (
        <div className="mt-4 space-y-4">
          {/* File info */}
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-semibold text-slate-600 uppercase">File Info</p>
            <div className="mt-2 grid grid-cols-3 gap-3 text-xs">
              <div>
                <p className="text-slate-600">Rows</p>
                <p className="font-semibold text-slate-900">{uploadData.shape?.rows}</p>
              </div>
              <div>
                <p className="text-slate-600">Columns</p>
                <p className="font-semibold text-slate-900">{uploadData.shape?.cols}</p>
              </div>
              <div>
                <p className="text-slate-600">Frequency</p>
                <p className="font-semibold text-slate-900">{uploadData.suggestions?.frequency}</p>
              </div>
            </div>
          </div>

          {/* Data quality warnings/errors */}
          {uploadData.validation?.warnings?.length > 0 && (
            <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3">
              <p className="text-xs font-semibold text-yellow-900">⚠ Warnings</p>
              {uploadData.validation.warnings.map((w, i) => (
                <p key={i} className="mt-1 text-xs text-yellow-800">
                  • {w}
                </p>
              ))}
            </div>
          )}

          {/* Column selectors */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-semibold uppercase text-slate-600">
                Date Column
              </label>
              <select
                value={selectedDateCol || ""}
                onChange={(e) => setSelectedDateCol(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 transition hover:border-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-20"
              >
                <option value="">— Select date column —</option>
                {dateColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase text-slate-600">
                Target Column
              </label>
              <select
                value={selectedTargetCol || ""}
                onChange={(e) => setSelectedTargetCol(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 transition hover:border-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-20"
              >
                <option value="">— Select target column —</option>
                {numericColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Group-by selector (optional, only shown if grouping columns were detected) */}
          {groupingColumns.length > 0 && (
            <div>
              <label className="block text-xs font-semibold uppercase text-slate-600">
                Group By (optional)
              </label>
              <select
                value={selectedGroupCol}
                onChange={(e) => setSelectedGroupCol(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 transition hover:border-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-20"
              >
                <option value="">None — aggregate whole dataset</option>
                {groupingColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-slate-600">
                Adds a per-{selectedGroupCol || "group"} sales ranking below the main forecast, with
                the option to forecast an individual one.
              </p>
            </div>
          )}

          {/* Country selector (optional, adds public-holiday effects to Prophet) */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-600">
              Country (optional — for holiday effects)
            </label>
            <select
              value={selectedCountry}
              onChange={(e) => setSelectedCountry(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 transition hover:border-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-20"
            >
              <option value="">None — no holiday effects</option>
              {COUNTRY_OPTIONS.map(({ code, label }) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-slate-600">
              Lets the Prophet model account for that country's public holidays.
            </p>
          </div>

          {/* Forecast horizon */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-600">
              Forecast Horizon (days)
            </label>
            <input
              type="number"
              min="1"
              max="365"
              value={forecastHorizon}
              onChange={(e) => setForecastHorizon(Math.max(1, parseInt(e.target.value) || 30))}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 transition hover:border-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-20"
            />
            <p className="mt-1 text-xs text-slate-600">
              Forecast up to {forecastHorizon} periods into the future
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex gap-3">
            <button
              onClick={() => {
                setFile(null);
                setUploadData(null);
                setStep("upload");
              }}
              className="flex-1 rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-900 transition hover:bg-slate-50"
            >
              Back
            </button>
            <button
              onClick={handleRunForecast}
              disabled={!selectedDateCol || !selectedTargetCol || loading}
              className="flex-1 rounded-md bg-green-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-green-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Running...
                </span>
              ) : (
                "Run Forecast"
              )}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Running */}
      {step === "running" && (
        <div className="mt-4 flex flex-col items-center gap-3 py-8">
          <svg className="h-12 w-12 animate-spin text-blue-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <p className="text-sm font-medium text-slate-900">Training models...</p>
          <p className="text-xs text-slate-600">This may take a few minutes</p>
        </div>
      )}

      {/* Debug Info */}
      <div className="mt-4 rounded-lg border border-slate-300 bg-slate-100 p-3 text-xs font-mono text-slate-700">
        <p>Step: {step} | File: {file?.name || "none"} | UploadData: {uploadData ? "✓" : "✗"}</p>
      </div>
    </section>
  );
}
