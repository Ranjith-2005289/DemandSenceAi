import { useState } from "react";
import { uploadCsv, runForecast } from "../api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { UploadCloud, AlertCircle, Loader2 } from "lucide-react";

const SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xlsm", ".json"];

// Curated list of common countries the `holidays` package supports
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
  const [error, setError] = useState(null);

  // Column selection state
  const [selectedDateCol, setSelectedDateCol] = useState("");
  const [selectedTargetCol, setSelectedTargetCol] = useState("");
  const [selectedGroupCol, setSelectedGroupCol] = useState("none");
  const [selectedCountry, setSelectedCountry] = useState("none");
  const [forecastHorizon, setForecastHorizon] = useState(30);

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

  const handleUpload = async () => {
    if (!file) {
      setError("Select a CSV file first.");
      onStatus?.("Select a CSV file first.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setSelectedDateCol("");
      setSelectedTargetCol("");
      setSelectedGroupCol("none");
      setSelectedCountry("none");
      onStatus?.("Uploading and analyzing...");

      const formData = new FormData();
      formData.append("file", file);

      const result = await uploadCsv(formData);
      setUploadData(result);

      if (result?.suggestions?.date_col) {
        setSelectedDateCol(result.suggestions.date_col);
      }
      if (result?.suggestions?.target_col) {
        setSelectedTargetCol(result.suggestions.target_col);
      }

      setStep("config");
      onStatus?.("File uploaded. Please configure columns and forecast horizon.");
    } catch (error) {
      const errorMsg = error?.response?.data?.detail || error?.message || "Upload failed.";
      setError(errorMsg);
      onStatus?.(errorMsg);
      setStep("upload");
    } finally {
      setLoading(false);
    }
  };

  const handleRunForecast = async () => {
    if (!selectedDateCol || !selectedTargetCol) {
      onStatus?.("Please select both date and target columns.");
      return;
    }

    if (!uploadData?.file?.saved_as) {
      onStatus?.("Upload data first.");
      return;
    }

    try {
      setLoading(true);
      onForecastStart?.();
      
      const groupColActual = selectedGroupCol === "none" ? null : selectedGroupCol;
      const countryActual = selectedCountry === "none" ? null : selectedCountry;
      
      const uploadMetadata = {
        filename: file.name,
        num_rows: uploadData.shape?.rows,
        num_cols: uploadData.shape?.cols,
        selected_date_col: selectedDateCol,
        selected_target_col: selectedTargetCol,
        forecast_horizon: forecastHorizon,
        saved_as: uploadData.file.saved_as,
        group_col: groupColActual,
        country: countryActual,
      };
      
      onUploadStart?.(uploadMetadata);

      setStep("running");
      onStatus?.("Training models and generating forecast...");

      const result = await runForecast(
        uploadData.file.saved_as,
        selectedDateCol,
        selectedTargetCol,
        forecastHorizon,
        null,
        null,
        countryActual,
        sessionId || null
      );

      onForecastComplete?.(result);
      onStatus?.("Forecast completed!");
    } catch (error) {
      const errorMsg = error?.response?.data?.detail || error?.message || "Forecast failed.";
      setError(errorMsg);
      onStatus?.(errorMsg);
      setStep("config");
    } finally {
      setLoading(false);
    }
  };

  const fileSize = file ? (file.size / 1024).toFixed(2) : 0;
  const dateColumns = uploadData?.detected?.date_columns || [];
  const numericColumns = uploadData?.detected?.numeric_columns || [];
  const groupingColumns = uploadData?.detected?.grouping_columns || [];

  return (
    <Card className="shadow-sm border border-slate-200">
      <CardContent className="pt-0 px-0 pb-0">
        {/* Header row */}
        <div className="px-7 pt-6 pb-5 border-b border-slate-100">
          <p className="text-xs font-bold uppercase tracking-widest text-slate-400">Upload & Configure</p>
          <p className="mt-1 text-sm text-slate-500">Select your file, map columns, and start the forecast.</p>
        </div>

        <div className="px-7 py-6">
        {error && (
          <Alert variant="destructive" className="mb-5">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {step === "upload" && (
          <div className="space-y-4">
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
              className="group relative rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 p-10 text-center transition-all hover:border-blue-400 hover:bg-blue-50 cursor-pointer block"
            >
              <div className="flex flex-col items-center gap-3">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white shadow-sm border border-slate-200 group-hover:border-blue-200 transition-all">
                  <UploadCloud className="h-7 w-7 text-slate-400 group-hover:text-blue-500 transition-colors" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Drag & drop your file here
                  </p>
                  <p className="mt-0.5 text-xs text-slate-400">CSV, Excel (.xlsx, .xlsm), or JSON</p>
                </div>
              </div>
              <div className="mt-5">
                <span className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 shadow-sm group-hover:border-blue-200 group-hover:text-blue-700 transition-all">
                  Browse files
                </span>
              </div>
            </label>

            {file && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-900">{file.name}</p>
                  <p className="text-xs text-slate-600">{fileSize} KB</p>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setFile(null)}>Change</Button>
              </div>
            )}

            <Button
              onClick={handleUpload}
              disabled={!file || loading}
              className="w-full h-11 font-semibold bg-blue-600 hover:bg-blue-700 text-white shadow-sm"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Uploading & analysing...
                </>
              ) : (
                "Analyse & Continue"
              )}
            </Button>
          </div>
        )}

        {step === "config" && uploadData && (
          <div className="space-y-6">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold text-slate-600 uppercase">File Info</p>
              <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="text-slate-500 text-xs">Rows</p>
                  <p className="font-semibold text-slate-900">{uploadData.shape?.rows}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-xs">Columns</p>
                  <p className="font-semibold text-slate-900">{uploadData.shape?.cols}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-xs">Frequency</p>
                  <p className="font-semibold text-slate-900">{uploadData.suggestions?.frequency}</p>
                </div>
              </div>
            </div>

            {uploadData.validation?.warnings?.length > 0 && (
              <Alert className="bg-yellow-50 border-yellow-200 text-yellow-900">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <AlertTitle className="text-yellow-900">Warnings</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc pl-5 mt-2 space-y-1 text-xs">
                    {uploadData.validation.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            <div className="grid gap-6 sm:grid-cols-3">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase text-slate-600">
                  Date Column
                </label>
                <Select value={selectedDateCol} onValueChange={setSelectedDateCol}>
                  <SelectTrigger>
                    <SelectValue placeholder="— Select date —" />
                  </SelectTrigger>
                  <SelectContent>
                    {dateColumns.map((col) => (
                      <SelectItem key={col} value={col}>{col}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase text-slate-600">
                  Target Column
                </label>
                <Select value={selectedTargetCol} onValueChange={setSelectedTargetCol}>
                  <SelectTrigger>
                    <SelectValue placeholder="— Select target —" />
                  </SelectTrigger>
                  <SelectContent>
                    {numericColumns.map((col) => (
                      <SelectItem key={col} value={col}>{col}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase text-slate-600">
                  Forecast Horizon
                </label>
                <Input
                  type="number"
                  min="1"
                  max="365"
                  value={forecastHorizon}
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val === '') setForecastHorizon('');
                    else setForecastHorizon(Math.max(1, parseInt(val) || 1));
                  }}
                  className="[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                  placeholder="Days (e.g. 30)"
                />
              </div>
            </div>

            {groupingColumns.length > 0 && (
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase text-slate-600">
                  Group By (optional)
                </label>
                <Select value={selectedGroupCol} onValueChange={setSelectedGroupCol}>
                  <SelectTrigger>
                    <SelectValue placeholder="None — aggregate whole dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None — aggregate whole dataset</SelectItem>
                    {groupingColumns.map((col) => (
                      <SelectItem key={col} value={col}>{col}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-500">
                  Adds a per-group sales ranking below the main forecast, with the option to forecast an individual one.
                </p>
              </div>
            )}

            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase text-slate-600">
                Country (optional — for holiday effects)
              </label>
              <Select value={selectedCountry} onValueChange={setSelectedCountry}>
                <SelectTrigger>
                  <SelectValue placeholder="None — no holiday effects" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None — no holiday effects</SelectItem>
                  {COUNTRY_OPTIONS.map(({ code, label }) => (
                    <SelectItem key={code} value={code}>{label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-slate-500">
                Lets the Prophet model account for that country's public holidays.
              </p>
            </div>


            <div className="flex gap-4 pt-2">
              <Button
                variant="outline"
                onClick={() => {
                  setFile(null);
                  setUploadData(null);
                  setStep("upload");
                }}
                className="flex-1"
              >
                Back
              </Button>
              <Button
                onClick={handleRunForecast}
                disabled={!selectedDateCol || !selectedTargetCol || loading}
                className="flex-1 bg-green-600 hover:bg-green-700 text-white"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  "Run Forecast"
                )}
              </Button>
            </div>
          </div>
        )}

        {step === "running" && (
          <div className="flex flex-col items-center gap-4 py-12">
            <Loader2 className="h-12 w-12 animate-spin text-blue-600" />
            <div className="text-center">
              <p className="text-sm font-medium text-slate-900">Training models...</p>
              <p className="text-xs text-slate-500 mt-1">This may take a few minutes</p>
            </div>
          </div>
        )}
        </div>
      </CardContent>
    </Card>
  );
}
