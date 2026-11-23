import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  errorMessage,
  generateInsightsReport,
  getAsyncReportStatus,
  getProductSummary,
  runForecast,
  uploadCsv,
} from "../api";

// One place for the whole workflow: the uploaded dataset, the column mapping,
// the forecast run, a per-group drill-down, the product breakdown and the
// insights report. Pages read from here, so a run keeps going (and its result
// stays) while the user moves between pages.
const AppStateContext = createContext(null);

// featureCols: external drivers the user switched on (off by default).
const EMPTY_CONFIG = { dateCol: "", targetCol: "", groupCol: "", country: "", horizon: 30, featureCols: [] };

export function AppStateProvider({ children }) {
  const [sessionId] = useState(() => crypto.randomUUID());

  // Dataset + mapping
  const [file, setFile] = useState(null); // { name, size }
  const [upload, setUpload] = useState(null); // /upload response
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [config, setConfig] = useState(EMPTY_CONFIG);

  // Run
  const [run, setRun] = useState({ status: "idle", error: null, startedAt: null, finishedAt: null });
  const [runMeta, setRunMeta] = useState(null); // snapshot of what the finished run used
  const [result, setResult] = useState(null); // aggregate forecast
  const [groupResult, setGroupResult] = useState(null); // single-group drill-down
  const [groupRunning, setGroupRunning] = useState(null); // group value being forecast
  const [groupError, setGroupError] = useState(null);

  // Derived data
  const [products, setProducts] = useState({ data: null, loading: false, error: null });
  const [report, setReport] = useState({ data: null, loading: false, error: null, refining: false });
  const reportPoll = useRef(null);
  const hasResult = useRef(false);

  // After a failed attempt, picking a new file returns the status to what it
  // was before (ready if an earlier run's result is still shown).
  const clearFailure = () =>
    setRun((r) => (r.status === "failed" ? { ...r, status: hasResult.current ? "ready" : "idle", error: null } : r));

  const stopReportPoll = () => {
    if (reportPoll.current) clearInterval(reportPoll.current);
    reportPoll.current = null;
  };
  useEffect(() => stopReportPoll, []);

  const uploadFile = useCallback(async (f) => {
    setUploading(true);
    setUploadError(null);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await uploadCsv(fd);
      clearFailure();
      setFile({ name: f.name, size: f.size });
      setUpload(res);
      setConfig({
        ...EMPTY_CONFIG,
        dateCol: res?.suggestions?.date_col || "",
        targetCol: res?.suggestions?.target_col || "",
      });
      return true;
    } catch (err) {
      setUploadError(errorMessage(err, "Upload failed."));
      return false;
    } finally {
      setUploading(false);
    }
  }, []);

  const clearDataset = useCallback(() => {
    clearFailure();
    setFile(null);
    setUpload(null);
    setUploadError(null);
    setConfig(EMPTY_CONFIG);
  }, []);

  const startRun = useCallback(async () => {
    if (!upload?.file?.saved_as || !config.dateCol || !config.targetCol) return;
    const meta = {
      fileName: file?.name,
      savedAs: upload.file.saved_as,
      rows: upload.shape?.rows,
      cols: upload.shape?.cols,
      frequency: upload.suggestions?.frequency,
      warnings: upload.validation?.warnings || [],
      dateCol: config.dateCol,
      targetCol: config.targetCol,
      groupCol: config.groupCol || null,
      country: config.country || null,
      horizon: Number(config.horizon) || 30,
      // Drivers the user switched on (used by XGBoost, Random Forest and SARIMAX).
      featureCols: (config.featureCols || []).filter((c) => c !== config.targetCol),
    };
    const startedAt = Date.now();
    setRun({ status: "running", error: null, startedAt, finishedAt: null });
    try {
      const res = await runForecast(
        meta.savedAs,
        meta.dateCol,
        meta.targetCol,
        meta.horizon,
        null,
        null,
        meta.country,
        sessionId,
        meta.featureCols
      );
      stopReportPoll();
      hasResult.current = true;
      setResult(res);
      setGroupResult(null);
      setGroupError(null);
      setRunMeta({ ...meta, startedAt, finishedAt: Date.now() });
      setProducts({ data: null, loading: false, error: null });
      setReport({ data: null, loading: false, error: null, refining: false });
      setRun({ status: "ready", error: null, startedAt, finishedAt: Date.now() });
    } catch (err) {
      setRun({ status: "failed", error: errorMessage(err, "Forecast failed."), startedAt, finishedAt: Date.now() });
    }
  }, [upload, config, file, sessionId]);

  // Product breakdown: loaded once per finished run when a group column was chosen.
  useEffect(() => {
    if (!result || !runMeta?.groupCol) return;
    let cancelled = false;
    setProducts({ data: null, loading: true, error: null });
    getProductSummary(runMeta.savedAs, runMeta.dateCol, runMeta.targetCol, runMeta.groupCol, 50)
      .then((data) => !cancelled && setProducts({ data, loading: false, error: null }))
      .catch((err) => !cancelled && setProducts({ data: null, loading: false, error: errorMessage(err) }));
    return () => {
      cancelled = true;
    };
  }, [result, runMeta]);

  const forecastGroup = useCallback(
    async (value) => {
      if (!runMeta) return false;
      setGroupRunning(value);
      setGroupError(null);
      try {
        const res = await runForecast(
          runMeta.savedAs,
          runMeta.dateCol,
          runMeta.targetCol,
          runMeta.horizon,
          runMeta.groupCol,
          value,
          runMeta.country,
          sessionId,
          runMeta.featureCols
        );
        setGroupResult(res);
        return true;
      } catch (err) {
        setGroupError(errorMessage(err, "Forecast failed for this group."));
        return false;
      } finally {
        setGroupRunning(null);
      }
    },
    [runMeta, sessionId]
  );

  const loadReport = useCallback(
    async ({ force = false } = {}) => {
      if (!result || !runMeta) return;
      if (!force && (report.data || report.loading || report.error)) return;
      stopReportPoll();
      setReport({ data: null, loading: true, error: null, refining: false });
      const summary = runMeta.groupCol
        ? products.data ||
          (await getProductSummary(runMeta.savedAs, runMeta.dateCol, runMeta.targetCol, runMeta.groupCol, 50).catch(() => null))
        : null;
      try {
        const data = await generateInsightsReport(
          runMeta.savedAs,
          runMeta.dateCol,
          runMeta.targetCol,
          runMeta.groupCol,
          result,
          summary
        );
        setReport({ data, loading: false, error: null, refining: Boolean(data.async_report_path), at: Date.now() });
        // The backend also writes a fuller report in the background. Poll for
        // it (up to ~2 minutes) and swap it in, keeping the first one's charts.
        if (data.async_report_path) {
          let polls = 0;
          reportPoll.current = setInterval(async () => {
            polls += 1;
            if (polls > 60) {
              stopReportPoll();
              setReport((r) => ({ ...r, refining: false }));
              return;
            }
            try {
              const status = await getAsyncReportStatus(runMeta.savedAs);
              if (status.ready && status.report) {
                stopReportPoll();
                setReport((r) => ({
                  data: {
                    ...status.report,
                    chart_data: r.data?.chart_data,
                    stockout_alerts: status.report.stockout_alerts ?? r.data?.stockout_alerts,
                  },
                  loading: false,
                  error: null,
                  refining: false,
                  at: Date.now(),
                }));
              }
            } catch {
              /* transient: keep polling */
            }
          }, 2000);
        }
      } catch (err) {
        setReport({ data: null, loading: false, error: errorMessage(err, "Couldn't generate the report."), refining: false });
      }
    },
    [result, runMeta, products.data, report.data, report.loading, report.error]
  );

  const value = useMemo(
    () => ({
      sessionId,
      file,
      upload,
      uploading,
      uploadError,
      config,
      setConfig,
      uploadFile,
      clearDataset,
      run,
      runMeta,
      startRun,
      result,
      view: groupResult || result,
      groupResult,
      groupRunning,
      groupError,
      forecastGroup,
      clearGroup: () => setGroupResult(null),
      products,
      report,
      loadReport,
    }),
    [
      sessionId, file, upload, uploading, uploadError, config, uploadFile, clearDataset, run, runMeta,
      startRun, result, groupResult, groupRunning, groupError, forecastGroup, products, report, loadReport,
    ]
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export const useAppState = () => useContext(AppStateContext);
