import axios from "axios";

// Set VITE_API_BASE_URL in a .env file (or the host platform's env config)
// to point at a deployed backend; defaults to the local dev server.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const api = axios.create({ baseURL: API_BASE_URL });

/** Human-readable message from an axios error (FastAPI puts it in `detail`). */
export function errorMessage(error, fallback = "Something went wrong.") {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  if (error?.code === "ERR_NETWORK") {
    return `Can't reach the forecasting server at ${API_BASE_URL}. Is the backend running?`;
  }
  return error?.message || fallback;
}

export const uploadCsv = async (formData) => {
  const { data } = await api.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const runForecast = async (
  filename,
  date_col,
  target_col,
  forecast_horizon = 30,
  group_col = null,
  group_value = null,
  country = null,
  session_id = null
) => {
  const payload = {
    filename,
    date_col,
    target_col,
    forecast_horizon,
    ...(group_col ? { group_col, group_value } : {}),
    ...(country ? { country } : {}),
    ...(session_id ? { session_id } : {}),
  };
  const { data } = await api.post("/forecast", payload);
  return data;
};

export const getLatestForecast = async (session_id = null) => {
  const { data } = await api.get("/forecast/latest", {
    params: session_id ? { session_id } : {},
  });
  return data;
};

export const getProductSummary = async (filename, date_col, target_col, group_col, top_n = 50) => {
  const { data } = await api.post("/forecast/product-summary", {
    filename,
    date_col,
    target_col,
    group_col,
    top_n,
  });
  return data;
};

export const generateInsightsReport = async (
  filename,
  date_col,
  target_col,
  group_col,
  forecast_result,
  product_summary
) => {
  const { data } = await api.post("/api/insights/generate", {
    filename,
    date_col,
    target_col,
    group_col: group_col || null,
    forecast_result,
    product_summary: product_summary || null,
  });
  return data;
};

export const getAsyncReportStatus = async (filename) => {
  const { data } = await api.get("/api/insights/async_report_status", { params: { filename } });
  return data;
};

export const askChat = async ({ question, session_id, forecast_data, product_summary }) => {
  const { data } = await api.post("/api/chat/ask", {
    question,
    session_id,
    forecast_data,
    product_summary,
  });
  return data;
};

export const resetChat = (session_id) => api.post(`/api/chat/reset/${session_id}`);

export const getHealth = async () => {
  const { data } = await api.get("/health", { timeout: 5000 });
  return data;
};
