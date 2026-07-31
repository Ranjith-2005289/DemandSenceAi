import axios from "axios";

// Set VITE_API_BASE_URL in a .env file (or the host platform's env config)
// to point at a deployed backend; defaults to the local dev server.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const uploadCsv = async (formData) => {
  try {
    const { data } = await api.post("/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  } catch (error) {
    console.error("Upload error:", error.response?.data || error.message);
    throw error;
  }
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
  console.log("\n🔗 API Layer: runForecast called");
  console.log("  Parameters:", { filename, date_col, target_col, forecast_horizon, group_col, group_value, country, session_id });

  try {
    const payload = {
      filename,
      date_col,
      target_col,
      forecast_horizon,
      ...(group_col ? { group_col, group_value } : {}),
      ...(country ? { country } : {}),
      ...(session_id ? { session_id } : {}),
    };
    console.log("  📤 Preparing POST request to /forecast");
    console.log("  📋 Payload:", JSON.stringify(payload, null, 2));
    
    const { data } = await api.post("/forecast", payload);
    
    console.log("  ✅ API response received successfully");
    console.log("  📊 Response status:", data?.status);
    console.log("  📊 Response data:", data);
    return data;
  } catch (error) {
    console.error("  ❌ Forecast API error");
    console.error("  Error type:", error?.constructor?.name);
    console.error("  Error message:", error?.message);
    console.error("  Response status:", error?.response?.status);
    console.error("  Response statusText:", error?.response?.statusText);
    console.error("  Response data:", error?.response?.data);
    console.error("  Full error:", error);
    throw error;
  }
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
