import { createContext, useContext, useEffect, useMemo, useState } from "react";

// Light by default; the user can pick light, dark, or follow the OS.
const ThemeContext = createContext(null);
const STORAGE_KEY = "demandsense.theme";

function readStored() {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "dark" || v === "system" ? v : "light";
  } catch {
    return "light";
  }
}

function systemPrefersDark() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
}

export function ThemeProvider({ children }) {
  const [preference, setPreference] = useState(readStored);
  const [systemDark, setSystemDark] = useState(systemPrefersDark);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const onChange = (e) => setSystemDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const resolved = preference === "system" ? (systemDark ? "dark" : "light") : preference;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark");
  }, [resolved]);

  const value = useMemo(
    () => ({
      preference,
      resolved,
      setPreference: (p) => {
        setPreference(p);
        try {
          localStorage.setItem(STORAGE_KEY, p);
        } catch {
          /* storage unavailable: preference lasts for this tab only */
        }
      },
    }),
    [preference, resolved]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

const CHART_VARS = {
  actual: "--chart-actual",
  forecast: "--chart-forecast",
  band: "--chart-band",
  grid: "--chart-grid",
  axis: "--chart-axis",
  alert: "--chart-alert",
  c1: "--chart-1",
  c2: "--chart-2",
  c3: "--chart-3",
  c4: "--chart-4",
  other: "--chart-other",
  ink: "--ink",
  ink2: "--ink-2",
  line: "--line",
  surface: "--surface",
};

/** Resolved chart colours for the active theme (SVG attributes can't use var()). */
export function useChartPalette() {
  const { resolved } = useTheme();
  return useMemo(() => {
    const css = getComputedStyle(document.documentElement);
    return Object.fromEntries(
      Object.entries(CHART_VARS).map(([k, v]) => [k, css.getPropertyValue(v).trim()])
    );
    // `resolved` is the trigger: the .dark class flips before this recomputes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolved]);
}
