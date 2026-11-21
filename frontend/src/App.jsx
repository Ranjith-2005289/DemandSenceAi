import { useEffect, useState } from "react";
import Sidebar from "./components/shell/Sidebar";
import TopBar from "./components/shell/TopBar";
import AssistantPanel from "./components/AssistantPanel";
import { navigate, useRoute } from "./lib/router";
import { useAppState } from "./state/AppState";
import DashboardPage from "./pages/Dashboard";
import NewForecastPage from "./pages/NewForecast";
import ForecastPage from "./pages/Forecast";
import ModelsPage from "./pages/Models";
import ProductsPage from "./pages/Products";
import ReportPage from "./pages/Report";
import SettingsPage from "./pages/Settings";

const PAGES = {
  dashboard: DashboardPage,
  new: NewForecastPage,
  forecast: ForecastPage,
  models: ModelsPage,
  products: ProductsPage,
  report: ReportPage,
  settings: SettingsPage,
};

const SIDEBAR_KEY = "demandsense.sidebar-collapsed";

export default function App() {
  const route = useRoute();
  const { result, runMeta } = useAppState();
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_KEY) === "1";
    } catch {
      return false;
    }
  });

  const narrow = useNarrow();
  const hasRun = Boolean(result);
  const hasGroup = Boolean(runMeta?.groupCol);

  // Pages that need a finished run fall back to the dashboard (e.g. after a reload).
  useEffect(() => {
    const needsRun = ["forecast", "models", "products", "report"].includes(route);
    if ((needsRun && !hasRun) || (route === "products" && !hasGroup)) navigate("dashboard");
  }, [route, hasRun, hasGroup]);

  const toggleSidebar = () =>
    setCollapsed((c) => {
      try {
        localStorage.setItem(SIDEBAR_KEY, c ? "0" : "1");
      } catch {
        /* per-tab only */
      }
      return !c;
    });

  const Page = PAGES[route];

  return (
    <div className="flex h-full overflow-hidden bg-app">
      <Sidebar route={route} hasRun={hasRun} hasGroup={hasGroup} collapsed={collapsed || narrow} onToggle={toggleSidebar} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBarConnected route={route} assistantOpen={assistantOpen} setAssistantOpen={setAssistantOpen} hasRun={hasRun} />
        <div className="flex min-h-0 flex-1">
          <main className="scroll-thin min-w-0 flex-1 overflow-y-auto" key={route}>
            <Page openAssistant={() => setAssistantOpen(true)} />
          </main>
          {hasRun && <AssistantPanel open={assistantOpen} onClose={() => setAssistantOpen(false)} />}
        </div>
      </div>
    </div>
  );
}

// Below 900px the sidebar stays in its icon-only form.
function useNarrow() {
  const query = "(max-width: 900px)";
  const [narrow, setNarrow] = useState(() => window.matchMedia?.(query).matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return;
    const onChange = (e) => setNarrow(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

function TopBarConnected({ route, assistantOpen, setAssistantOpen, hasRun }) {
  const { run } = useAppState();
  return (
    <TopBar
      route={route}
      run={run}
      assistantOpen={assistantOpen && hasRun}
      assistantAvailable={hasRun}
      onToggleAssistant={() => setAssistantOpen((o) => !o)}
    />
  );
}
