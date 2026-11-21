import {
  BarChart3,
  ChevronsLeft,
  ChevronsRight,
  FilePlus2,
  FileText,
  LayoutDashboard,
  Layers,
  LineChart,
  Package,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { navigate } from "@/lib/router";

export const NAV = [
  {
    section: "Overview",
    items: [{ id: "dashboard", label: "Dashboard", icon: LayoutDashboard }],
  },
  {
    section: "Planning",
    items: [
      { id: "new", label: "New forecast", icon: FilePlus2 },
      { id: "forecast", label: "Forecast", icon: LineChart, needsRun: true },
      { id: "models", label: "Model comparison", icon: Layers, needsRun: true },
      { id: "products", label: "Product breakdown", icon: Package, needsRun: true, needsGroup: true },
      { id: "report", label: "Insights report", icon: FileText, needsRun: true },
    ],
  },
];

export const PAGE_TITLE = {
  dashboard: "Dashboard",
  new: "New forecast",
  forecast: "Forecast",
  models: "Model comparison",
  products: "Product breakdown",
  report: "Insights report",
  settings: "Settings",
};

export function Logo({ compact }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded bg-brand">
        <BarChart3 className="h-4 w-4 text-white" strokeWidth={2.25} />
      </div>
      {!compact && <span className="text-[15px] font-semibold tracking-tight text-white">DemandSense</span>}
    </div>
  );
}

function NavItem({ item, active, disabledReason, collapsed }) {
  const Icon = item.icon;
  const disabled = Boolean(disabledReason);
  return (
    <button
      type="button"
      onClick={() => !disabled && navigate(item.id)}
      aria-current={active ? "page" : undefined}
      aria-disabled={disabled}
      title={disabled ? disabledReason : collapsed ? item.label : undefined}
      className={cn(
        "group relative flex h-8 w-full items-center gap-2.5 rounded px-2.5 text-[13px] transition-colors",
        active ? "bg-nav-active font-medium text-nav-ink-active" : "text-nav-ink",
        !active && !disabled && "hover:bg-nav-hover hover:text-nav-ink-active",
        disabled && "cursor-not-allowed opacity-45",
        collapsed && "justify-center px-0"
      )}
    >
      {active && <span className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-r bg-brand" />}
      <Icon className="h-4 w-4 shrink-0" />
      {!collapsed && <span className="truncate">{item.label}</span>}
    </button>
  );
}

export default function Sidebar({ route, hasRun, hasGroup, collapsed, onToggle }) {
  const reason = (item) => {
    if (item.needsRun && !hasRun) return "Run a forecast first";
    if (item.needsGroup && !hasGroup) return "Choose a 'Group by' column when configuring the forecast";
    return null;
  };

  return (
    <aside
      className={cn(
        "flex h-full shrink-0 flex-col border-r border-nav-line bg-nav transition-[width] duration-150",
        collapsed ? "w-14" : "w-56"
      )}
    >
      <div className={cn("flex h-12 items-center border-b border-nav-line", collapsed ? "justify-center" : "px-4")}>
        <Logo compact={collapsed} />
      </div>

      {!collapsed && (
        <div className="px-4 pb-1 pt-3">
          <p className="text-2xs font-medium uppercase tracking-wider text-nav-ink opacity-70">Workspace</p>
          <p className="mt-0.5 truncate text-[13px] font-medium text-nav-ink-active">Demand planning</p>
        </div>
      )}

      <nav className="scroll-thin flex-1 overflow-y-auto px-2 py-2" aria-label="Main">
        {NAV.map((group) => (
          <div key={group.section} className="mb-3">
            {!collapsed && (
              <p className="px-2.5 pb-1 pt-2 text-2xs font-medium uppercase tracking-wider text-nav-ink opacity-70">
                {group.section}
              </p>
            )}
            <div className="space-y-0.5">
              {group.items.map((item) => (
                <NavItem key={item.id} item={item} active={route === item.id} disabledReason={reason(item)} collapsed={collapsed} />
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="space-y-0.5 border-t border-nav-line px-2 py-2">
        <NavItem item={{ id: "settings", label: "Settings", icon: Settings }} active={route === "settings"} collapsed={collapsed} />
        <button
          type="button"
          onClick={onToggle}
          className={cn(
            "flex h-8 w-full items-center gap-2.5 rounded px-2.5 text-[13px] text-nav-ink transition-colors hover:bg-nav-hover hover:text-nav-ink-active",
            collapsed && "justify-center px-0"
          )}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronsRight className="h-4 w-4" /> : <ChevronsLeft className="h-4 w-4" />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
