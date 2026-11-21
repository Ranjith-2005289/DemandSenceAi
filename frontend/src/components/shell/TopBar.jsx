import { useEffect, useRef, useState } from "react";
import { Check, ChevronRight, Monitor, Moon, MessageSquareText, Settings, Sun, User } from "lucide-react";
import { navigate } from "@/lib/router";
import { useTheme } from "@/lib/theme";
import { Button, StatusDot } from "@/components/ui";
import { PAGE_TITLE } from "./Sidebar";

function Elapsed({ since }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const s = Math.max(0, Math.floor((now - since) / 1000));
  return <span className="tabular-nums">{`${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`}</span>;
}

function RunStatus({ run }) {
  const map = {
    idle: { tone: "neutral", label: "No active run" },
    running: { tone: "brand", label: "Forecast running" },
    ready: { tone: "ok", label: "Forecast ready" },
    failed: { tone: "bad", label: "Last run failed" },
  };
  const s = map[run.status];
  return (
    <button
      type="button"
      onClick={() => navigate(run.status === "running" || run.status === "failed" ? "new" : "dashboard")}
      className="hidden h-7 items-center gap-2 rounded border border-line px-2.5 text-xs text-ink-2 transition-colors hover:bg-surface-hover md:inline-flex"
      title="Run status"
    >
      <StatusDot tone={s.tone} />
      {s.label}
      {run.status === "running" && (
        <span className="text-ink-3">
          · <Elapsed since={run.startedAt} />
        </span>
      )}
    </button>
  );
}

function UserMenu() {
  const { preference, setPreference } = useTheme();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => !ref.current?.contains(e.target) && setOpen(false);
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const themes = [
    { id: "light", label: "Light", icon: Sun },
    { id: "dark", label: "Dark", icon: Moon },
    { id: "system", label: "Match system", icon: Monitor },
  ];

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex h-8 items-center gap-2 rounded pl-1 pr-2 transition-colors hover:bg-surface-hover"
      >
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-surface-muted ring-1 ring-line">
          <User className="h-3.5 w-3.5 text-ink-2" />
        </span>
        <span className="hidden text-[13px] text-ink-2 sm:inline">Account</span>
      </button>
      {open && (
        <div role="menu" className="absolute right-0 top-9 z-50 w-56 rounded-md border border-line bg-surface py-1 shadow-pop">
          <div className="border-b border-line px-3 pb-2 pt-1.5">
            <p className="text-[13px] font-medium text-ink">Signed-in user</p>
            <p className="text-xs text-ink-3">Demand planning workspace</p>
          </div>
          <p className="px-3 pb-1 pt-2 text-2xs font-medium uppercase tracking-wide text-ink-3">Appearance</p>
          {themes.map((t) => (
            <button
              key={t.id}
              type="button"
              role="menuitemradio"
              aria-checked={preference === t.id}
              onClick={() => setPreference(t.id)}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-[13px] text-ink hover:bg-surface-hover"
            >
              <t.icon className="h-3.5 w-3.5 text-ink-3" />
              <span className="flex-1 text-left">{t.label}</span>
              {preference === t.id && <Check className="h-3.5 w-3.5 text-brand" />}
            </button>
          ))}
          <div className="my-1 border-t border-line" />
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              navigate("settings");
            }}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-[13px] text-ink hover:bg-surface-hover"
          >
            <Settings className="h-3.5 w-3.5 text-ink-3" /> Settings
          </button>
        </div>
      )}
    </div>
  );
}

export default function TopBar({ route, run, assistantOpen, onToggleAssistant, assistantAvailable }) {
  return (
    <header className="flex h-12 shrink-0 items-center justify-between gap-4 border-b border-line bg-surface px-4">
      <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5 text-[13px]">
        <span className="text-ink-3">Demand planning</span>
        <ChevronRight className="h-3.5 w-3.5 text-ink-3" />
        <span className="truncate font-medium text-ink">{PAGE_TITLE[route]}</span>
      </nav>
      <div className="flex items-center gap-2">
        <RunStatus run={run} />
        {/* Wrapper carries the tooltip: browsers don't show titles on disabled buttons. */}
        <span title={assistantAvailable ? "Ask questions about this forecast" : "Available after a forecast has run"}>
          <Button
            size="sm"
            variant={assistantOpen ? "primary" : "secondary"}
            icon={MessageSquareText}
            onClick={onToggleAssistant}
            disabled={!assistantAvailable}
            aria-pressed={assistantOpen}
          >
            Assistant
          </Button>
        </span>
        <div className="mx-1 h-5 w-px bg-line" />
        <UserMenu />
      </div>
    </header>
  );
}
