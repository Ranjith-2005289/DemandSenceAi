// Shared UI primitives — one visual language for every screen.

export function Card({ className = "", children, ...props }) {
  return (
    <section
      className={`rounded-2xl border border-slate-200/80 bg-white shadow-card ${className}`}
      {...props}
    >
      {children}
    </section>
  );
}

export function CardHeader({ title, subtitle, actions, className = "" }) {
  return (
    <div className={`flex flex-col gap-3 border-b border-slate-100 px-6 py-4 sm:flex-row sm:items-center sm:justify-between ${className}`}>
      <div className="min-w-0">
        <h3 className="text-[15px] font-semibold text-slate-900">{title}</h3>
        {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

const BADGE_TONES = {
  neutral: "bg-slate-100 text-slate-700 ring-slate-200",
  brand: "bg-brand-50 text-brand-700 ring-brand-200",
  success: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  warning: "bg-amber-50 text-amber-800 ring-amber-200",
  danger: "bg-rose-50 text-rose-700 ring-rose-200",
};

export function Badge({ tone = "neutral", children, className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${BADGE_TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

const BUTTON_VARIANTS = {
  primary:
    "bg-brand-600 text-white shadow-sm hover:bg-brand-700 focus-visible:outline-brand-600 disabled:bg-slate-300 disabled:text-white",
  secondary:
    "bg-white text-slate-700 ring-1 ring-inset ring-slate-300 shadow-sm hover:bg-slate-50 disabled:text-slate-400",
  ghost: "text-slate-600 hover:bg-slate-100 hover:text-slate-900 disabled:text-slate-400",
  dark: "bg-slate-900 text-white shadow-sm hover:bg-slate-800 disabled:bg-slate-300",
};

export function Button({ variant = "primary", size = "md", className = "", children, ...props }) {
  const sizes = { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2 text-sm", lg: "px-5 py-2.5 text-sm" };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed ${sizes[size]} ${BUTTON_VARIANTS[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Stat({ label, value, sub, tone, icon }) {
  const subTone =
    tone === "good" ? "text-emerald-600" : tone === "bad" ? "text-rose-600" : "text-slate-500";
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>
      <p className="mt-2 truncate text-2xl font-semibold tracking-tight text-slate-900 tabular-nums">
        {value}
      </p>
      {sub && <p className={`mt-1 text-xs font-medium ${subTone}`}>{sub}</p>}
    </Card>
  );
}

export function Spinner({ className = "h-4 w-4" }) {
  return (
    <svg className={`animate-spin ${className}`} fill="none" viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

export function Alert({ tone = "danger", title, children }) {
  const tones = {
    danger: "border-rose-200 bg-rose-50 text-rose-800",
    warning: "border-amber-200 bg-amber-50 text-amber-900",
    info: "border-brand-200 bg-brand-50 text-brand-900",
  };
  return (
    <div className={`rounded-xl border px-4 py-3 text-sm ${tones[tone]}`} role="alert">
      {title && <p className="font-semibold">{title}</p>}
      <div className={title ? "mt-0.5" : ""}>{children}</div>
    </div>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "block w-full rounded-lg border-0 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm ring-1 ring-inset ring-slate-300 transition placeholder:text-slate-400 focus:ring-2 focus:ring-inset focus:ring-brand-500";

// ── Icons (inline, stroke-based, 20px) ─────────────────────────────────────
const base = {
  width: 20, height: 20, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
  strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true,
};
export const Icon = {
  Logo: (p) => (
    <svg {...base} {...p}><path d="M3 17l5-5 4 4 8-8" /><path d="M14 8h6v6" /></svg>
  ),
  Upload: (p) => (
    <svg {...base} {...p}><path d="M12 16V4" /><path d="M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" /></svg>
  ),
  File: (p) => (
    <svg {...base} {...p}><path d="M14 3H6a1 1 0 00-1 1v16a1 1 0 001 1h12a1 1 0 001-1V8z" /><path d="M14 3v5h5" /></svg>
  ),
  Trophy: (p) => (
    <svg {...base} {...p}><path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 01-10 0z" /><path d="M17 5h3v2a3 3 0 01-3 3M7 5H4v2a3 3 0 003 3" /></svg>
  ),
  Target: (p) => (
    <svg {...base} {...p}><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1" /></svg>
  ),
  Chart: (p) => (
    <svg {...base} {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
  ),
  Calendar: (p) => (
    <svg {...base} {...p}><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M16 3v4M8 3v4M3 11h18" /></svg>
  ),
  Sparkles: (p) => (
    <svg {...base} {...p}><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z" /><path d="M19 17l.8 2.2L22 20l-2.2.8L19 23l-.8-2.2L16 20l2.2-.8z" /></svg>
  ),
  Chat: (p) => (
    <svg {...base} {...p}><path d="M21 12a8 8 0 01-11.6 7.1L4 20l1-4.6A8 8 0 1121 12z" /></svg>
  ),
  Download: (p) => (
    <svg {...base} {...p}><path d="M12 4v12M7 11l5 5 5-5M4 20h16" /></svg>
  ),
  Refresh: (p) => (
    <svg {...base} {...p}><path d="M20 11a8 8 0 10-2.3 5.7" /><path d="M20 4v7h-7" /></svg>
  ),
  ArrowLeft: (p) => (
    <svg {...base} {...p}><path d="M19 12H5M11 18l-6-6 6-6" /></svg>
  ),
  Check: (p) => (
    <svg {...base} {...p}><path d="M5 12l5 5L20 7" /></svg>
  ),
  X: (p) => (
    <svg {...base} {...p}><path d="M6 6l12 12M18 6L6 18" /></svg>
  ),
  Send: (p) => (
    <svg {...base} {...p}><path d="M5 12l14-7-5 14-2-6z" /></svg>
  ),
  Alert: (p) => (
    <svg {...base} {...p}><path d="M12 9v4M12 17h.01" /><path d="M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" /></svg>
  ),
};
