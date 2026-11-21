import { forwardRef } from "react";
import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

/* ── Buttons ─────────────────────────────────────────────────────────────── */

const BUTTON_VARIANTS = {
  primary: "bg-brand text-white border-brand hover:bg-brand-hover hover:border-brand-hover",
  secondary: "bg-surface text-ink border-line-strong hover:bg-surface-hover",
  ghost: "bg-transparent text-ink-2 border-transparent hover:bg-surface-hover hover:text-ink",
  danger: "bg-surface text-bad border-line-strong hover:bg-bad-soft",
};
const BUTTON_SIZES = {
  sm: "h-7 px-2.5 text-xs gap-1.5",
  md: "h-8 px-3 text-[13px] gap-1.5",
  lg: "h-9 px-4 text-[13px] gap-2",
};

export const Button = forwardRef(function Button(
  { variant = "secondary", size = "md", icon: Icon, loading = false, className, children, disabled, ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type="button"
      disabled={disabled || loading}
      className={cn(
        "inline-flex shrink-0 items-center justify-center whitespace-nowrap rounded border font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className
      )}
      {...props}
    >
      {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : Icon ? <Icon className="h-3.5 w-3.5" /> : null}
      {children}
    </button>
  );
});

export function IconButton({ icon: Icon, label, className, active, ...props }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      className={cn(
        "inline-flex h-8 w-8 items-center justify-center rounded text-ink-2 transition-colors hover:bg-surface-hover hover:text-ink disabled:opacity-40",
        active && "bg-brand-soft text-brand-ink",
        className
      )}
      {...props}
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}

/* ── Panels ──────────────────────────────────────────────────────────────── */

export function Panel({ className, children, ...props }) {
  return (
    <section className={cn("rounded-md border border-line bg-surface shadow-panel", className)} {...props}>
      {children}
    </section>
  );
}

export function PanelHeader({ title, subtitle, actions, className }) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-3 border-b border-line px-4 py-3", className)}>
      <div className="min-w-0">
        <h3 className="text-[13px] font-semibold text-ink">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function PanelBody({ className, children }) {
  return <div className={cn("p-4", className)}>{children}</div>;
}

export function PanelFooter({ className, children }) {
  return <div className={cn("flex items-center justify-between gap-3 border-t border-line px-4 py-2.5", className)}>{children}</div>;
}

/* ── Status chips ────────────────────────────────────────────────────────── */

const CHIP_TONES = {
  neutral: "bg-surface-muted text-ink-2 border-line",
  brand: "bg-brand-soft text-brand-ink border-transparent",
  ok: "bg-ok-soft text-ok border-transparent",
  warn: "bg-warn-soft text-warn border-transparent",
  bad: "bg-bad-soft text-bad border-transparent",
};

export function Chip({ tone = "neutral", icon: Icon, children, className, ...props }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-px text-2xs font-medium",
        CHIP_TONES[tone],
        className
      )}
      {...props}
    >
      {Icon && <Icon className="h-3 w-3" />}
      {children}
    </span>
  );
}

export function StatusDot({ tone = "neutral" }) {
  const c = { neutral: "bg-ink-3", ok: "bg-ok", warn: "bg-warn", bad: "bg-bad", brand: "bg-brand" }[tone];
  return <span className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", c)} />;
}

/* ── KPI tile ────────────────────────────────────────────────────────────── */

export function Kpi({ label, value, sub, tone, title }) {
  const subTone = { ok: "text-ok", warn: "text-warn", bad: "text-bad" }[tone] || "text-ink-3";
  return (
    <Panel className="px-4 py-3" title={title}>
      <p className="text-2xs font-medium uppercase tracking-wide text-ink-3">{label}</p>
      <p className="mt-1.5 truncate text-xl font-semibold tabular-nums text-ink">{value}</p>
      {sub && <p className={cn("mt-0.5 truncate text-xs", subTone)}>{sub}</p>}
    </Panel>
  );
}

/* ── Banners (inline messages) ───────────────────────────────────────────── */

const BANNER = {
  info: { icon: Info, cls: "bg-brand-soft text-ink", ic: "text-brand" },
  ok: { icon: CheckCircle2, cls: "bg-ok-soft text-ink", ic: "text-ok" },
  warn: { icon: AlertTriangle, cls: "bg-warn-soft text-ink", ic: "text-warn" },
  bad: { icon: XCircle, cls: "bg-bad-soft text-ink", ic: "text-bad" },
};

export function Banner({ tone = "info", title, children, action, className }) {
  const b = BANNER[tone];
  const Icon = b.icon;
  return (
    <div className={cn("flex items-start gap-2.5 rounded-md border border-transparent px-3 py-2.5", b.cls, className)} role={tone === "bad" ? "alert" : "status"}>
      <Icon className={cn("mt-px h-4 w-4 shrink-0", b.ic)} />
      <div className="min-w-0 flex-1 text-[13px]">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={cn("text-ink-2", title && "mt-0.5")}>{children}</div>}
      </div>
      {action}
    </div>
  );
}

/* ── Form fields ─────────────────────────────────────────────────────────── */

export function Field({ label, hint, required, htmlFor, children, className }) {
  return (
    <div className={cn("space-y-1", className)}>
      <label htmlFor={htmlFor} className="block text-xs font-medium text-ink-2">
        {label}
        {required && <span className="ml-0.5 text-bad">*</span>}
      </label>
      {children}
      {hint && <p className="text-2xs text-ink-3">{hint}</p>}
    </div>
  );
}

const CONTROL =
  "h-8 w-full rounded border border-line-strong bg-surface px-2.5 text-[13px] text-ink transition-colors placeholder:text-ink-3 hover:border-ink-3 focus:border-brand disabled:cursor-not-allowed disabled:opacity-60";

export function Select({ value, onChange, options, placeholder, className, ...props }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={cn(CONTROL, "cursor-pointer appearance-none bg-[length:16px] bg-[right_6px_center] bg-no-repeat pr-7", className)}
      style={{
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2375839a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")",
      }}
      {...props}
    >
      {placeholder !== undefined && <option value="">{placeholder}</option>}
      {options.map((o) => {
        const opt = typeof o === "string" ? { value: o, label: o } : o;
        return (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        );
      })}
    </select>
  );
}

export const TextInput = forwardRef(function TextInput({ className, ...props }, ref) {
  return <input ref={ref} className={cn(CONTROL, className)} {...props} />;
});

/* ── Segmented control ───────────────────────────────────────────────────── */

export function Segmented({ value, onChange, options, className }) {
  return (
    <div className={cn("inline-flex rounded border border-line-strong bg-surface p-0.5", className)} role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-sm px-2.5 py-0.5 text-xs font-medium transition-colors",
            value === o.value ? "bg-brand-soft text-brand-ink" : "text-ink-2 hover:text-ink"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ── Empty / loading states ──────────────────────────────────────────────── */

export function EmptyState({ icon: Icon, title, children, action, className }) {
  return (
    <div className={cn("flex flex-col items-center justify-center px-6 py-14 text-center", className)}>
      {Icon && (
        <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-md border border-line bg-surface-muted">
          <Icon className="h-5 w-5 text-ink-3" />
        </div>
      )}
      <p className="text-sm font-semibold text-ink">{title}</p>
      {children && <div className="mt-1 max-w-md text-[13px] text-ink-3">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Loading({ label = "Loading…", className }) {
  return (
    <div className={cn("flex items-center justify-center gap-2 py-10 text-[13px] text-ink-3", className)}>
      <Loader2 className="h-4 w-4 animate-spin text-brand" /> {label}
    </div>
  );
}

/* ── Key/value list (object-page style facts) ────────────────────────────── */

export function Facts({ items, className }) {
  return (
    <dl className={cn("divide-y divide-line", className)}>
      {items.filter(Boolean).map(([k, v], i) => (
        <div key={`${k}-${i}`} className="flex items-baseline justify-between gap-4 py-1.5">
          <dt className="shrink-0 text-xs text-ink-3">{k}</dt>
          <dd className="min-w-0 truncate text-right text-[13px] font-medium text-ink" title={typeof v === "string" ? v : undefined}>
            {v}
          </dd>
        </div>
      ))}
    </dl>
  );
}
