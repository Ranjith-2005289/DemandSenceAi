import { cn } from "@/lib/utils";

/**
 * White header band at the top of each page (object-page style):
 * title, an optional line of facts, and the page's actions on the right.
 */
export default function PageHeader({ title, description, meta, actions, children, className }) {
  return (
    <div className={cn("border-b border-line bg-surface", className)}>
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-start justify-between gap-4 px-6 pb-4 pt-5">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold tracking-tight text-ink">{title}</h1>
          {description && <p className="mt-0.5 text-[13px] text-ink-3">{description}</p>}
          {meta?.length > 0 && (
            <dl className="mt-3 flex flex-wrap gap-x-8 gap-y-2">
              {meta.filter(Boolean).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-2xs font-medium uppercase tracking-wide text-ink-3">{k}</dt>
                  <dd className="mt-0.5 text-[13px] font-medium text-ink">{v}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
        {actions && <div className="no-print flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  );
}

export function PageBody({ className, children }) {
  return <div className={cn("mx-auto max-w-[1400px] space-y-4 px-6 py-5", className)}>{children}</div>;
}
