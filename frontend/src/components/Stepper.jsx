import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

/** Horizontal workflow stepper. `current` is the active index; earlier steps show as done. */
export default function Stepper({ steps, current, failedIndex }) {
  return (
    <ol className="flex w-full items-center">
      {steps.map((s, i) => {
        const done = i < current;
        const active = i === current;
        const failed = i === failedIndex;
        return (
          <li key={s.label} className={cn("flex items-center", i < steps.length - 1 && "flex-1")}>
            <div className="flex items-center gap-2.5">
              <span
                className={cn(
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-xs font-semibold",
                  failed
                    ? "border-bad bg-bad-soft text-bad"
                    : done
                    ? "border-brand bg-brand text-white"
                    : active
                    ? "border-brand bg-surface text-brand-ink"
                    : "border-line-strong bg-surface text-ink-3"
                )}
              >
                {done && !failed ? <Check className="h-3.5 w-3.5" strokeWidth={3} /> : i + 1}
              </span>
              <div className="hidden sm:block">
                <p className={cn("text-[13px] leading-tight", active || done ? "font-medium text-ink" : "text-ink-3")}>{s.label}</p>
                {s.sub && <p className="text-2xs text-ink-3">{s.sub}</p>}
              </div>
            </div>
            {i < steps.length - 1 && <div className={cn("mx-3 h-px flex-1", done ? "bg-brand" : "bg-line-strong")} />}
          </li>
        );
      })}
    </ol>
  );
}
