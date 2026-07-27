import { useStore } from "@/store/useStore";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtTime } from "@/lib/utils";
import type { Severity } from "@/lib/types";

const sevColor: Record<Severity, string> = {
  critical: "bg-alert",
  warning: "bg-watch",
  info: "bg-info",
};

export function TimelineStrip() {
  const events = useStore((s) => s.events);
  const now = useStore((s) => s.now);
  const openEvent = useStore((s) => s.openEvent);

  const span = 120_000; // last 2 minutes
  const start = now - span;

  return (
    <div className="flex h-14 shrink-0 items-center gap-3 border-t border-border bg-panel px-4">
      <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
        Session
      </div>
      <div className="relative h-8 flex-1 rounded-md border border-border bg-background">
        <div className="grid-noise absolute inset-0 rounded-md opacity-30" />
        {/* minute gridlines */}
        {[0.25, 0.5, 0.75].map((f) => (
          <div key={f} className="absolute top-0 h-full w-px bg-border" style={{ left: `${f * 100}%` }} />
        ))}
        {events.map((ev) => {
          const pos = ((ev.ts - start) / span) * 100;
          if (pos < 0 || pos > 100) return null;
          return (
            <Tooltip key={ev.id}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => openEvent(ev.id)}
                  className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-background transition-transform hover:scale-150 ${sevColor[ev.severity]}`}
                  style={{ left: `${pos}%` }}
                />
              </TooltipTrigger>
              <TooltipContent>
                <div className="font-mono text-[10px]">{fmtTime(ev.ts)}</div>
                <div className="max-w-[200px] text-xs">{ev.description}</div>
              </TooltipContent>
            </Tooltip>
          );
        })}
        <div className="absolute right-1 top-1/2 -translate-y-1/2 font-mono text-[10px] text-muted-foreground">
          now
        </div>
      </div>
    </div>
  );
}
