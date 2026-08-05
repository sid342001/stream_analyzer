import { Check, ChevronsUp, Sparkles, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useStore } from "@/store/useStore";
import { ago, cn } from "@/lib/utils";
import type { EventItem, Severity } from "@/lib/types";

const sevRing: Record<Severity, string> = {
  critical: "border-l-alert",
  warning: "border-l-watch",
  info: "border-l-info",
};
const sevBadge: Record<Severity, "alert" | "watch" | "info"> = {
  critical: "alert",
  warning: "watch",
  info: "info",
};

/** The real analyzed-frame image retained for this event (see
 *  backend/app/pipeline.py's _queue_scene_analysis) when there is one, else
 *  a procedural placeholder for events from before that field existed. */
function Thumb({ ev }: { ev: EventItem }) {
  const label = ev.trackId != null ? `#${ev.trackId}` : "scene";
  if (ev.image) {
    return (
      <div className="relative h-14 w-20 shrink-0 overflow-hidden rounded border border-border bg-[#0a1017]">
        {/* eslint-disable-next-line jsx-a11y/alt-text */}
        <img src={ev.image} className="h-full w-full object-cover" />
        <div className="absolute bottom-0 left-0 right-0 bg-black/50 px-1 py-0.5 font-mono text-[8px] text-emerald-300">
          {label}
        </div>
      </div>
    );
  }
  const seed = ev.trackId ?? 0;
  return (
    <div className="relative h-14 w-20 shrink-0 overflow-hidden rounded border border-border bg-[#0a1017]">
      <div className="absolute inset-0 grid-noise opacity-40" />
      <div
        className="absolute h-5 w-6 rounded-sm border"
        style={{
          borderColor: ev.conceptColor,
          background: ev.conceptColor + "33",
          left: `${20 + (seed % 5) * 10}%`,
          top: `${25 + (seed % 4) * 12}%`,
        }}
      />
      <div className="absolute bottom-0 left-0 right-0 bg-black/50 px-1 py-0.5 font-mono text-[8px] text-emerald-300">
        {label}
      </div>
    </div>
  );
}

export function EventCard({ ev }: { ev: EventItem }) {
  const now = useStore((s) => s.now);
  const openEvent = useStore((s) => s.openEvent);
  const setVerdict = useStore((s) => s.setVerdict);
  const dismissed = ev.verdict === "dismissed";

  return (
    <div
      onClick={() => openEvent(ev.id)}
      className={cn(
        "group animate-slide-in cursor-pointer rounded-lg border border-border border-l-2 bg-card p-2.5 transition-colors hover:border-primary/40",
        sevRing[ev.severity],
        dismissed && "opacity-45",
      )}
    >
      <div className="flex gap-2.5">
        <Thumb ev={ev} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-[11px] text-muted-foreground">{ago(ev.ts, now)}</span>
            <div className="flex items-center gap-1">
              {ev.deepAnalyzed && (
                <Badge variant="info" className="gap-1 text-[9px]">
                  <Sparkles className="h-2.5 w-2.5" /> deep
                </Badge>
              )}
              <Badge variant={sevBadge[ev.severity]} className="text-[9px] capitalize">
                {ev.kind}
              </Badge>
            </div>
          </div>
          <p className="mt-1 line-clamp-2 text-xs leading-snug text-foreground">{ev.description}</p>
          <div className="mt-1.5 flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ background: ev.conceptColor }} />
            <span className="text-[10px] capitalize text-muted-foreground">{ev.concept}</span>
          </div>
        </div>
      </div>

      <div
        className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100"
        onClick={(e) => e.stopPropagation()}
      >
        <Button size="xs" variant="ghost" className="text-muted-foreground" onClick={() => setVerdict(ev.id, "dismissed")}>
          <X /> Dismiss
        </Button>
        <Button size="xs" variant="ghost" className="text-online" onClick={() => setVerdict(ev.id, "confirmed")}>
          <Check /> Confirm
        </Button>
        <Button size="xs" variant="ghost" className="text-alert" onClick={() => setVerdict(ev.id, "escalated")}>
          <ChevronsUp /> Escalate
        </Button>
        {ev.verdict !== "pending" && !dismissed && (
          <Badge variant="online" className="ml-auto text-[9px] capitalize">
            {ev.verdict}
          </Badge>
        )}
      </div>
    </div>
  );
}
