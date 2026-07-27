import { Activity, Cpu, Radio, Wifi } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useStore } from "@/store/useStore";
import { fmtClock } from "@/lib/utils";

export function TopBar() {
  const now = useStore((s) => s.now);
  const events = useStore((s) => s.events);
  const pending = events.filter((e) => e.verdict === "pending").length;

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-panel px-4">
      <div className="flex items-center gap-3">
        <div className="flex h-7 w-7 items-center justify-center rounded bg-primary/15 text-primary">
          <Radio className="h-4 w-4" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-tight">UAV Analysis Console</div>
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
            AI analyst · offline
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4 text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <Cpu className="h-3.5 w-3.5" /> local · profile:local
        </span>
        <span className="flex items-center gap-1.5 text-online">
          <Wifi className="h-3.5 w-3.5" /> SAM 3 · DINOv3 · VLM
        </span>
        <Badge variant={pending ? "watch" : "default"} className="gap-1.5">
          <Activity className="h-3 w-3" /> {pending} to triage
        </Badge>
        <span className="font-mono text-sm tabular-nums text-foreground">{fmtClock(now)}</span>
      </div>
    </header>
  );
}
