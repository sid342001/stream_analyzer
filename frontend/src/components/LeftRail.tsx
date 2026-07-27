import { ChevronRight, Library, Plus, Radio } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";
import type { Priority } from "@/lib/types";

const priorityBadge: Record<Priority, "alert" | "watch" | "default"> = {
  alert: "alert",
  watch: "watch",
  ignore: "default",
};

export function LeftRail() {
  const streams = useStore((s) => s.streams);
  const selectedStream = useStore((s) => s.selectedStream);
  const selectStream = useStore((s) => s.selectStream);
  const concepts = useStore((s) => s.concepts);
  const toggleConcept = useStore((s) => s.toggleConcept);
  const cyclePriority = useStore((s) => s.cycleConceptPriority);

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-panel">
      <div className="p-3">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Streams
        </div>
        <div className="flex flex-col gap-1">
          {streams.map((s) => (
            <button
              key={s.id}
              onClick={() => selectStream(s.id)}
              className={cn(
                "group flex items-center gap-2 rounded-md border px-2.5 py-2 text-left transition-colors",
                selectedStream === s.id
                  ? "border-primary/40 bg-primary/10"
                  : "border-transparent hover:bg-accent",
              )}
            >
              <Radio className={cn("h-3.5 w-3.5", s.online ? "text-online" : "text-muted-foreground")} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs font-medium">{s.name}</div>
                <div className="truncate font-mono text-[10px] text-muted-foreground">{s.source}</div>
              </div>
              {s.online && <span className="h-1.5 w-1.5 rounded-full bg-online" />}
            </button>
          ))}
        </div>
      </div>

      <Separator />

      <div className="flex items-center justify-between px-3 pt-3">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Watch-list
        </div>
        <Badge variant="outline" className="text-[10px]">{concepts.filter((c) => c.enabled).length} live</Badge>
      </div>

      <ScrollArea className="flex-1 px-2 py-2">
        <div className="flex flex-col gap-1">
          {concepts.map((c) => (
            <div
              key={c.id}
              className="flex items-center gap-2 rounded-md px-1.5 py-1.5 hover:bg-accent/60"
            >
              <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: c.color }} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs font-medium capitalize">{c.label}</div>
                <div className="text-[10px] text-muted-foreground">{c.exemplars} exemplars</div>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button onClick={() => cyclePriority(c.id)}>
                    <Badge variant={priorityBadge[c.priority]} className="cursor-pointer text-[10px] capitalize">
                      {c.priority}
                    </Badge>
                  </button>
                </TooltipTrigger>
                <TooltipContent>Click to cycle priority</TooltipContent>
              </Tooltip>
              <Switch checked={c.enabled} onCheckedChange={() => toggleConcept(c.id)} />
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="flex flex-col gap-2 border-t border-border p-3">
        <Button variant="outline" size="sm" className="justify-start">
          <Plus /> Add context
        </Button>
        <Button variant="ghost" size="sm" className="justify-between text-muted-foreground">
          <span className="flex items-center gap-2">
            <Library /> Concept library
          </span>
          <ChevronRight className="h-3.5 w-3.5" />
        </Button>
      </div>
    </aside>
  );
}
