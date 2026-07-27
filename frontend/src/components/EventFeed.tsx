import { Inbox, FileDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { EventCard } from "./EventCard";
import { useStore } from "@/store/useStore";

export function EventFeed() {
  const events = useStore((s) => s.events);
  const pending = events.filter((e) => e.verdict === "pending").length;

  return (
    <aside className="flex w-[22rem] shrink-0 flex-col border-l border-border bg-panel">
      <div className="flex items-center justify-between border-b border-border px-3.5 py-2.5">
        <div>
          <div className="text-sm font-semibold">Event feed</div>
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
            what the system flagged
          </div>
        </div>
        <Badge variant={pending ? "watch" : "default"}>{pending} new</Badge>
      </div>

      <ScrollArea className="flex-1 px-2.5 py-2.5">
        {events.length === 0 ? (
          <div className="flex h-40 flex-col items-center justify-center gap-2 text-center text-muted-foreground">
            <Inbox className="h-6 w-6" />
            <p className="text-xs">Watching the feed…<br />flagged moments appear here.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {events.map((ev) => (
              <EventCard key={ev.id} ev={ev} />
            ))}
          </div>
        )}
      </ScrollArea>

      <div className="border-t border-border p-3">
        <Button variant="secondary" size="sm" className="w-full">
          <FileDown /> Export incident timeline
        </Button>
      </div>
    </aside>
  );
}
