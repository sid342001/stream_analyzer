import { Inbox, FileDown, Crosshair } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { EventCard } from "./EventCard";
import { useStore } from "@/store/useStore";

export function EventFeed() {
  const events = useStore((s) => s.events);
  const tracks = useStore((s) => s.tracks);
  const openTrack = useStore((s) => s.openTrack);
  const pending = events.filter((e) => e.verdict === "pending").length;

  return (
    <aside className="flex w-[22rem] shrink-0 flex-col border-l border-border bg-panel">
      <Tabs defaultValue="scene" className="flex min-h-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-border px-3.5 py-2.5">
          <TabsList>
            <TabsTrigger value="scene">Scene Feed</TabsTrigger>
            <TabsTrigger value="objects">Objects ({tracks.length})</TabsTrigger>
          </TabsList>
          <Badge variant={pending ? "watch" : "default"}>{pending} new</Badge>
        </div>

        <TabsContent value="scene" className="flex min-h-0 flex-1 flex-col data-[state=inactive]:hidden">
          <ScrollArea className="flex-1 px-2.5 py-2.5">
            {events.length === 0 ? (
              <div className="flex h-40 flex-col items-center justify-center gap-2 text-center text-muted-foreground">
                <Inbox className="h-6 w-6" />
                <p className="text-xs">
                  Watching the feed…
                  <br />
                  periodic scene overviews appear here.
                </p>
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
        </TabsContent>

        <TabsContent value="objects" className="flex min-h-0 flex-1 flex-col data-[state=inactive]:hidden">
          <ScrollArea className="flex-1 px-2.5 py-2.5">
            {tracks.length === 0 ? (
              <div className="flex h-40 flex-col items-center justify-center gap-2 text-center text-muted-foreground">
                <Crosshair className="h-6 w-6" />
                <p className="text-xs">
                  Nothing detected right now.
                  <br />
                  live objects appear here as they're seen.
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-1">
                {tracks.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => openTrack(t.id)}
                    className="flex items-center gap-2 rounded-md px-2 py-2 text-left transition-colors hover:bg-accent"
                  >
                    <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: t.color }} />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-medium capitalize">
                        {t.concept} <span className="text-muted-foreground">#{t.id}</span>
                      </div>
                      <div className="text-[10px] text-muted-foreground">{Math.round(t.score * 100)}% confidence</div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </aside>
  );
}
