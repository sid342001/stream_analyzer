import { useEffect, useRef, useState } from "react";
import { Crosshair, MapPin, Send, Sparkles } from "lucide-react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useStore } from "@/store/useStore";
import { mockAnswer } from "@/lib/feed";
import { fmtClock } from "@/lib/utils";

export function DetailView() {
  const selectedEvent = useStore((s) => s.selectedEvent);
  const events = useStore((s) => s.events);
  const openEvent = useStore((s) => s.openEvent);
  const markDeep = useStore((s) => s.markDeepAnalyzed);
  const chats = useStore((s) => s.chats);
  const pushChat = useStore((s) => s.pushChat);

  const ev = events.find((e) => e.id === selectedEvent) ?? null;
  const [q, setQ] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ev && !ev.deepAnalyzed) {
      const t = setTimeout(() => markDeep(ev.id), 650);
      return () => clearTimeout(t);
    }
  }, [ev, markDeep]);

  if (!ev) return null;
  const messages = chats[ev.id] ?? [];

  const ask = (text: string) => {
    if (!text.trim()) return;
    pushChat(ev.id, { id: `m_${Date.now()}`, role: "user", content: text });
    const answer = mockAnswer(text, ev);
    setTimeout(() => {
      pushChat(ev.id, {
        id: `a_${Date.now()}`,
        role: "assistant",
        content: answer,
        grounded: `track #${ev.trackId} · ${fmtClock(ev.ts)}`,
      });
      scrollRef.current?.scrollTo({ top: 1e9 });
    }, 450);
    setQ("");
  };

  const suggestions = ["How many people near the tents?", "What is this object doing?", "When did the vehicle first appear?"];

  return (
    <Dialog open={!!selectedEvent} onOpenChange={(o) => !o && openEvent(null)}>
      <DialogContent>
        <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-3">
          <Crosshair className="h-4 w-4 text-primary" />
          <DialogTitle>Event investigation</DialogTitle>
          <Badge variant="info" className="gap-1">
            <Sparkles className="h-3 w-3" /> Tier 3 · deep-analyzed
          </Badge>
        </div>

        <div className="grid grid-cols-[1.1fr_1fr] gap-0">
          {/* left: pinned evidence */}
          <div className="border-r border-border p-4">
            <div className="relative aspect-video overflow-hidden rounded-md border border-border bg-[#0a1017]">
              <div className="grid-noise absolute inset-0 opacity-40" />
              <div
                className="absolute rounded-sm border-2"
                style={{
                  borderColor: ev.conceptColor,
                  background: ev.conceptColor + "22",
                  left: "38%",
                  top: "34%",
                  width: "18%",
                  height: "26%",
                }}
              />
              <div
                className="absolute px-1 py-0.5 font-mono text-[9px]"
                style={{ background: ev.conceptColor, color: "#04121f", left: "38%", top: "30%" }}
              >
                {ev.concept} #{ev.trackId}
              </div>
              <div className="absolute left-2 top-2 rounded bg-black/50 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                {fmtClock(ev.ts)}
              </div>
            </div>

            <p className="mt-3 text-sm leading-relaxed">{ev.description}</p>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <Meta label="Track" value={`#${ev.trackId}`} />
              <Meta label="Severity" value={ev.severity} />
              <Meta label="Concept" value={ev.concept} />
              <Meta label="Verdict" value={ev.verdict} />
            </div>
            <div className="mt-2 flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground">
              <MapPin className="h-3 w-3" /> {ev.telemetry.lat.toFixed(5)}, {ev.telemetry.lon.toFixed(5)} · {ev.telemetry.alt.toFixed(0)}m
            </div>
          </div>

          {/* right: scoped chat */}
          <div className="flex h-[26rem] flex-col">
            <div className="border-b border-border px-4 py-2 text-xs text-muted-foreground">
              Scoped to this moment · grounded to track #{ev.trackId}
            </div>
            <ScrollArea className="flex-1 px-4 py-3">
              <div ref={scrollRef} className="flex flex-col gap-3">
                {messages.length === 0 && (
                  <div className="flex flex-col gap-1.5">
                    <p className="text-xs text-muted-foreground">Ask about this event:</p>
                    {suggestions.map((s) => (
                      <button
                        key={s}
                        onClick={() => ask(s)}
                        className="rounded-md border border-border bg-card px-2.5 py-1.5 text-left text-xs transition-colors hover:border-primary/40"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
                {messages.map((m) => (
                  <div key={m.id} className={m.role === "user" ? "self-end" : "self-start"}>
                    <div
                      className={
                        m.role === "user"
                          ? "max-w-[15rem] rounded-lg rounded-br-sm bg-primary px-3 py-2 text-xs text-primary-foreground"
                          : "max-w-[16rem] rounded-lg rounded-bl-sm border border-border bg-card px-3 py-2 text-xs"
                      }
                    >
                      {m.content}
                      {m.grounded && (
                        <div className="mt-1 font-mono text-[9px] text-muted-foreground">↳ {m.grounded}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
            <form
              className="flex items-center gap-2 border-t border-border p-2.5"
              onSubmit={(e) => {
                e.preventDefault();
                ask(q);
              }}
            >
              <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a question…" className="h-8" />
              <Button size="icon" type="submit" className="h-8 w-8 shrink-0">
                <Send />
              </Button>
            </form>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-card px-2.5 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-medium capitalize">{value}</div>
    </div>
  );
}
