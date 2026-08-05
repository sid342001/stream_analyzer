import { useEffect, useRef, useState } from "react";
import { Crosshair, Send } from "lucide-react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useStore } from "@/store/useStore";
import * as api from "@/lib/api";

/** Live, on-demand query for a single currently-tracked object — opened
 *  from the Objects tab (LeftRail's sibling panel, see EventFeed.tsx).
 *
 *  Deliberately a separate component from DetailView rather than a
 *  generalization of it: a live query has no severity/verdict/telemetry/
 *  deep-analyzed workflow state (there's no "event" here, just a fresh
 *  snapshot + a question), so forcing both shapes through one component
 *  would mean DetailView growing a pile of conditionals for fields that
 *  don't apply to this mode. The chat interaction pattern (scroll/history/
 *  ask) and the image+box overlay math are intentionally the same as
 *  DetailView's, just re-expressed for this simpler data shape.
 */
export function ObjectQueryDialog() {
  const selectedTrackId = useStore((s) => s.selectedTrackId);
  const trackSnapshot = useStore((s) => s.trackSnapshot);
  const trackSnapshotError = useStore((s) => s.trackSnapshotError);
  const closeTrack = useStore((s) => s.closeTrack);
  const chats = useStore((s) => s.chats);
  const pushChat = useStore((s) => s.pushChat);

  const [q, setQ] = useState("");
  const [asking, setAsking] = useState(false);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setImgSize(null);
  }, [trackSnapshot?.image]);

  if (selectedTrackId == null) return null;

  const chatKey = `track_${selectedTrackId}`;
  const messages = chats[chatKey] ?? [];

  const ask = async (text: string) => {
    const question = text.trim();
    if (!question || !trackSnapshot || asking) return;
    const history = messages.map(({ role, content }) => ({ role, content }));
    pushChat(chatKey, { id: `m_${Date.now()}`, role: "user", content: question });
    setQ("");
    setAsking(true);
    try {
      const answer = await api.ask(
        question, trackSnapshot.image, trackSnapshot.box, trackSnapshot.concept,
        trackSnapshot.conceptColor, history,
      );
      pushChat(chatKey, { id: `a_${Date.now()}`, role: "assistant", content: answer });
    } catch (e) {
      pushChat(chatKey, {
        id: `err_${Date.now()}`,
        role: "assistant",
        content: e instanceof Error ? e.message : "Failed to reach the analysis model.",
      });
    } finally {
      setAsking(false);
      scrollRef.current?.scrollTo({ top: 1e9 });
    }
  };

  return (
    <Dialog open={selectedTrackId != null} onOpenChange={(o) => !o && closeTrack()}>
      <DialogContent>
        <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-3">
          <Crosshair className="h-4 w-4 text-primary" />
          <DialogTitle>Query object #{selectedTrackId}</DialogTitle>
        </div>

        <div className="grid grid-cols-[1.1fr_1fr] gap-0">
          {/* left: fresh snapshot */}
          <div className="border-r border-border p-4">
            <div className="relative overflow-hidden rounded-md border border-border bg-[#0a1017]">
              {trackSnapshotError ? (
                <div className="relative aspect-video">
                  <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-[11px] text-alert">
                    {trackSnapshotError}
                  </div>
                </div>
              ) : trackSnapshot ? (
                <>
                  {/* eslint-disable-next-line jsx-a11y/alt-text */}
                  <img
                    src={trackSnapshot.image}
                    className="block w-full"
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
                    }}
                  />
                  {imgSize && trackSnapshot.box.length === 4 && (
                    <div
                      className="absolute rounded-sm border-2"
                      style={{
                        borderColor: trackSnapshot.conceptColor,
                        background: trackSnapshot.conceptColor + "22",
                        left: `${(trackSnapshot.box[0] / imgSize.w) * 100}%`,
                        top: `${(trackSnapshot.box[1] / imgSize.h) * 100}%`,
                        width: `${((trackSnapshot.box[2] - trackSnapshot.box[0]) / imgSize.w) * 100}%`,
                        height: `${((trackSnapshot.box[3] - trackSnapshot.box[1]) / imgSize.h) * 100}%`,
                      }}
                    />
                  )}
                </>
              ) : (
                <div className="relative aspect-video">
                  <div className="grid-noise absolute inset-0 opacity-40" />
                  <div className="absolute inset-0 flex items-center justify-center text-[11px] text-muted-foreground">
                    Capturing a fresh frame…
                  </div>
                </div>
              )}
            </div>
            {trackSnapshot && (
              <div className="mt-3 flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: trackSnapshot.conceptColor }} />
                <span className="text-xs font-medium capitalize">{trackSnapshot.concept}</span>
              </div>
            )}
          </div>

          {/* right: live chat */}
          <div className="flex h-[26rem] flex-col">
            <div className="border-b border-border px-4 py-2 text-xs text-muted-foreground">
              Live query · grounded to this exact frame
            </div>
            <ScrollArea className="flex-1 px-4 py-3">
              <div ref={scrollRef} className="flex flex-col gap-3">
                {messages.length === 0 && trackSnapshot && (
                  <p className="text-xs text-muted-foreground">Ask about this object…</p>
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
              <Input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder={trackSnapshot ? "Ask a question…" : "Waiting for snapshot…"}
                disabled={!trackSnapshot || asking}
                className="h-8"
              />
              <Button size="icon" type="submit" className="h-8 w-8 shrink-0" disabled={!trackSnapshot || asking}>
                <Send />
              </Button>
            </form>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
