import { useEffect, useRef } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TopBar } from "@/components/TopBar";
import { LeftRail } from "@/components/LeftRail";
import { CenterFeed } from "@/components/CenterFeed";
import { EventFeed } from "@/components/EventFeed";
import { TimelineStrip } from "@/components/TimelineStrip";
import { DetailView } from "@/components/DetailView";
import { ObjectQueryDialog } from "@/components/ObjectQueryDialog";
import { useStore } from "@/store/useStore";
import type { Feed } from "@/lib/feed";
import { WsFeed } from "@/lib/wsFeed";

export default function App() {
  const setFrame = useStore((s) => s.setFrame);
  const setTracks = useStore((s) => s.setTracks);
  const setTelemetry = useStore((s) => s.setTelemetry);
  const addEvent = useStore((s) => s.addEvent);
  const tickNow = useStore((s) => s.tickNow);
  const loadConcepts = useStore((s) => s.loadConcepts);
  const setStreamStatus = useStore((s) => s.setStreamStatus);
  const feedRef = useRef<Feed | null>(null);

  useEffect(() => {
    loadConcepts();
  }, [loadConcepts]);

  // live feed, backed by backend/app/main.py's /ws — see lib/wsFeed.ts
  useEffect(() => {
    const feed = new WsFeed();
    feedRef.current = feed;
    feed.start({
      onFrame: (f) => useStore.getState().playing && setFrame(f),
      onTracks: (t) => useStore.getState().playing && setTracks(t),
      onTelemetry: setTelemetry,
      onEvent: addEvent,
      onStatus: setStreamStatus,
    });
    const clock = setInterval(tickNow, 1000);
    return () => {
      feed.stop();
      clearInterval(clock);
    };
  }, [setFrame, setTracks, setTelemetry, addEvent, tickNow, setStreamStatus]);

  // keep the feed's enabled-concept set in sync with the watch-list
  const concepts = useStore((s) => s.concepts);
  useEffect(() => {
    feedRef.current?.setEnabledConcepts(concepts.filter((c) => c.enabled).map((c) => c.label));
  }, [concepts]);

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-screen flex-col overflow-hidden">
        <TopBar />
        <div className="flex min-h-0 flex-1">
          <LeftRail />
          <main className="min-w-0 flex-1 p-3">
            <CenterFeed />
          </main>
          <EventFeed />
        </div>
        <TimelineStrip />
      </div>
      <DetailView />
      <ObjectQueryDialog />
    </TooltipProvider>
  );
}
