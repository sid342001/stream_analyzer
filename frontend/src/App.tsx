import { useEffect, useRef } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TopBar } from "@/components/TopBar";
import { LeftRail } from "@/components/LeftRail";
import { CenterFeed } from "@/components/CenterFeed";
import { EventFeed } from "@/components/EventFeed";
import { TimelineStrip } from "@/components/TimelineStrip";
import { DetailView } from "@/components/DetailView";
import { useStore } from "@/store/useStore";
import { MockFeed } from "@/lib/feed";

export default function App() {
  const setTracks = useStore((s) => s.setTracks);
  const setTelemetry = useStore((s) => s.setTelemetry);
  const addEvent = useStore((s) => s.addEvent);
  const tickNow = useStore((s) => s.tickNow);
  const feedRef = useRef<MockFeed | null>(null);

  // start the (simulated) live feed — swap MockFeed for a WsFeed to go live
  useEffect(() => {
    const feed = new MockFeed();
    feedRef.current = feed;
    feed.start({
      onTracks: (t) => useStore.getState().playing && setTracks(t),
      onTelemetry: setTelemetry,
      onEvent: addEvent,
    });
    const clock = setInterval(tickNow, 1000);
    return () => {
      feed.stop();
      clearInterval(clock);
    };
  }, [setTracks, setTelemetry, addEvent, tickNow]);

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
    </TooltipProvider>
  );
}
