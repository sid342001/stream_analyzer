/**
 * Live `Feed` implementation — connects to the backend's /ws (see
 * backend/app/main.py and backend/app/pipeline.py's docstring for the exact
 * wire shape). Replaces `MockFeed` in App.tsx.
 */
import type { Feed, FeedHandlers } from "./feed";

type WireMessage =
  | { type: "frame"; data: unknown }
  | { type: "tracks"; data: unknown }
  | { type: "telemetry"; data: unknown }
  | { type: "event"; data: unknown };

const RECONNECT_DELAY_MS = 2000;

export class WsFeed implements Feed {
  private ws: WebSocket | null = null;
  private handlers?: FeedHandlers;
  private stopped = false;
  private reconnectTimer = 0;

  start(handlers: FeedHandlers) {
    this.handlers = handlers;
    this.stopped = false;
    this.connect();
  }

  stop() {
    this.stopped = true;
    clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }

  setEnabledConcepts(_labels: string[]) {
    // No-op by design: enabling/disabling a concept goes straight to the
    // backend via lib/api.ts's `updateConcept` (see useStore.ts's
    // toggleConcept) the moment the operator flips the switch — the pipeline
    // reads concept_store.enabled() fresh every sampled frame, so there's
    // nothing further to signal here. Kept only to satisfy the `Feed`
    // interface, which MockFeed's simulation genuinely needs this for.
  }

  private connect() {
    if (this.stopped) return;
    // Dev mode connects straight to the backend, bypassing vite's proxy —
    // its WebSocket proxying repeatedly dropped the connection
    // ("socket hang up") once video frames pushed sustained throughput over
    // /ws (~1.2MB/s at 12fps), confirmed by a direct connection working
    // instantly where the proxied one kept failing. A production build (not
    // served by vite) falls back to same-origin /ws, e.g. behind a real
    // reverse proxy.
    const url = import.meta.env.DEV
      ? "ws://localhost:8000/ws"
      : `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
    this.ws = new WebSocket(url);

    this.ws.onmessage = (ev) => {
      let msg: WireMessage;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (msg.type) {
        case "frame":
          this.handlers?.onFrame?.(msg.data as string);
          break;
        case "tracks":
          this.handlers?.onTracks(msg.data as Parameters<FeedHandlers["onTracks"]>[0]);
          break;
        case "telemetry":
          this.handlers?.onTelemetry(msg.data as Parameters<FeedHandlers["onTelemetry"]>[0]);
          break;
        case "event":
          this.handlers?.onEvent(msg.data as Parameters<FeedHandlers["onEvent"]>[0]);
          break;
      }
    };

    this.ws.onclose = () => {
      if (!this.stopped) {
        this.reconnectTimer = window.setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
      }
    };
  }
}
