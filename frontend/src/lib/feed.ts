/**
 * Simulated live feed — the seam where the real backend plugs in.
 *
 * Today this drives the console from an in-browser simulation so the UI is
 * demoable with no backend. To go live, replace `MockFeed` with a `WsFeed` that
 * opens `/ws`, and emit the same events (`onFrame`, `onEvent`, `onTelemetry`).
 * The rest of the app depends only on the `Feed` interface below.
 */
import type { EventItem, Severity, StreamStatus, TrackedObject, Telemetry } from "./types";

export interface FeedHandlers {
  onTracks: (tracks: TrackedObject[]) => void;
  onTelemetry: (t: Telemetry) => void;
  onEvent: (e: EventItem) => void;
  /** JPEG data URL for one decoded video frame. Optional: MockFeed has no
   * real video to simulate, only WsFeed calls this. */
  onFrame?: (dataUrl: string) => void;
  /** Live-source connection state. Optional: MockFeed has no real source to
   * report on, only WsFeed calls this. */
  onStatus?: (s: StreamStatus) => void;
}

export interface Feed {
  start(handlers: FeedHandlers): void;
  stop(): void;
  setEnabledConcepts(labels: string[]): void;
}

const CONCEPTS: { label: string; color: string }[] = [
  { label: "person", color: "#38bdf8" },
  { label: "vehicle", color: "#a78bfa" },
  { label: "tent", color: "#34d399" },
  { label: "fire", color: "#f87171" },
];

const EARTH_R = 6378137;
function offset(lat: number, lon: number, n: number, e: number) {
  const dlat = (n / EARTH_R) * (180 / Math.PI);
  const dlon = (e / (EARTH_R * Math.cos((lat * Math.PI) / 180))) * (180 / Math.PI);
  return { lat: lat + dlat, lon: lon + dlon };
}

let uid = 1;

export class MockFeed implements Feed {
  private raf = 0;
  private evTimer = 0;
  private t0 = performance.now();
  private tracks: TrackedObject[] = [];
  private enabled = new Set(CONCEPTS.map((c) => c.label));
  private handlers?: FeedHandlers;
  private centerLat = 28.6139;
  private centerLon = 77.209;

  start(handlers: FeedHandlers) {
    this.handlers = handlers;
    this.tracks = this.seed();
    const loop = () => {
      this.step();
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
    this.evTimer = window.setInterval(() => this.maybeEvent(), 3200);
  }

  stop() {
    cancelAnimationFrame(this.raf);
    clearInterval(this.evTimer);
  }

  setEnabledConcepts(labels: string[]) {
    this.enabled = new Set(labels);
  }

  private seed(): TrackedObject[] {
    const mk = (concept: string, x: number, y: number): TrackedObject => {
      const c = CONCEPTS.find((k) => k.label === concept)!;
      return {
        id: uid++,
        concept,
        color: c.color,
        x,
        y,
        w: concept === "vehicle" ? 0.12 : concept === "tent" ? 0.1 : 0.05,
        h: concept === "vehicle" ? 0.08 : concept === "tent" ? 0.09 : 0.09,
        vx: (Math.random() - 0.5) * 0.0006,
        vy: (Math.random() - 0.5) * 0.0006,
        score: 0.72 + Math.random() * 0.26,
        trail: [],
      };
    };
    return [
      mk("tent", 0.62, 0.55),
      mk("tent", 0.7, 0.62),
      mk("vehicle", 0.3, 0.35),
      mk("person", 0.2, 0.72),
      mk("person", 0.5, 0.5),
    ];
  }

  private step() {
    const t = (performance.now() - this.t0) / 1000;
    for (const o of this.tracks) {
      // people wander toward the tent cluster; others drift
      if (o.concept === "person") {
        o.vx += (0.66 - o.x) * 0.000004;
        o.vy += (0.58 - o.y) * 0.000004;
      }
      o.x = Math.min(0.94, Math.max(0.06, o.x + o.vx));
      o.y = Math.min(0.92, Math.max(0.08, o.y + o.vy));
      o.vx *= 0.985;
      o.vy *= 0.985;
      o.trail.push({ x: o.x, y: o.y });
      if (o.trail.length > 40) o.trail.shift();
    }
    const visible = this.tracks.filter((o) => this.enabled.has(o.concept));
    this.handlers?.onTracks(visible);

    const orbit = offset(this.centerLat, this.centerLon, 1500 * Math.cos(t * 0.03), 1500 * Math.sin(t * 0.03));
    this.handlers?.onTelemetry({
      lat: orbit.lat,
      lon: orbit.lon,
      alt: 1200 + 15 * Math.sin(t * 0.3),
      hdg: (90 + t * 1.7) % 360,
      speed: 45,
    });
  }

  private maybeEvent() {
    if (!this.handlers) return;
    const people = this.tracks.filter((o) => o.concept === "person" && this.enabled.has("person"));
    const tents = this.tracks.filter((o) => o.concept === "tent");
    if (!people.length || !tents.length) return;

    // subject approaching cluster → the hero event
    let best: { p: TrackedObject; d: number } | null = null;
    for (const p of people) {
      for (const t of tents) {
        const d = Math.hypot(p.x - t.x, p.y - t.y);
        if (!best || d < best.d) best = { p, d };
      }
    }
    if (!best) return;

    const roll = Math.random();
    let severity: Severity = "info";
    let kind = "new detection";
    let description = "";
    const orbit = offset(this.centerLat, this.centerLon, Math.random() * 1000, Math.random() * 1000);
    const tele: Telemetry = { lat: orbit.lat, lon: orbit.lon, alt: 1200, hdg: 92, speed: 45 };

    if (best.d < 0.16) {
      severity = "critical";
      kind = "approach";
      description = `Person (track #${best.p.id}) closing on tent cluster — ${Math.round(best.d * 1000)} px separation.`;
    } else if (roll < 0.4) {
      severity = "warning";
      kind = "low confidence";
      const c = CONCEPTS[Math.floor(Math.random() * 2)];
      description = `Low-confidence ${c.label} at frame edge — flagged for adjudication.`;
    } else {
      severity = "info";
      kind = "first appearance";
      const c = CONCEPTS[Math.floor(Math.random() * CONCEPTS.length)];
      if (!this.enabled.has(c.label)) return;
      description = `First appearance of ${c.label} in this sector.`;
    }

    const concept = kind === "approach" ? "person" : CONCEPTS[Math.floor(Math.random() * CONCEPTS.length)].label;
    const cc = CONCEPTS.find((k) => k.label === concept)!;
    const ev: EventItem = {
      id: `ev_${uid++}`,
      ts: Date.now(),
      concept,
      conceptColor: cc.color,
      severity,
      kind,
      description,
      verdict: "pending",
      tier: 2,
      deepAnalyzed: false,
      trackId: best.p.id,
      telemetry: tele,
    };
    this.handlers.onEvent(ev);
  }
}
