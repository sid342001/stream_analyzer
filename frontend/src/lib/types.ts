export type Priority = "alert" | "watch" | "ignore";
export type Severity = "critical" | "warning" | "info";
export type Verdict = "pending" | "dismissed" | "confirmed" | "escalated";
export type Tier = 1 | 2 | 3;

export interface Concept {
  id: string;
  label: string;
  color: string; // hex, used for overlay + dot
  priority: Priority;
  exemplars: number;
  enabled: boolean;
}

export interface TrackedObject {
  id: number;
  concept: string;
  color: string;
  // normalized 0..1 within the frame
  x: number;
  y: number;
  w: number;
  h: number;
  vx: number;
  vy: number;
  score: number;
  trail: { x: number; y: number }[];
}

export interface Telemetry {
  lat: number;
  lon: number;
  alt: number; // m MSL
  hdg: number; // deg
  speed: number; // m/s
}

export interface EventItem {
  id: string;
  ts: number;
  concept: string;
  conceptColor: string;
  severity: Severity;
  kind: string; // "first appearance" | "approach" | "low confidence"
  description: string;
  verdict: Verdict;
  tier: Tier;
  deepAnalyzed: boolean;
  // null for scene-overview events (see backend/app/pipeline.py's
  // _queue_scene_analysis) — a periodic whole-frame narrative isn't about
  // any one tracked object. Only set for events sourced from a specific track.
  trackId: number | null;
  telemetry: Telemetry;
  // Base64 JPEG data URL of the analyzed frame — the whole scene for a
  // scene-overview event, or a margin crop for a track-scoped one (not the
  // full frame either way — bounds memory) — and that image's own box
  // coordinates, see backend/app/schemas.py. Undefined/null/empty when
  // there's no object box (scene notes) or the crop failed; render a
  // fallback rather than assuming presence.
  image?: string | null;
  box?: number[];
}

// Response shape of GET /api/tracks/{id}/snapshot — a fresh, on-demand crop
// of a currently-live object, for the Objects tab's "query" flow. Not
// retained anywhere server-side; fetched fresh every time it's opened.
export interface TrackSnapshot {
  trackId: number;
  image: string;
  box: number[];
  concept: string;
  conceptColor: string;
}

// Mirrors backend/app/pipeline.py's "status" WS message — connection state
// of the single live UDP source (see backend/app/config.py's stream_url).
export interface StreamStatus {
  connected: boolean;
  url: string;
  error: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  grounded?: string; // e.g. "track #7 · 14:22:07"
}
