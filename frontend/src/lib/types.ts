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
  trackId: number;
  telemetry: Telemetry;
}

export interface Stream {
  id: string;
  name: string;
  source: string; // udp://...
  sensor: string; // EO / IR
  online: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  grounded?: string; // e.g. "track #7 · 14:22:07"
}
