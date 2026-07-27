import { create } from "zustand";
import type {
  ChatMessage,
  Concept,
  EventItem,
  Stream,
  TrackedObject,
  Telemetry,
  Verdict,
} from "@/lib/types";

const initialConcepts: Concept[] = [
  { id: "c_person", label: "person", color: "#38bdf8", priority: "alert", exemplars: 4, enabled: true },
  { id: "c_vehicle", label: "vehicle", color: "#a78bfa", priority: "watch", exemplars: 3, enabled: true },
  { id: "c_tent", label: "tent", color: "#34d399", priority: "watch", exemplars: 5, enabled: true },
  { id: "c_fire", label: "fire", color: "#f87171", priority: "alert", exemplars: 2, enabled: true },
];

const streams: Stream[] = [
  { id: "s1", name: "Sector 7 · North perimeter", source: "udp://239.1.1.1:5600", sensor: "EO/IR", online: true },
  { id: "s2", name: "Sector 3 · Camp overwatch", source: "udp://239.1.1.2:5600", sensor: "IR", online: true },
  { id: "s3", name: "Convoy escort · UAV-04", source: "udp://127.0.0.1:5601", sensor: "EO", online: false },
];

interface State {
  streams: Stream[];
  selectedStream: string;
  concepts: Concept[];
  tracks: TrackedObject[];
  telemetry: Telemetry;
  events: EventItem[];
  selectedEvent: string | null;
  playing: boolean;
  now: number;
  chats: Record<string, ChatMessage[]>;

  selectStream: (id: string) => void;
  setTracks: (t: TrackedObject[]) => void;
  setTelemetry: (t: Telemetry) => void;
  addEvent: (e: EventItem) => void;
  setVerdict: (id: string, v: Verdict) => void;
  openEvent: (id: string | null) => void;
  markDeepAnalyzed: (id: string) => void;
  addConcept: (c: Concept) => void;
  toggleConcept: (id: string) => void;
  cycleConceptPriority: (id: string) => void;
  setPlaying: (p: boolean) => void;
  tickNow: () => void;
  pushChat: (eventId: string, m: ChatMessage) => void;
}

export const useStore = create<State>((set) => ({
  streams,
  selectedStream: "s1",
  concepts: initialConcepts,
  tracks: [],
  telemetry: { lat: 28.6139, lon: 77.209, alt: 1200, hdg: 90, speed: 45 },
  events: [],
  selectedEvent: null,
  playing: true,
  now: Date.now(),
  chats: {},

  selectStream: (id) => set({ selectedStream: id }),
  setTracks: (tracks) => set({ tracks }),
  setTelemetry: (telemetry) => set({ telemetry }),
  addEvent: (e) => set((s) => ({ events: [e, ...s.events].slice(0, 60) })),
  setVerdict: (id, verdict) =>
    set((s) => ({ events: s.events.map((e) => (e.id === id ? { ...e, verdict } : e)) })),
  openEvent: (selectedEvent) => set({ selectedEvent }),
  markDeepAnalyzed: (id) =>
    set((s) => ({ events: s.events.map((e) => (e.id === id ? { ...e, deepAnalyzed: true, tier: 3 } : e)) })),
  addConcept: (c) => set((s) => ({ concepts: [...s.concepts, c] })),
  toggleConcept: (id) =>
    set((s) => ({ concepts: s.concepts.map((c) => (c.id === id ? { ...c, enabled: !c.enabled } : c)) })),
  cycleConceptPriority: (id) =>
    set((s) => ({
      concepts: s.concepts.map((c) =>
        c.id === id
          ? { ...c, priority: c.priority === "alert" ? "watch" : c.priority === "watch" ? "ignore" : "alert" }
          : c,
      ),
    })),
  setPlaying: (playing) => set({ playing }),
  tickNow: () => set({ now: Date.now() }),
  pushChat: (eventId, m) =>
    set((s) => ({ chats: { ...s.chats, [eventId]: [...(s.chats[eventId] ?? []), m] } })),
}));
