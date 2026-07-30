import { create } from "zustand";
import * as api from "@/lib/api";
import type {
  ChatMessage,
  Concept,
  EventItem,
  Priority,
  Stream,
  TrackedObject,
  Telemetry,
  Verdict,
} from "@/lib/types";

// Streams stay client-side mock data — the backend has no concept of
// multiple streams yet (a single STREAM_URL, see backend/app/config.py).
const streams: Stream[] = [
  { id: "s1", name: "Sector 7 · North perimeter", source: "udp://239.1.1.1:5600", sensor: "EO/IR", online: true },
  { id: "s2", name: "Sector 3 · Camp overwatch", source: "udp://239.1.1.2:5600", sensor: "IR", online: true },
  { id: "s3", name: "Convoy escort · UAV-04", source: "udp://127.0.0.1:5601", sensor: "EO", online: false },
];

interface State {
  streams: Stream[];
  selectedStream: string;
  concepts: Concept[];
  frame: string | null; // latest decoded video frame, JPEG data URL — see lib/wsFeed.ts
  tracks: TrackedObject[];
  telemetry: Telemetry;
  events: EventItem[];
  selectedEvent: string | null;
  playing: boolean;
  now: number;
  chats: Record<string, ChatMessage[]>;

  selectStream: (id: string) => void;
  setFrame: (dataUrl: string) => void;
  setTracks: (t: TrackedObject[]) => void;
  setTelemetry: (t: Telemetry) => void;
  addEvent: (e: EventItem) => void;
  setVerdict: (id: string, v: Verdict) => void;
  openEvent: (id: string | null) => void;
  markDeepAnalyzed: (id: string) => void;
  loadConcepts: () => Promise<void>;
  addConcept: (label: string) => Promise<void>;
  toggleConcept: (id: string) => Promise<void>;
  cycleConceptPriority: (id: string) => Promise<void>;
  setPlaying: (p: boolean) => void;
  tickNow: () => void;
  pushChat: (eventId: string, m: ChatMessage) => void;
}

const nextPriority: Record<Priority, Priority> = { alert: "watch", watch: "ignore", ignore: "alert" };

export const useStore = create<State>((set, get) => ({
  streams,
  selectedStream: "s1",
  concepts: [],
  frame: null,
  tracks: [],
  telemetry: { lat: 28.6139, lon: 77.209, alt: 1200, hdg: 90, speed: 45 },
  events: [],
  selectedEvent: null,
  playing: true,
  now: Date.now(),
  chats: {},

  selectStream: (id) => set({ selectedStream: id }),
  setFrame: (frame) => set({ frame }),
  setTracks: (tracks) => set({ tracks }),
  setTelemetry: (telemetry) => set({ telemetry }),
  addEvent: (e) => set((s) => ({ events: [e, ...s.events].slice(0, 60) })),
  setVerdict: (id, verdict) =>
    set((s) => ({ events: s.events.map((e) => (e.id === id ? { ...e, verdict } : e)) })),
  openEvent: (selectedEvent) => set({ selectedEvent }),
  markDeepAnalyzed: (id) =>
    set((s) => ({ events: s.events.map((e) => (e.id === id ? { ...e, deepAnalyzed: true, tier: 3 } : e)) })),

  // Concepts are the backend's watch-list (backend/app/store.py) — every
  // mutation round-trips through lib/api.ts and reconciles from the server's
  // response, rather than guessing the result client-side, so this stays the
  // source of truth (e.g. exemplar counts, server-assigned colors/ids).
  loadConcepts: async () => {
    const concepts = await api.fetchConcepts();
    set({ concepts });
  },
  addConcept: async (label) => {
    const c = await api.createConcept(label);
    set((s) => ({ concepts: [...s.concepts, c] }));
  },
  toggleConcept: async (id) => {
    const current = get().concepts.find((c) => c.id === id);
    if (!current) return;
    const updated = await api.updateConcept(id, { enabled: !current.enabled });
    set((s) => ({ concepts: s.concepts.map((c) => (c.id === id ? updated : c)) }));
  },
  cycleConceptPriority: async (id) => {
    const current = get().concepts.find((c) => c.id === id);
    if (!current) return;
    const updated = await api.updateConcept(id, { priority: nextPriority[current.priority] });
    set((s) => ({ concepts: s.concepts.map((c) => (c.id === id ? updated : c)) }));
  },

  setPlaying: (playing) => set({ playing }),
  tickNow: () => set({ now: Date.now() }),
  pushChat: (eventId, m) =>
    set((s) => ({ chats: { ...s.chats, [eventId]: [...(s.chats[eventId] ?? []), m] } })),
}));
