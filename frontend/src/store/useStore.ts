import { create } from "zustand";
import * as api from "@/lib/api";
import type {
  ChatMessage,
  Concept,
  EventItem,
  Priority,
  StreamStatus,
  TrackedObject,
  TrackSnapshot,
  Telemetry,
  Verdict,
} from "@/lib/types";

interface State {
  concepts: Concept[];
  frame: string | null; // latest decoded video frame, JPEG data URL — see lib/wsFeed.ts
  tracks: TrackedObject[];
  telemetry: Telemetry;
  events: EventItem[];
  selectedEvent: string | null;
  // Live per-object query (Objects tab) — mutually exclusive with
  // selectedEvent; a click on either closes the other. trackSnapshot is
  // null while the fetch is in flight, trackSnapshotError set if it failed
  // (e.g. the object coasted out between the click and the response).
  selectedTrackId: number | null;
  trackSnapshot: TrackSnapshot | null;
  trackSnapshotError: string | null;
  playing: boolean;
  now: number;
  chats: Record<string, ChatMessage[]>;
  streamStatus: StreamStatus | null;

  setFrame: (dataUrl: string) => void;
  setTracks: (t: TrackedObject[]) => void;
  setTelemetry: (t: Telemetry) => void;
  addEvent: (e: EventItem) => void;
  setVerdict: (id: string, v: Verdict) => void;
  openEvent: (id: string | null) => void;
  openTrack: (id: number) => Promise<void>;
  closeTrack: () => void;
  markDeepAnalyzed: (id: string) => void;
  loadConcepts: () => Promise<void>;
  addConcept: (label: string) => Promise<Concept>;
  toggleConcept: (id: string) => Promise<void>;
  cycleConceptPriority: (id: string) => Promise<void>;
  setPlaying: (p: boolean) => void;
  tickNow: () => void;
  pushChat: (key: string, m: ChatMessage) => void;
  setStreamStatus: (s: StreamStatus) => void;
}

const nextPriority: Record<Priority, Priority> = { alert: "watch", watch: "ignore", ignore: "alert" };

export const useStore = create<State>((set, get) => ({
  concepts: [],
  frame: null,
  tracks: [],
  telemetry: { lat: 28.6139, lon: 77.209, alt: 1200, hdg: 90, speed: 45 },
  events: [],
  selectedEvent: null,
  selectedTrackId: null,
  trackSnapshot: null,
  trackSnapshotError: null,
  playing: true,
  now: Date.now(),
  chats: {},
  streamStatus: null,

  setFrame: (frame) => set({ frame }),
  setTracks: (tracks) => set({ tracks }),
  setTelemetry: (telemetry) => set({ telemetry }),
  addEvent: (e) => set((s) => ({ events: [e, ...s.events].slice(0, 60) })),
  setVerdict: (id, verdict) =>
    set((s) => ({ events: s.events.map((e) => (e.id === id ? { ...e, verdict } : e)) })),
  openEvent: (selectedEvent) =>
    set({ selectedEvent, selectedTrackId: null, trackSnapshot: null, trackSnapshotError: null }),
  openTrack: async (id) => {
    set({ selectedTrackId: id, trackSnapshot: null, trackSnapshotError: null, selectedEvent: null });
    try {
      const snapshot = await api.fetchTrackSnapshot(id);
      // Stale-response guard: the operator may have closed this or opened a
      // different track while the request was in flight.
      if (get().selectedTrackId === id) set({ trackSnapshot: snapshot });
    } catch (e) {
      if (get().selectedTrackId === id) {
        set({ trackSnapshotError: e instanceof Error ? e.message : "Failed to fetch snapshot" });
      }
    }
  },
  closeTrack: () => set({ selectedTrackId: null, trackSnapshot: null, trackSnapshotError: null }),
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
    return c;
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
  setStreamStatus: (streamStatus) => set({ streamStatus }),
}));
