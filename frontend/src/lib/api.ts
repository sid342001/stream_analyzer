/**
 * REST client for the backend's context-authoring API (backend/app/main.py).
 * Routed through vite's `/api` proxy in dev (see vite.config.ts) so this
 * works the same whether the backend runs on localhost or in a container.
 */
import type { ChatMessage, Concept, Priority, TrackSnapshot } from "./types";

const BASE = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export function fetchConcepts(): Promise<Concept[]> {
  return fetch(`${BASE}/concepts`).then(json<Concept[]>);
}

export function createConcept(label: string, priority: Priority = "watch"): Promise<Concept> {
  return fetch(`${BASE}/concepts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, priority }),
  }).then(json<Concept>);
}

export function updateConcept(
  id: string,
  patch: { enabled?: boolean; priority?: Priority },
): Promise<Concept> {
  return fetch(`${BASE}/concepts/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  }).then(json<Concept>);
}

/** Draw a box on a reference image (a hi-res still or an uploaded file) to
 * author a new exemplar for a concept — box is [x1,y1,x2,y2] in that image's
 * own pixel coordinates. Backs both context-authoring entry points. */
export function addExemplar(
  conceptId: string,
  image: Blob,
  box: [number, number, number, number],
): Promise<Concept> {
  const form = new FormData();
  form.append("image", image);
  form.append("box", JSON.stringify(box));
  return fetch(`${BASE}/concepts/${conceptId}/exemplars`, { method: "POST", body: form }).then(
    json<Concept>,
  );
}

/** One-shot full-ingest-resolution frame for drawing a new exemplar box on —
 * deliberately not the (downscaled) live display frame; see
 * backend/app/main.py's get_hires_frame docstring. Rejects (503) if the
 * stream isn't currently live. */
export function fetchHiresFrame(): Promise<string> {
  return fetch(`${BASE}/frame/hires`)
    .then(json<{ image: string }>)
    .then((r) => r.image);
}

/** Swap the live UDP source at runtime. Connection outcome arrives async via
 * the WS `status` message, not this response — see lib/wsFeed.ts. */
export function setStream(host: string, port: number): Promise<void> {
  return fetch(`${BASE}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: `udp://${host}:${port}` }),
  }).then((res) => {
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  });
}

/** Fresh, on-demand crop of a currently-live tracked object — see
 * backend/app/main.py's get_track_snapshot. 404s if the object has since
 * coasted out of the tracker; callers should show that as a clean error,
 * not retry silently (the object is just gone). */
export function fetchTrackSnapshot(trackId: number): Promise<TrackSnapshot> {
  return fetch(`${BASE}/tracks/${trackId}/snapshot`)
    .then(json<{ image: string; box: number[]; concept: string; conceptColor: string }>)
    .then((r) => ({ trackId, ...r }));
}

/** Tier-3 chat about a specific event's evidence image. `history` is the
 * prior turns already in the store (see useStore's `chats`) — the backend
 * is stateless and needs them resent every call to hold a conversation. */
export function ask(
  question: string,
  image: string | null | undefined,
  box: number[] | undefined,
  concept: string,
  conceptColor: string,
  history: Pick<ChatMessage, "role" | "content">[],
): Promise<string> {
  return fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      image: image ?? null,
      box: box ?? [],
      concept,
      conceptColor,
      history: history.map(({ role, content }) => ({ role, content })),
    }),
  })
    .then(json<{ answer: string }>)
    .then((r) => r.answer);
}
